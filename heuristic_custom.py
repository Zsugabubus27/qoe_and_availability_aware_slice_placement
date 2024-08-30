import networkx as nx
import numpy as np
import cspy
import copy
import itertools


def is_S_D_connected(G, s, d):
    """
    Check if there is any path between S and D in Graph G
    """
    try:
        # Check if there's a path between s and d
        return nx.has_path(G, s, d)
    except nx.NetworkXError:
        # If s or d are not in the graph, return False
        return False

def max_avail_path(G, src, dst, max_delay):
    G = copy.deepcopy(G)
    if not is_S_D_connected(G, src, dst):
        return None, None, None
    # Set the resource array to be the delay, since we have a e2e delay constraint
    # And the weight to be availability
    for n, m, data in G.edges(data=True):
        G[n][m]['res_cost'] = np.array([  1, data['delay']  ])
        G[n][m]['weight'] = np.abs(np.log(data['availability'])) 
    
    # Create a graph that is suitable for the problem
    G = nx.DiGraph(G.copy(), directed=True, n_res=2)
    G = nx.relabel_nodes(G, mapping={src : 'Source', dst : 'Sink'})
    # Set resource boundaries
    max_res, min_res = [len(G.nodes), max_delay], [0, 0]
    # init algorithm
    bidirec = cspy.BiDirectional(G, max_res, min_res)
    # Call and query attributes
    bidirec.run()
    found_path = bidirec.path
    
    if found_path is not None:
        found_path[0] = src
        found_path[-1] = dst
        path_cost = bidirec.total_cost
        avail_prod = np.exp(-1*bidirec.total_cost)
        return found_path, path_cost, avail_prod
    else: 
        return None, None, None

def k_shortest_paths(G, s, d, K, max_delay):
    # List of shortest paths
    shortest_paths = []
    
    # Priority queue to store potential shortest paths
    potential_paths = []
    
    # Find the first shortest path
    best_path, best_cost, best_avail_prod = max_avail_path(G, s, d, max_delay)
    if best_path is None:
        return shortest_paths
    shortest_paths.append({'path': best_path, 'cost' : best_cost, 'availability': best_avail_prod})
    for k in range(1, K):
        last_path = shortest_paths[-1]['path']
        for idx in range(len(last_path) - 1):
            # Spur node is the i-th node in the last found shortest path
            spur_node = last_path[idx]
            # Root path is the path before the spur node
            root_path = last_path[:idx + 1]

            # Create a new graph by removing edges and nodes that have been used
            # in the previous k-1 shortest paths sharing the same root path
            G_modified = modify_graph(G, shortest_paths, root_path, idx)
            
            # Calculate the remaining delay 
            _, _, root_path_delay = path_attributes(root_path, G)
            remaining_delay = max_delay - root_path_delay
            
            # Find the spur path from the spur node to the destination
            spur_path, spur_cost, spur_avail_prod = max_avail_path(G_modified, spur_node, d, remaining_delay)
            
            
            # If no spur path is found, continue to the next spur node
            if spur_path is None:
                continue
            
            # Combine the root path and spur path to form the new candidate path
            candidate_path = root_path + spur_path[1:]
            candidate_avail, candidate_cost, _ = path_attributes(candidate_path, G)
            # Add the candidate path to the potential paths list
            potential_paths.append({'path': candidate_path, 'cost' : candidate_cost, 
                                    'availability': candidate_avail})
        
        # If there are no more potential paths, break out of the loop
        if not potential_paths:
            break
        
        # Sort the potential paths based on their total length (or cost)
        potential_paths.sort(key=lambda path_dict: path_dict['cost'])
        
        # The next shortest path is the one with the smallest length
        next_shortest_path = potential_paths.pop(0)
        shortest_paths.append(next_shortest_path)

    shortest_paths.sort(key=lambda path_dict: path_dict['cost'])
    return shortest_paths

def modify_graph(G, shortest_paths, root_path, idx):
    """
    According to Yen's algorithm we remove edges and nodes such that:
        - Remove the links that are part of the previous shortest paths which share the same root path.
        - Remove the nodes from the root path except the spur node
    """
    modified_graph = G.copy()
    # Remove edges
    edges_to_remove = set()
    for path in (p['path'] for p in shortest_paths):
        if len(path) > idx and (path[:len(root_path)] == root_path):
            u, v = path[idx], path[idx + 1]
            edges_to_remove.add( (u, v) )
    modified_graph.remove_edges_from(edges_to_remove)
    # Remove the nodes from the root path
    nodes_to_remove = set( root_path[n] for n in range(len(root_path) - 1))
    modified_graph.remove_nodes_from(nodes_to_remove)
    return modified_graph

def path_attributes(path, G):
    avail_prod = 1
    cost_sum = 0
    delay = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        avail = G[u][v]['availability']
        avail_prod *= avail
        cost_sum += np.abs(np.log(avail))
        delay += G[u][v]['delay']
    return avail_prod, cost_sum, delay


class HeuristicCustom():
    def __init__(self, G_sub, SFC_dict, method):
        if method not in ["resource_saving", "node_disjoint", "link_disjoint"]:
            raise ValueError('Method must be in ["resource_saving", "node_disjoint", "link_disjoint"]')
        self.method = method
        self.G_sub = copy.deepcopy(G_sub)
        self.SFC_dict = SFC_dict
        
        self.phys_nodes_sorted = sorted(list(self.G_sub.nodes(data=True)), 
                                   key=lambda x: x[1]['availability'], reverse=True)
        self.phys_nodes_sorted = [x[0] for x in self.phys_nodes_sorted]
        self.phys_links = list(self.G_sub.edges)

        self.resources = list(self.G_sub.nodes[self.phys_nodes_sorted[0]]['capacity'].keys())
        self.slices = list(self.SFC_dict.keys())
        self.v_bandwidth_demand = {(i, u, v): self.SFC_dict[i][u][v]['bandwidth'] 
                                   for i in self.slices for (u, v) in self.SFC_dict[i].edges}
        self.v_resource_demand = {(i, v, p): self.SFC_dict[i].nodes[v]['demand'][p] 
                                  for i in self.slices for v in self.SFC_dict[i].nodes for p in self.resources}
        
        self.phys_resource_capacity = {(n, p): self.G_sub.nodes[n]['capacity'][p] 
                                       for n in self.phys_nodes_sorted for p in self.resources}
        self.delay_req = {i: self.SFC_dict[i].graph['max_delay'] for i in self.slices}
        self.availability_req = {i: self.SFC_dict[i].graph['min_availability'] for i in self.slices}
        self.max_groups = {i: self.SFC_dict[i].graph['max_placement_groups'] for i in self.slices}
        self.availability_node = {n: self.G_sub.nodes[n]['availability'] for n in self.phys_nodes_sorted}
        self.availability_link = {(i, j): self.G_sub.edges[i, j]['availability'] for (i, j) in self.phys_links}

    def calc_slice_availability(self, placements, i):
        # Get all relevant mappings:
        p_groups = [val for (_i, _), val in placements.items() if _i == i if val is not None]
        # Extract all the nodes and edges from the pgroups
        used_nodes_links = dict()
        for g, mapping in enumerate(p_groups, 1):
            used_nodes_links[g] = dict()
            used_nodes_links[g]['node'] = {item['node'] for v, item in mapping.items()}
            edges = [item['edge'] for v, item in mapping.items()]
            edges = list(itertools.chain.from_iterable(edges))
            edges = {(edges[i], edges[i+1]) for i in range(len(edges) - 1) if edges[i] != edges[i+1]}
            used_nodes_links[g]['edge'] = edges
        
        # Calculate the slice availability
        allS = list(itertools.chain.from_iterable(itertools.combinations(used_nodes_links.keys(), r) 
                                                  for r in range(1, len(used_nodes_links)+1)))
        availabilities = []
        for S in allS:
            sign = (-1)**(len(S)+1)
            nodes = set.union(*(v['node'] for k, v in used_nodes_links.items() if k in S) )
            edges = set.union(*(v['edge'] for k, v in used_nodes_links.items() if k in S) )
            node_avail_prod = [self.availability_node[n] for n in nodes]
            node_avail_prod = np.prod(node_avail_prod)
            edge_avail_prod = [self.availability_link.get((i, j)) or self.availability_link.get((j, i)) for i, j in edges]
            edge_avail_prod = np.prod(edge_avail_prod)
            availabilities.append(sign*node_avail_prod*edge_avail_prod)
        avail = sum(availabilities)
        return avail
    
    def _addMapping(self, G, vnf_mapping, i, vnf, node, path, vnf_list, idx_v, idx_n, slice_placement):
        if idx_v == 0:
            edge = []
        else:
            prev_v_node = vnf_mapping[ vnf_list[idx_v-1] ]["node"]
            edge_from_idx = path.index(prev_v_node)
            edge_to_idx = idx_n + 1
            edge = path[edge_from_idx : edge_to_idx]
        # Add mapping to the mapping dict
        vnf_mapping[vnf] = {'node' : node, 'edge' : edge}
        # If we already placed the same VNF on the same node we dont decrease capacity
        if not (node in {pg[vnf]['node'] for pg in slice_placement.values()}):
            # Decrease capacity on node
            for res in self.resources:
                G.nodes[node]['capacity'][res] -= self.v_resource_demand[i, vnf, res]
        #print("ADD MAPPING", vnf_mapping, i, vnf, node, path, vnf_list, idx_v, idx_n, slice_placement, edge)
        if idx_v != 0:
            # Determine the physical edges for the current virtual link placement
            curr_phys_edges = {(edge[i], edge[i+1]) for i in range(len(edge) - 1)}
            # Determine the physical edges which has been used to map this virtual edge in prev. pg-s
            prev_phys_edges = [item[vnf]['edge'] for v, item in slice_placement.items()]
            prev_phys_edges = {(prev_mapping[i], prev_mapping[i+1]) for prev_mapping in prev_phys_edges for i in range(len(prev_mapping) - 1) }
            # Determine the bw requirement
            prev_vnf = vnf_list[idx_v-1]
            bw_req = self.v_bandwidth_demand[i, prev_vnf, vnf]
            # FIXME: In this case the edge matching is direction dependent. If we consider a set of sets then it can be made direction independent.
            for c_phys_edge in curr_phys_edges:
                if not c_phys_edge in prev_phys_edges:
                    # This physical edge was not used before to place this virtual link: we reduce the capacity
                    x, y = c_phys_edge
                    G[x][y]['bandwidth'] -= bw_req

        
    def _removeMapping(self, G, vnf_mapping, i, vnf, node, path, vnf_list, idx_v, idx_n, slice_placement):
        # If we already placed the same VNF on the same node we dont increase capacity
        if not (node in {pg[vnf]['node'] for pg in slice_placement.values()}):
            # Decrease capacity on node
            for res in self.resources:
                G.nodes[node]['capacity'][res] += self.v_resource_demand[i, vnf, res]
        if idx_v != 0:
            # If we already placed the same virtual link on the same node we dont increase capacity
            edge = vnf_mapping[vnf]['edge']
            # Determine the physical edges for the current virtual link placement
            curr_phys_edges = {(edge[i], edge[i+1]) for i in range(len(edge) - 1)}
            # Determine the physical edges which has been used to map this virtual edge in prev. pg-s
            prev_phys_edges = [item[vnf]['edge'] for v, item in slice_placement.items()]
            prev_phys_edges = {(prev_mapping[i], prev_mapping[i+1]) for prev_mapping in prev_phys_edges for i in range(len(prev_mapping) - 1) }
            # Determine the bw requirement
            prev_vnf = vnf_list[idx_v-1]
            bw_req = self.v_bandwidth_demand[i, prev_vnf, vnf]
            # FIXME: In this case the edge matching is direction dependent. If we consider a set of sets then it can be made direction independent.
            for c_phys_edge in curr_phys_edges:
                if not c_phys_edge in prev_phys_edges:
                    # This physical edge was not used before to place this virtual link: we reduce the capacity
                    x, y = c_phys_edge
                    G[x][y]['bandwidth'] += bw_req

        # Remove mapping
        del vnf_mapping[vnf]
    
    def _check_vnf_node_placement(self, G, i, v, n, slice_placement):
        # Check if the VNF can be placed on the node based on the capacity and the demand
        # If we placed this VNF on the same physical node in a previous placement group
        # we can place the VNF to this node in this placement also.
        if n in {pg[v]['node'] for pg in slice_placement.values()}:
            return True
        # Else check the resource demand
        is_res_enough = all(self.v_resource_demand[i, v, res] <= G.nodes[n]['capacity'][res] 
                            for res in self.resources)
        
        src_v, src_n = self.SFC_dict[i].graph['node_s_mapping']
        dst_v, dst_n = self.SFC_dict[i].graph['node_d_mapping']
        if (v == src_v):
            return (n == src_n) and is_res_enough
        elif (v == dst_v):
            return (n == dst_n) and is_res_enough
        else:
            return is_res_enough

    def _check_link_placement(self, G, i, vnf, node, idx_v, idx_n, path, vnf_mapping, vnf_list, slice_placement):
        # Check if the virtual link placement is valid
        if idx_v == 0:
            # For the first node there is no edge
            return True
        else:
            prev_v_node = vnf_mapping[ vnf_list[idx_v-1] ]["node"]
            edge_from_idx = path.index(prev_v_node)
            edge_to_idx = idx_n + 1
            edge = path[edge_from_idx : edge_to_idx]
        # Determine the physical edges for the current virtual link placement
        curr_phys_edges = {(edge[i], edge[i+1]) for i in range(len(edge) - 1)}
        # Determine the physical edges which has been used to map this virtual edge in prev. pg-s
        prev_phys_edges = [item[vnf]['edge'] for v, item in slice_placement.items()]
        prev_phys_edges = {(prev_mapping[i], prev_mapping[i+1]) for prev_mapping in prev_phys_edges for i in range(len(prev_mapping) - 1) }
        # Determine the bw requirement
        prev_vnf = vnf_list[idx_v-1]
        bw_req = self.v_bandwidth_demand[i, prev_vnf, vnf]
        # FIXME: In this case the edge matching is direction dependent. If we consider a set of sets then it can be made direction independent.
        link_bools = []
        for c_phys_edge in curr_phys_edges:
            if c_phys_edge in prev_phys_edges:
                link_bools.append(True)
            else:
                x, y = c_phys_edge
                link_bools.append(bw_req <= G[x][y]['bandwidth'])
        return all(link_bools)
        
    def place_vnfs_recursive(self, G, i, path, vnf_list, idx_v, idx_n, vnf_mapping, slice_placement):
        if idx_v == len(vnf_list):
            # Succesfully placed all VNFs
            return vnf_mapping
            
        # Go through all the nodes on the path
        for j in range(idx_n, len(path)):
            node = path[j]
            vnf = vnf_list[idx_v]
            is_node_ok = self._check_vnf_node_placement(G=G, i=i, v=vnf, n=node, slice_placement=slice_placement)
            is_link_ok = self._check_link_placement(G=G, i=i, vnf=vnf, node=node, idx_v=idx_v, idx_n=j,
                            path=path, vnf_mapping=vnf_mapping, vnf_list=vnf_list, slice_placement=slice_placement)
            if is_node_ok and is_link_ok:
                # Place the VNF on the node
                self._addMapping(G, vnf_mapping, i, vnf, node, path, vnf_list, idx_v, j, slice_placement)
    
                # Recur for the next VNF on the next node
                result = self.place_vnfs_recursive(G=G, i=i, path=path, vnf_list=vnf_list, idx_v=idx_v+1,idx_n=j+1, 
                                                   vnf_mapping=vnf_mapping, slice_placement=slice_placement)
                # If the placement of the next VNFs is succesfull we return the result
                if result is not None:
                    return result
                # Backtrack if placement was not successful
                self._removeMapping(G, vnf_mapping, i, vnf, node, path, vnf_list, idx_v, j, slice_placement)
    
        # We could not place the VNFs on this path, thus we return None
        return None

    
    def run_for_i_resource_saving(self, G, i):
        """
        Places all placement groups of slice i, using the resource saving approach.
        Resource saving approach is the following:
        - Find the paths between S and D that comply with the delay requirement and the which are the "most available"
        - Place the placement groups on these paths
            - Try to place the VNF on the given path recursively
            - If the placement on the path is succesful, the gth placement group is found
        - Repeat this process until for all placement group a placement found OR run out of possible paths.
        - Finally check the availability for the slice.
        """
        # Get the source and destination nodes
        (_, src_n), (_, dst_n) = self.SFC_dict[i].graph['node_s_mapping'], self.SFC_dict[i].graph['node_d_mapping']
        # Get the best paths between S and D
        all_paths = k_shortest_paths(G=G, s=src_n, d=dst_n, 
                                     K=10, max_delay=self.delay_req[i])
        if len(all_paths) == 0:
            print("Infeasible! No path between S and D")
            return None
        all_paths = [x['path'] for x in all_paths]
        # Get the list of all VNFs
        vnf_list = list(self.SFC_dict[i].nodes)
        # Init the placement dict and the path index
        slice_placement = {}
        idx_path = 0
        for g in range(1, self.max_groups[i] + 1):
            placement = None
            while (idx_path < len(all_paths)):
                path = all_paths[idx_path]
                placement = self.place_vnfs_recursive(G=G, i=i, path=path, vnf_list=vnf_list, idx_v=0, idx_n=0, 
                                                      vnf_mapping={}, slice_placement=slice_placement)
                if placement is not None:
                    # We found a good placement on this path
                    idx_path += 1
                    break
                else:
                    # We did not find a placement on this path, go to the next path
                    idx_path += 1
            
            if placement is not None:
                # We found a placement for placement group g, store it
                slice_placement[(i, g)] = placement
                # Check if the availability is satisfied, if not go to next g
                slice_availability = self.calc_slice_availability(slice_placement, i)
                if slice_availability >= self.availability_req[i]:
                    # We dont place anymore placement groups.
                    return slice_placement
                else:
                    # Availability requirement not reached, try place a new g
                    continue
            else:
                # We did not find placement for group g, thus the placement procedure has failed
                return None
        # We did not met the availability requirement using all groups, we fail
        return None
        
    def _update_graph_attributes(self, G_super, G_sub):
        """
        Updates the node and edge attributes of G_super by the values in G_sub.
        It is assumed that G_sub is the subgraph of G_super.
        """
        # Update node attributes
        for node, data in G_sub.nodes(data=True):
            G_super.nodes[node].update(data)
        # Update edge attributes
        for u, v, data in G_sub.edges(data=True):
            G_super.edges[u, v].update(data)

    def run_for_i_resource_wasteful_node_disjoint(self, G, i):
        """
        Places all placement groups of slice i, in a way that they are edge and node disjoint. 
        This will make the placement groups completely independent of each other (except obviously the src and dst nodes)
       
        The edge and node disjoint placement logic is the following:
        - Find the paths between S and D that comply with the delay requirement and the which are the "most available"
        - Place the placement groups on these paths
            - Try to place the VNF on the given path recursively
            - If the placement on the path is succesful, the gth placement group is found
        - If the group was succesfully placed, check the availability
            - If it is satisfied, we done
        - If not, remove the path from the graph (including, the nodes)
            - And try again with the next group
        """
        # Get the source and destination nodes
        (src_v, src_n), (dst_v, dst_n) = self.SFC_dict[i].graph['node_s_mapping'], self.SFC_dict[i].graph['node_d_mapping']
        # Copy the graph
        G_copy = copy.deepcopy(G)
        # Get the list of all VNFs
        vnf_list = list(self.SFC_dict[i].nodes)
        # Init the placement dict and the path index
        slice_placement = {}
        for g in range(1, self.max_groups[i] + 1):
            # Get the best paths between S and D
            all_paths = k_shortest_paths(G=G_copy, s=src_n, d=dst_n, K=10, max_delay=self.delay_req[i])
            if len(all_paths) == 0:
                print("Infeasible! No path between S and D")
                return None
            all_paths = [x['path'] for x in all_paths]
            
            placement = None
            for path in all_paths:
                placement = self.place_vnfs_recursive(G=G_copy, i=i, path=path, vnf_list=vnf_list, idx_v=0, idx_n=0, 
                                                      vnf_mapping={}, slice_placement=slice_placement)
                if placement is not None:
                    # We found a good placement on this path
                    # Update the attributes of the original graph
                    self._update_graph_attributes(G_super=G, G_sub=G_copy)
                    # Remove all nodes 
                    nodes_remove = [place['node'] for vnf, place in placement.items() if vnf not in (src_v, dst_v)]
                    G_copy.remove_nodes_from(nodes_remove)
                    break
            
            if placement is not None:
                # We found a placement for placement group g, store it
                slice_placement[(i, g)] = placement
                # Check if the availability is satisfied, if not go to next g
                slice_availability = self.calc_slice_availability(slice_placement, i)
                if slice_availability >= self.availability_req[i]:
                    # We dont place anymore placement groups.
                    return slice_placement
                else:
                    # Availability requirement not reached, try place a new g
                    continue
            else:
                # We did not find placement for group g, thus the placement procedure has failed
                return None
        # We did not met the availability requirement using all groups, we fail
        return None

    def run_for_i_resource_wasteful_link_disjoint(self, G, i):
        """
        Places all placement groups of slice i, in a way that they are edge and node disjoint. 
        This will make the placement groups completely independent of each other (except obviously the src and dst nodes)
       
        The edge and node disjoint placement logic is the following:
        - Find the paths between S and D that comply with the delay requirement and the which are the "most available"
        - Place the placement groups on these paths
            - Try to place the VNF on the given path recursively
            - If the placement on the path is succesful, the gth placement group is found
        - If the group was succesfully placed, check the availability
            - If it is satisfied, we done
        - If not, remove the path from the graph (including, the nodes)
            - And try again with the next group
        """
        # Get the source and destination nodes
        (src_v, src_n), (dst_v, dst_n) = self.SFC_dict[i].graph['node_s_mapping'], self.SFC_dict[i].graph['node_d_mapping']
        # Copy the graph
        G_copy = copy.deepcopy(G)
        # Get the list of all VNFs
        vnf_list = list(self.SFC_dict[i].nodes)
        # Init the placement dict and the path index
        slice_placement = {}
        for g in range(1, self.max_groups[i] + 1):
            # Get the best paths between S and D
            all_paths = k_shortest_paths(G=G_copy, s=src_n, d=dst_n, K=10, max_delay=self.delay_req[i])
            if len(all_paths) == 0:
                print("Infeasible! No path between S and D")
                return None
            all_paths = [x['path'] for x in all_paths]
            
            placement = None
            for path in all_paths:
                placement = self.place_vnfs_recursive(G=G_copy, i=i, path=path, vnf_list=vnf_list, idx_v=0, idx_n=0, 
                                                      vnf_mapping={}, slice_placement=slice_placement)
                if placement is not None:
                    # We found a good placement on this path
                    # Update the attributes of the original graph
                    self._update_graph_attributes(G_super=G, G_sub=G_copy)
                    # Remove all links 
                    edges_to_remove = [item['edge'] for v, item in placement.items()]
                    edges_to_remove = list(itertools.chain.from_iterable(edges_to_remove))
                    edges_to_remove = {(edges_to_remove[i], edges_to_remove[i+1]) for i in range(len(edges_to_remove) - 1) 
                                       if edges_to_remove[i] != edges_to_remove[i+1]}
                    G_copy.remove_edges_from(edges_to_remove)
                    break
            
            if placement is not None:
                # We found a placement for placement group g, store it
                slice_placement[(i, g)] = placement
                # Check if the availability is satisfied, if not go to next g
                slice_availability = self.calc_slice_availability(slice_placement, i)
                if slice_availability >= self.availability_req[i]:
                    # We dont place anymore placement groups.
                    return slice_placement
                else:
                    # Availability requirement not reached, try place a new g
                    continue
            else:
                # We did not find placement for group g, thus the placement procedure has failed
                return None
        # We did not met the availability requirement using all groups, we fail
        return None

    def run(self):
        placements = {(i, g) : None for i in self.slices for g in range(1, self.max_groups[i] + 1)}
        # Sort slices by availability
        slices_sorted = sorted(self.slices, key=lambda x: self.availability_req[x], reverse=True)
        for i in slices_sorted:
            if self.method == "resource_saving":
                sfc_mapping = self.run_for_i_resource_saving(G=self.G_sub, i=i)
            elif self.method == "node_disjoint":
                sfc_mapping = self.run_for_i_resource_wasteful_node_disjoint(G=self.G_sub, i=i)
            elif self.method == "link_disjoint":
                sfc_mapping = self.run_for_i_resource_wasteful_link_disjoint(G=self.G_sub, i=i)
            
            if sfc_mapping is None:
                # We could not place SFCs in slice i --> Not feasible
                return None
            else:
                placements.update(sfc_mapping)
                
        return placements
