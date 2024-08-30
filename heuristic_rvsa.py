import networkx as nx
import numpy as np
import cspy
import copy
import itertools

class HeuristicRVSA():
    def __init__(self, G_sub, SFC_dict):
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

    
    def TAMCRA(self, G, src, dst, kappa, min_bw, method='exact'):
        """
        G: Graph which we find the shortest path between src and dst
        
        """
        G = copy.deepcopy(G)
        if method != 'exact':
            NotImplementedError("Not yet implemented. Should remove the first resource")
        # According to the original article we want to find the shortest path (according to delay), 
        # where the product of the link availablities are less then a given value kappa
        # Also the bandwidth of the path should be at least the requested amount.
        #   Note: The product of the availabilities is the sum of the log of the availabilites.
        #         Kappa should be changed accordingly
        log_kappa = np.log(kappa)
        
        # Eliminate nodes with less then the min_bw
        G.remove_edges_from([(u, v) for u, v, data in G.edges(data=True) if data['bandwidth'] < min_bw])
        
        # Set the resource array to be the log of the availability
        # And the weight to be the delay
        for n, m, data in G.edges(data=True):
            G[n][m]['res_cost'] = np.array([  1, np.log(data['availability'])  ])
            G[n][m]['weight'] = data['delay']
        
        # Create a graph that is suitable for the problem
        G = nx.DiGraph(G.copy(), directed=True, n_res=2)
        G = nx.relabel_nodes(G, mapping={src : 'Source', dst : 'Sink'})
        # Set resource boundaries
        max_res, min_res = [len(G.nodes), np.inf], [0, log_kappa]
        # init algorithm
        bidirec = cspy.BiDirectional(G, max_res, min_res)
        # Call and query attributes
        bidirec.run()
        found_path = bidirec.path
        total_delay = bidirec.total_cost
        if found_path is not None:
            found_path[0] = src
            found_path[-1] = dst
        return found_path, total_delay
    def RVSA_addMapping(self, i, substrate_mapping, G_substrate, v, n, shortest_path, bw_req):
        # Add mapping to the mapping dict
        substrate_mapping[v] = {'node' : n, 'edge' : shortest_path}
        # Decrease capacity on node
        for res in self.resources:
            G_substrate.nodes[n]['capacity'][res] -= self.v_resource_demand[i, v, res]
        # Decrease capacity on links
        for j in range(1, len(shortest_path)):
            x, y = shortest_path[j-1], shortest_path[j]
            G_substrate[x][y]['bandwidth'] -= bw_req
        
    def RVSA_removeMapping(self, i, substrate_mapping, G_substrate, v, n, shortest_path, bw_req):
        # Increase capacity on node
        for res in self.resources:
            G_substrate.nodes[n]['capacity'][res] += self.v_resource_demand[i, v, res]
        # Increase capacity on links
        for j in range(1, len(shortest_path)):
            x, y = shortest_path[j-1], shortest_path[j]
            G_substrate[x][y]['bandwidth'] += bw_req
        # Remove mapping
        del substrate_mapping[v]
    
    def RVSA(self, i, G_substrate, idx_v, substrate_mapping, delta, max_delay, vnf_list, pi_v_n):
        if idx_v == len(vnf_list):
            # Successfully placed all VNFs
            return True
        
        v = vnf_list[idx_v]
        for n in self.phys_nodes_sorted:
            # 1) Check if this substrate node N is already used
            if n in  (val['node'] for vnf, val in substrate_mapping.items()):
                continue
            # 2) Check if V can be placed on N
            if pi_v_n[v, n]:
                # If yes, get the shortest path between current N and the previous N
                prev_v = vnf_list[idx_v-1]
                prev_n = substrate_mapping[prev_v]['node']
                bw_req = self.v_bandwidth_demand[i, prev_v, v]
                shortest_path, path_delay = self.TAMCRA(G=G_substrate, src=prev_n, dst=n, kappa=delta, min_bw=bw_req)
                # If the shortest path is correct then we place V on N
                if (shortest_path is not None) and (max_delay >= path_delay):
                    # Place V on N and place the next V
                    self.RVSA_addMapping(i, substrate_mapping, G_substrate, v, n, shortest_path, bw_req)
                    if self.RVSA(i, G_substrate, idx_v + 1, substrate_mapping, delta, max_delay - path_delay, vnf_list, pi_v_n):
                        return True
                    # If RVSA returned with False, we could not find feasible placement, revert the mapping.
                    self.RVSA_removeMapping(i, substrate_mapping, G_substrate, v, n, shortest_path, bw_req)
    
            # At this point current V cannot be placed on N because
            # 1) node capacity is not enough
            # 2) Shortest path is not existing
            # We jump to the next N and check if V can be placed there
        
        # If we checked all N to place V and none is working then we backtrack
        return False
    def run_for_i_g(self, i, g, G):
        vnf_list = list(self.SFC_dict[i].nodes)
        
        Tx = self.delay_req[i]
        delta = self.availability_req[i]
        
        # Create substrate mapping dict
        substrate_mapping = {}
        # First node mapping
        src_v, src_n = self.SFC_dict[i].graph['node_s_mapping']
        dst_v, dst_n = self.SFC_dict[i].graph['node_d_mapping']
        substrate_mapping[src_v] = {'node' : src_n, 'edge' : []}
        pi_v_n = {
            (v, n): all(self.v_resource_demand[i, v, res] <= G.nodes[n]['capacity'][res] for res in self.resources)
            for v in self.SFC_dict[i].nodes
            for n in self.phys_nodes_sorted
        }
        for v, n in pi_v_n.keys():
            if (v == src_v and n != src_n) or (v != src_v and n == src_n):
                pi_v_n[v, n] = False
            if (v == dst_v and n != dst_n) or (v != dst_v and n == dst_n):
                pi_v_n[v, n] = False

        if self.RVSA(i, G, 1, substrate_mapping, delta, Tx, vnf_list, pi_v_n):
            return substrate_mapping
        else:
            return None
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

    
    def run(self):
        placements = {(i, g) : None for i in self.slices for g in range(1, self.max_groups[i] + 1)}
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                sfc_mapping = self.run_for_i_g(i, g, self.G_sub)
                # Major flaw in the RVSA problem, that if all links and nodes has free capacity
                #   the algorithm will place the SFC on the same nodes/links, thus not increasing the
                #   availability, making high not acceptances. 
                if sfc_mapping is None:
                    # We could not place SFC (i, g) --> Not feasible
                    return None
                placements[i, g] = sfc_mapping
                # Check the availability of slice i so far
                slice_availability = self.calc_slice_availability(placements, i)
                is_slice_avail_ok = slice_availability >= self.availability_req[i] 
                if is_slice_avail_ok:
                    # We found SFCs that satisfy the constraints we dont need to place more, jump to next slice
                    break
                else:
                    # We need more placement groups to satisfy the requirement, jump to next g
                    continue
            
            if not is_slice_avail_ok:
                # If we ran out of the placement groups and the slice is not ok, then it is not feasible
                return None
        
        return placements