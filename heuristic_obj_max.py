import copy
import numpy as np


class ObjectiveFunctionMaximizer():
    def __init__(self, G_sub, SFC_dict, obj_coeffs, placement):
        self.placement = placement
        self.G_sub = copy.deepcopy(G_sub)
        self.SFC_dict = SFC_dict
        self.obj_coeffs = obj_coeffs
        
        self.phys_nodes = list(self.G_sub.nodes)
        self.phys_links = list(self.G_sub.edges)

        self.resources = list(self.G_sub.nodes[self.phys_nodes[0]]['capacity'].keys())
        self.slices = list(self.SFC_dict.keys())
        self.max_groups = {i: self.SFC_dict[i].graph['max_placement_groups'] for i in self.slices}
        self.v_resource_demand = {(i, v, p): self.SFC_dict[i].nodes[v]['demand'][p] 
                          for i in self.slices for v in self.SFC_dict[i].nodes for p in self.resources}
        self.v_bandwidth_demand = {(i, u, v): self.SFC_dict[i][u][v]['bandwidth'] 
                                   for i in self.slices for (u, v) in self.SFC_dict[i].edges}
        self.delay_req = {i: self.SFC_dict[i].graph['max_delay'] for i in self.slices}

        self.phys_bandwidth_capacity = {(i, j): self.G_sub.edges[i, j]['bandwidth'] for (i, j) in self.phys_links}
        self.phys_resource_capacity = {(n, p): self.G_sub.nodes[n]['capacity'][p] for n in self.phys_nodes for p in self.resources}
        self.phys_delay = {(i, j): self.G_sub.edges[i, j]['delay'] for (i, j) in self.phys_links}

        self._create_variables()
        self._set_variables()
        self._create_resource_sharing_groups()
    
    def _create_variables(self):
        """
        Create and initialize the decision variables using in the optimization
        """
        self.x = { (i, g, v, p) : 0
            for i in self.slices for g in range(1, self.max_groups[i] + 1) for v in self.SFC_dict[i].nodes for p in self.resources
        }
        self.y = { (i, g, u, v) : 0
            for i in self.slices for g in range(1, self.max_groups[i] + 1) for (u, v) in self.SFC_dict[i].edges
        }
        self.X = { (i, g, v, n) : 0
            for i in self.slices for g in range(1, self.max_groups[i] + 1) for v in self.SFC_dict[i].nodes for n in self.phys_nodes
        }
        self.Y = { (i, g, u, v, n, m) : 0
            for i in self.slices for g in range(1, self.max_groups[i] + 1) for (u, v) in self.SFC_dict[i].edges for (n, m) in self.phys_links
        }
    def _set_variables(self):
        for (i, g), mapping in self.placement.items():
            if mapping is None: continue
            vnf_list = list(self.SFC_dict[i].nodes)
            for j, v in enumerate(vnf_list):
                n = mapping[v]['node']
                # Set node mapping and node res alloc
                self.X[i, g, v, n] = 1
                for p in self.resources:
                    self.x[i, g, v, p] = self.v_resource_demand[i, v, p]
                if j != 0:
                    # Set link mapping and link bw alloc
                    edge = mapping[v]['edge']
                    u = vnf_list[j-1]
                    p_links = [(edge[idx], edge[idx+1]) for idx in range(len(edge) - 1)]
                    self.y[i, g, u, v] = self.v_bandwidth_demand[i, u, v]
                    for pnode1, pnode2 in p_links:
                        if (pnode1, pnode2) in self.phys_links:
                            self.Y[i, g, u, v, pnode1, pnode2] = 1
                        else:
                            self.Y[i, g, u, v, pnode2, pnode1] = 1
                            
    def _create_resource_sharing_groups(self):
        # Create a list where the elements are keys for self.x, 
        #    where each element is a list containing resource indicator keys, 
        #    which value should equal
        self.shared_resources_nodes = [
                [(_i, _g, _v, p) for (_i, _g, _v, _n), value in self.X.items() 
                 if _i == i and _v == v and _n == n and value == 1]
            for i in self.slices 
            for v in self.SFC_dict[i].nodes 
            for n in self.phys_nodes
            for p in self.resources
        ]
        self.shared_resources_nodes = [elem for elem in self.shared_resources_nodes if len(elem) > 0]

        # Create a list where the elements are keys for self.y
        #    where each element is a list of such keys which value should equal
        _iguv = {(i, g, u, v) for (i, g, u, v, n, m) in self.Y.keys()}
        # Helper dict: (i, g, u, v) --> what edges are used
        virtual_2_phys_edges = {
            (i, g, u, v) : [(_n, _m) for (_i, _g, _u, _v, _n, _m), value in self.Y.items()
                           if value == 1 if _i == i if _g == g if _u == u if _v == v]
            for (i, g, u, v) in _iguv
        }
        virtual_2_phys_edges = {key : val for key, val in virtual_2_phys_edges.items() if len(val) > 0}

        self.shared_bandwidth_links = {
                tuple({(_i, _g, _u, _v) for (_i, _g, _u, _v), _edges in virtual_2_phys_edges.items() 
                 if _i == i if _u == u if _v == v if len(set(edges).intersection(_edges)) > 0})
            for (i, g, u, v), edges in virtual_2_phys_edges.items()
        }
        self.shared_bandwidth_links = list(self.shared_bandwidth_links)

    def _calc_qos_i_g(self, i, g):
        # Store the QoS for each resource type for i, g
        qos_p_i_g = {(p, i, g) : 0 for p in self.resources + ["bw", "delay"]}
        # Calculate delay QoS
        e2e_delay_i_g = sum(
            self.phys_delay[n, m]*self.Y[i, g, u, v, n, m]
            for u, v in self.SFC_dict[i].edges
            for n, m in self.phys_links
        )
        qos_p_i_g["delay", i, g] = e2e_delay_i_g / self.delay_req[i]
        # Calculate BW QoS
        qos_p_i_g["bw", i, g] = sum(
            self.y[i, g, u, v]
            for u, v in self.SFC_dict[i].edges
        ) / sum(
            self.v_bandwidth_demand[i, u, v]
            for u, v in self.SFC_dict[i].edges
        )
        # Calculate Node Physical resources QoS
        for p in self.resources:
            qos_p_i_g[p, i, g] = sum(
                self.x[i, g, v, p]
                for v in self.SFC_dict[i].nodes
            ) / sum(
                self.v_resource_demand[i, v, p]
                for v in self.SFC_dict[i].nodes
            )
        
        return qos_p_i_g
    
    def _calc_qoe_i_g(self, i, g):
        if self.placement[i, g] is None: return 0
        qos_p_i_g = self._calc_qos_i_g(i, g)
        qoe_sum = 0
        for p in self.resources + ["bw", "delay"]:
            beta = self.obj_coeffs['beta'][i, p]
            gamma = self.obj_coeffs['gamma'][i, p]
            lambd = self.obj_coeffs['lambda'][i, p]
            mu = self.obj_coeffs['mu'][i, p]
            qos_val = qos_p_i_g[p, i, g]
            if p == "delay":
                # IQX
                qoe_val = beta * np.exp(gamma * qos_val + lambd) + mu
            else:
                # WFL
                qoe_val = beta * np.log(gamma * qos_val + lambd) + mu
            qoe_sum += qoe_val
        return qoe_sum
    
    def _calc_obj_o1(self):
        sum_qoe = 0
        for i in self.slices:
            slice_qoe = sum(
                self._calc_qoe_i_g(i, g)
                for g in range(1, self.max_groups[i] + 1)
            )
            num_active_pg = sum(
                1 if mapping is not None else 0
                for (_i, g), mapping in self.placement.items()
                if _i == i
            )
            sum_qoe += slice_qoe / num_active_pg
        return sum_qoe
        
    def _calc_obj_o2(self):
        slice_res_cost = 0
        for i in self.slices:
            slice_node_cost = sum(
                max(
                    self.x[i, g, v, p]*self.X[i, g, v, n]
                    for g in range(1, self.max_groups[i] + 1)
                )
                for v in self.SFC_dict[i].nodes
                for p in self.resources
                for n in self.phys_nodes
            )
            slice_link_cost = sum(
                max(
                    self.y[i, g, u, v]*self.Y[i, g, u, v, n, m]
                    for g in range(1, self.max_groups[i] + 1)
                )
                for u, v in self.SFC_dict[i].edges
                for n, m in self.phys_links
            )
            slice_res_cost += slice_node_cost + slice_link_cost
        return slice_res_cost
        
    def _calc_obj_o3(self):
        num_used_nodes = sum(
            max( val for (i, g, v, _n), val in self.X.items() if _n == n )
            for n in self.phys_nodes
        )
        num_used_links = sum(
            max( val for (i, g, u, v, _n, _m), val in self.Y.items() if _n == n if _m == m)
            for n, m in self.phys_links
        )
        return num_used_nodes + num_used_links

    def _calc_obj_o4(self):
        return sum(
            1 if mapping is not None else 0
            for (i, g), mapping in self.placement.items()
        )

    def calculate_obj_function(self):
        o1 = self._calc_obj_o1()
        o2 = self._calc_obj_o2()
        o3 = self._calc_obj_o3()
        o4 = self._calc_obj_o4()
        return self.obj_coeffs['rho1']*o1 - self.obj_coeffs['rho2']*o2 - \
               self.obj_coeffs['rho3']*o3 - o4
        
    def check_capacity_constraints(self):
        # Check link capacities
        for n, m in self.phys_links:
            LHS = sum(
                max(
                    self.y[i, g, u, v] * self.Y[i, g, u, v, n, m]
                    for g in range(1, self.max_groups[i] + 1)
                )
                for i in self.slices
                for u, v in self.SFC_dict[i].edges
            )
            RHS = self.phys_bandwidth_capacity[n, m]
            if LHS > RHS: return False
        # Check node capacities
        for n in self.phys_nodes:
            for p in self.resources:
                LHS = sum(
                    max(
                        self.x[i, g, v, p] * self.X[i, g, v, n]
                        for g in range(1, self.max_groups[i] + 1)
                    )
                    for i in self.slices
                    for v in self.SFC_dict[i].nodes
                )
                RHS = self.phys_resource_capacity[n, p]
                if LHS > RHS: return False
        return True
    def run(self, perc_step = 10):
        """
        Run the resource pumping heuristic.
        The method works by increasing the resources allocated to each VNF and virtual link 
            until the objective function increases with it, or until capacity does not allow further increases.
        """
        # Increase the node resources.
        is_node_res_increasable = [True for _ in range(len(self.shared_resources_nodes))]
        while any(is_node_res_increasable):
            # Try to increase resources until there is at least one resource that can be increased.
            for j in range(len(self.shared_resources_nodes)):
                if not is_node_res_increasable[j]: continue
                current_obj_val = self.calculate_obj_function()
                # Get the keys for allocated node resources
                res_keys = self.shared_resources_nodes[j]
                # Modify the allocated resources
                epsilon = 1e-5 # Required for resources with 0
                mod_val = self.x[res_keys[0]] * (perc_step / 100) + epsilon
                for key in res_keys:
                    self.x[key] += mod_val
                # Calculate and check whether we improved the obj val
                new_obj_val = self.calculate_obj_function()
                if (new_obj_val <= current_obj_val) or \
                    not self.check_capacity_constraints():
                    # Revert the changes if the objective function is not increased 
                    # or the capacity requirements are not met.
                    for key in res_keys:
                        self.x[key] -= mod_val
                    is_node_res_increasable[j] = False

        # Increase the allocated bandwidth .
        is_link_res_increasable = [True for _ in range(len(self.shared_bandwidth_links))]
        while any(is_link_res_increasable):
            # Try to increase resources until there is at least one resource that can be increased.
            for j in range(len(self.shared_bandwidth_links)):
                if not is_link_res_increasable[j]: continue
                current_obj_val = self.calculate_obj_function()
                # Get the keys for allocated bandwidth resources
                res_keys = self.shared_bandwidth_links[j]
                # Modify the allocated resources
                epsilon = 1e-5 # Required for resources with 0
                mod_val = self.y[res_keys[0]] * (perc_step / 100) + epsilon
                for key in res_keys:
                    self.y[key] += mod_val
                # Calculate and check whether we improved the obj val
                new_obj_val = self.calculate_obj_function()
                if (new_obj_val <= current_obj_val) or \
                    not self.check_capacity_constraints():
                    # Revert the changes if the objective function is not increased 
                    # or the capacity requirements are not met.
                    for key in res_keys:
                        self.y[key] -= mod_val
                    is_link_res_increasable[j] = False
    
    def get_final_solution(self):
        return {
            "X" : self.X,
            "Y" : self.Y,
            "x" : self.x, 
            "y" : self.y,
        }
    def get_objective_scores(self):
        objective_value = self.calculate_obj_function()
        objective_1_value = self._calc_obj_o1()
        objective_2_value = self._calc_obj_o2()
        objective_3_value = self._calc_obj_o3()
        objective_4_value = self._calc_obj_o4()
        return {
            "objective_value" : objective_value, 
            "objective_1_value" : objective_1_value, "objective_2_value" : objective_2_value, 
            "objective_3_value" : objective_3_value, "objective_4_value" : objective_4_value
        }