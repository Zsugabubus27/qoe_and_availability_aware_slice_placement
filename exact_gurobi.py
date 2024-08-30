import networkx as nx
import numpy as np
import gurobipy as gp
import copy
from gurobipy import GRB
import itertools
import json


class ExactSolutionGurobi():
    def __init__(self, G_sub, SFC_dict, obj_coeffs, log_file_path) -> None:
        self.substrate_network = copy.deepcopy(G_sub)
        self.slice_requests = SFC_dict
        self.object_coeffs = obj_coeffs
        # Initialize the Gurobi model
        self.model = gp.Model("Network_Slice_Optimization")
        self.model.params.LogFile = log_file_path


        # Sets and Parameters
        self.phys_nodes = list(self.substrate_network.nodes)
        self.phys_links = list(self.substrate_network.edges)
        def set_phys_links_bidirect():
            res_list = []
            for (n, m) in self.phys_links:
                res_list.append((n, m))
                res_list.append((m, n))
            return res_list
        self.phys_links_bidirect = set_phys_links_bidirect()
        self.slices = list(self.slice_requests.keys())
        self.resources = list(self.substrate_network.nodes[self.phys_nodes[0]]['capacity'].keys())

        self.all_vnfs = {v for i in self.slices for v in self.slice_requests[i].nodes}
        self.all_vlinks = {link for i in self.slices for link in self.slice_requests[i].edges}

        # Parameters
        self.delay_req = {i: self.slice_requests[i].graph['max_delay'] for i in self.slices}
        self.availability_req = {i: self.slice_requests[i].graph['min_availability'] for i in self.slices}
        self.max_groups = {i: self.slice_requests[i].graph['max_placement_groups'] for i in self.slices}
        self.slice_group_pairs = {i : [pairs for pairs in itertools.combinations(range(1, self.max_groups[i]+1), 2)]  for i in self.slices}

        # Physical resource capacities and demands
        self.phys_resource_capacity = {(n, p): self.substrate_network.nodes[n]['capacity'][p] for n in self.phys_nodes for p in self.resources}
        self.phys_bandwidth_capacity = {(i, j): self.substrate_network.edges[i, j]['bandwidth'] for (i, j) in self.phys_links}
        self.phys_delay = {(i, j): self.substrate_network.edges[i, j]['delay'] for (i, j) in self.phys_links}

        self.availability_node = {n: self.substrate_network.nodes[n]['availability'] for n in self.phys_nodes}
        self.availability_link = {(i, j): self.substrate_network.edges[i, j]['availability'] for (i, j) in self.phys_links}

        self.v_resource_demand = {(i, v, p): self.slice_requests[i].nodes[v]['demand'][p] for i in self.slices for v in self.slice_requests[i].nodes for p in self.resources}
        self.v_bandwidth_demand = {(i, u, v): self.slice_requests[i][u][v]['bandwidth'] for i in self.slices for (u, v) in self.slice_requests[i].edges}

        # Decision Variables
        self.X = self.model.addVars([(i, g, v, n) for i in self.slices for g in range(1, self.max_groups[i] + 1) for v in self.slice_requests[i].nodes for n in self.phys_nodes], vtype=GRB.BINARY, name="X")
        self.Y_FC = self.model.addVars([(i, g, u, v, n, m) for i in self.slices for g in range(1, self.max_groups[i] + 1) for (u, v) in self.slice_requests[i].edges for (n, m) in self.phys_links_bidirect], vtype=GRB.BINARY, name="Y_FC")
        self.Y = self.model.addVars([(i, g, u, v, n, m) for i in self.slices for g in range(1, self.max_groups[i] + 1) for (u, v) in self.slice_requests[i].edges for (n, m) in self.phys_links], vtype=GRB.BINARY, name="Y")
        self.x = self.model.addVars([(i, g, v, p) for i in self.slices for g in range(1, self.max_groups[i] + 1) for v in self.slice_requests[i].nodes for p in self.resources], vtype=GRB.CONTINUOUS, lb=0, name="x")
        self.y = self.model.addVars([(i, g, u, v) for i in self.slices for g in range(1, self.max_groups[i] + 1) for (u, v) in self.slice_requests[i].edges], vtype=GRB.CONTINUOUS, lb=0, name="y")

        # Init Helper Vars
        self.placement_group_subsets = {i : self.get_placement_group_subset(i) for i in self.slices}
        self.is_group_active_var = self.model.addVars(((i, g) for i in self.slices for g in range(1, self.max_groups[i] + 1)), 
                    vtype=GRB.BINARY, name=f"is_group_active_var")
        self.is_S_active_var = self.model.addVars(((i, S) for i in self.slices for S in self.placement_group_subsets[i]), 
                    vtype=GRB.BINARY, name=f"is_S_active_var")
        
        self.set_is_group_active_rule()
        self.set_is_s_active_rule()

        # ########### Create Constraints ###########
        # ###### Routing and placement
        self.flow_conservation_aux_var_rule()
        self.flow_conservation_rule()
        self.primary_placement_group_node_placement_rule()
        self.backup_placement_group_node_placement_rule()
        self.no_multiple_vnfs_on_same_node_rule()
        self.node_sharing_rule()
        self.link_sharing_rule()
        self.fix_start_and_destination_nodes()

        # ###### Capacity
        self.delay_slice_e2e = self.calculate_delay_slice_e2e()

        # It stores the x[i, g, v, p] * X[i, g, v, n] values
        self.node_allocated_res_var = self.model.addVars( ( (i, g, v, n, p) for i in self.slices 
                for g in range(1, self.max_groups[i] + 1) for v in self.slice_requests[i].nodes 
                for n in self.phys_nodes for p in self.resources), 
                vtype=GRB.CONTINUOUS, lb=0, name="node_allocated_res_var")
        # Aux var that stores the maximum of node_allocated_res_var among g in each i
        self.max_node_allocated_res_per_slice_var = self.model.addVars( ( (i, v, n, p) for i in self.slices 
                for v in self.slice_requests[i].nodes for n in self.phys_nodes for p in self.resources), 
                vtype=GRB.CONTINUOUS, lb=0, name="max_node_allocated_res_per_slice_var")

        # It stores the y[i, g, u, v]*Y[i, g, u, v, n, m] values
        self.link_allocated_bw_var = self.model.addVars( ( (i, g, u, v, n, m) for i in self.slices 
                for g in range(1, self.max_groups[i] + 1) for u, v in self.slice_requests[i].edges 
                for n, m in self.phys_links), 
                vtype=GRB.CONTINUOUS, lb=0, name="link_allocated_bw_var")
        # Aux var that stores the maximum of link_allocated_bw_var among g in each i
        self.max_link_allocated_bw_per_slice_var = self.model.addVars( ( (i, u, v, n, m) for i in self.slices 
                for u, v in self.slice_requests[i].edges for n, m in self.phys_links), 
                vtype=GRB.CONTINUOUS, lb=0, name="max_link_allocated_bw_per_slice_var")

        self.latency_rule()
        self.slice_bw_rule()
        self.slice_node_res_rule()
        self.shared_link_capacity_rule()
        self.shared_node_res_capacity_rule()
        self.deactivate_phantom_resource_allocation()

        # ###### Availability
        self.min_node_availability = self.model.addVars(
            ( (i, S, n) for i in self.slices for S in self.placement_group_subsets[i] for n in self.phys_nodes ), 
            vtype=GRB.CONTINUOUS, lb=0, ub=1, name="min_node_availability"
        )
        self.min_node_availability_element = self.model.addVars(
            ( (i, g, v, n) for i in self.slices for g in range(1, self.max_groups[i] + 1) 
             for v in self.slice_requests[i].nodes for n in self.phys_nodes ), 
            vtype=GRB.CONTINUOUS, lb=0, ub=1, name="min_node_availability_element"
        )
        self.min_link_availability = self.model.addVars(
            ( (i, S, n, m) for i in self.slices for S in self.placement_group_subsets[i] for n, m in self.phys_links ), 
            vtype=GRB.CONTINUOUS, lb=0, ub=1, name="min_link_availability"
        )
        self.min_link_availability_element = self.model.addVars(
            ( (i, g, u, v, n, m) for i in self.slices for g in range(1, self.max_groups[i] + 1) 
             for u, v in self.slice_requests[i].edges for n, m in self.phys_links ), 
            vtype=GRB.CONTINUOUS, lb=0, ub=1, name="min_link_availability_element"
        )
        self.avail_node_aux_var = self.model.addVars(((i, S, n) for i in self.slices 
                    for S in self.placement_group_subsets[i] for n in self.phys_nodes), 
                    vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"avail_node_aux_var")
        self.avail_link_aux_var = self.model.addVars(((i, S, n, m) for i in self.slices 
                    for S in self.placement_group_subsets[i] for n, m in self.phys_links), 
                    vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"avail_link_aux_var")
        self.avail_node_link_aux_var = self.model.addVars(((i, S) for i in self.slices 
                    for S in self.placement_group_subsets[i]), 
                    vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=f"avail_node_link_aux_var")
        
        self.set_min_node_availability_element()
        self.set_min_link_availability_element()
        self.min_node_availability_rule()
        self.min_link_availability_rule()
        self.availability_rule()
        
        # ########### Create Objective function ###########


        self.QoS_res_i_g = self.create_qos_res_i_g_var()
        self.QoE_res_i_g_var = self.create_QoE_res_i_g_var()


        # Variable that stores the QoE for each placement group
        self.QoE_i_g_var = self.model.addVars( ( (i, g) for i in self.slices for g in range(1, self.max_groups[i]+1) ), 
                                    vtype=GRB.CONTINUOUS, name=f"qoe_i_g_var")
        for i, g in self.QoE_i_g_var.keys():
            self.model.addConstr(self.QoE_i_g_var[i, g] == gp.quicksum(self.QoE_res_i_g_var.select(i, g)), 
                                name=f"set_qoe_i_g_var_{i}_{g}")

        # Variable that stores the QoE for each slice
        self.QoE_i_var = self.model.addVars( (i for i in self.slices), vtype=GRB.CONTINUOUS, name=f"qoe_i_var")
        for i in self.QoE_i_var.keys():
            self.model.addConstr(self.QoE_i_var[i] == gp.quicksum(self.QoE_i_g_var.select(i)), 
                                name=f"set_qoe_i_var_{i}")

        # Number of active groups in each slice
        self.active_groups_per_slice_var = self.model.addVars( (i for i in self.slices), 
                                    vtype=GRB.INTEGER, lb=1, name=f"active_groups_per_slice_var")
        for i in self.active_groups_per_slice_var.keys():
            self.model.addConstr(self.active_groups_per_slice_var[i] == gp.quicksum(self.is_group_active_var.select(i)), 
                                name=f"set_active_groups_per_slice_var_{i}")


        self.active_groups_per_slice_inverse_var = self.create_active_groups_per_slice_inverse_var()

        self.O1_expr = gp.quicksum(self.QoE_i_var[i] * self.active_groups_per_slice_inverse_var[i] for i in self.slices)
        self.O2_expr = self.O2_expression()
        self.O3_expr = self.O3_expression()
        self.O4_expr = self.O4_expression()

        self.model.setObjective(self.object_coeffs['rho1'] * self.O1_expr - self.object_coeffs['rho2'] * self.O2_expr - \
                        self.object_coeffs['rho3'] * self.O3_expr - self.O4_expr, GRB.MAXIMIZE)


    # ####### Constrainst #######
    def get_placement_group_subset(self, i):
        placement_groups = set(range(1, self.max_groups[i] + 1))
        placement_groups_combinations = itertools.chain.from_iterable(itertools.combinations(placement_groups, r) 
                                                                    for r in range(1, len(placement_groups)+1))
        placement_groups_set = {frozenset(s) for s in placement_groups_combinations}
        return placement_groups_set

    def set_is_group_active_rule(self):
        for i in self.slices:
            v = list(self.slice_requests[i].nodes)[0]
            for g in range(1, self.max_groups[i] + 1):
                is_g_active = gp.quicksum(self.X[i, g, v, n] for n in self.phys_nodes)
                self.model.addConstr(self.is_group_active_var[i, g] == is_g_active, 
                                     name=f"set_is_group_active_rule_{i}_{g}")
    def set_is_s_active_rule(self):
        for i in self.slices:
            for S in self.placement_group_subsets[i]:
                v = list(self.slice_requests[i].nodes)[0]
                indicator = gp.all_(self.is_group_active_var[i, g] for g in S)
                self.model.addConstr(self.is_S_active_var[i, S] == indicator, 
                                     name=f"is_s_active_rule_{i}_{str(set(S)).replace(' ', '')}")

    # Constraints
    def flow_conservation_aux_var_rule(self):
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                for (u, v) in self.slice_requests[i].edges:
                    for (n, m) in self.phys_links:
                        self.model.addConstr(self.Y[i, g, u, v, n, m] == (self.Y_FC[i, g, u, v, m, n] + self.Y_FC[i, g, u, v, n, m]), 
                                             name=f"flow_conservation_aux_var_{i}_{g}_{u}_{v}_{n}_{m}")

    def flow_conservation_rule(self):
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                for (u, v) in self.slice_requests[i].edges:
                    for n in self.phys_nodes:
                        lhs = self.X[i, g, u, n] - self.X[i, g, v, n]
                        neighbors = list(self.substrate_network.neighbors(n))
                        rhs = gp.quicksum(self.Y_FC[i, g, u, v, n, m] for m in neighbors) - gp.quicksum(self.Y_FC[i, g, u, v, m, n] for m in neighbors)
                        self.model.addConstr(lhs == rhs, name=f"flow_conservation_{i}_{g}_{u}_{v}_{n}")

    # Constraint: Node placement - primary placement group
    def primary_placement_group_node_placement_rule(self):
        for i in self.slices:
            for v in self.slice_requests[i].nodes:
                self.model.addConstr( gp.quicksum(self.X[i, 1, v, n] for n in self.phys_nodes) == 1, 
                                     name=f"primary_node_placement_{i}_{v}")

    # Constraint: Node placement - other placement groups
    def backup_placement_group_node_placement_rule(self):
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                if g == 1: continue
                for v in self.slice_requests[i].nodes:
                    self.model.addConstr( gp.quicksum(self.X[i, g, v, n] for n in self.phys_nodes) <= 1, 
                                         name=f"backup_node_placement_{i}_{g}_{v}")

    def no_multiple_vnfs_on_same_node_rule(self):
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                for n in self.phys_nodes:
                    self.model.addConstr( gp.quicksum(self.X[i, g, v, n] for v in self.slice_requests[i].nodes) <= 1, 
                                    name=f"no_multiple_vnfs_on_same_node_{i}_{g}_{n}")
                    
    ### Node sharing
    def node_sharing_rule(self):
       
        # Add binary variables to represent if placements are equal
        is_node_placement_equal = self.model.addVars( (  (i, g, g_prime, v, n) for i in self.slices for g, g_prime in self.slice_group_pairs[i]
            for v in self.slice_requests[i].nodes for n in self.phys_nodes), vtype=GRB.BINARY, name="is_node_placement_equal")
        
        # Add constraints to set binary variables based on placement conditions
        for i in self.slices:
            for g, g_prime in self.slice_group_pairs[i]:
                for v in self.slice_requests[i].nodes:
                    for n in self.phys_nodes:
                        self.model.addConstr(is_node_placement_equal[i, g, g_prime, v, n] <= self.X[i, g, v, n], 
                                        name=f"is_node_placement_equal_constr_1_{i}_{g}_{g_prime}_{v}_{n}")
                        self.model.addConstr(is_node_placement_equal[i, g, g_prime, v, n] <= self.X[i, g_prime, v, n], 
                                        name=f"is_node_placement_equal_constr_2_{i}_{g}_{g_prime}_{v}_{n}")
                        self.model.addConstr(is_node_placement_equal[i, g, g_prime, v, n] >= self.X[i, g, v, n] + self.X[i, g_prime, v, n] - 1, 
                                        name=f"is_node_placement_equal_constr_3_{i}_{g}_{g_prime}_{v}_{n}")
                        for p in self.resources:
                            self.model.addConstr( (is_node_placement_equal[i, g, g_prime, v, n] == 1) >>\
                                                  (self.x[i, g, v, p] == self.x[i, g_prime, v, p]), 
                                    name=f"node_sharing_constr_{i}_{g}_{g_prime}_{v}_{n}_{p}")

    ### Link sharing
    def link_sharing_rule(self):
       
        # Add binary variables to represent if placements are equal
        is_link_placement_equal = self.model.addVars( (  (i, g, g_prime, u, v, n, m) for i in self.slices for g, g_prime in self.slice_group_pairs[i]
            for u, v in self.slice_requests[i].edges for n, m in self.phys_links), vtype=GRB.BINARY, name="is_link_placement_equal")
        
        # Add constraints to set binary variables based on placement conditions
        for i in self.slices:
            for g, g_prime in self.slice_group_pairs[i]:
                for u, v in self.slice_requests[i].edges:
                    for n, m in self.phys_links:
                        self.model.addConstr(is_link_placement_equal[i, g, g_prime, u, v, n, m] <= self.Y[i, g, u, v, n, m], 
                                        name=f"is_link_placement_equal_constr_1_{i}_{g}_{g_prime}_{u}_{v}_{n}_{m}")
                        self.model.addConstr(is_link_placement_equal[i, g, g_prime, u, v, n, m] <= self.Y[i, g_prime, u, v, n, m], 
                                        name=f"is_link_placement_equal_constr_2_{i}_{g}_{g_prime}_{u}_{v}_{n}_{m}")
                        self.model.addConstr(is_link_placement_equal[i, g, g_prime, u, v, n, m] >= self.Y[i, g, u, v, n, m] + self.Y[i, g_prime, u, v, n, m] - 1, 
                                        name=f"is_link_placement_equal_constr_3_{i}_{g}_{g_prime}_{u}_{v}_{n}_{m}")
                        self.model.addConstr( (is_link_placement_equal[i, g, g_prime, u, v, n, m] == 1) >>\
                                              (self.y[i, g, u, v] == self.y[i, g_prime, u, v]), 
                                    name=f"link_sharing_constr_{i}_{g}_{g_prime}_{u}_{v}_{n}_{m}")

    def fix_start_and_destination_nodes(self):
        """
        We want to ensure that the start and destination VNFs are mapped to the required physical nodes.
        Note mapping should only be enforced when the placement group is active.
        """
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                # Fix mapping for start node
                v, n = self.slice_requests[i].graph['node_s_mapping']
                self.model.addConstr( self.X[i, g, v, n] == self.is_group_active_var[i, g], name=f"fix_s_node_{i}_{g}")
                # Fix mapping for destination node
                v, n = self.slice_requests[i].graph['node_d_mapping']
                self.model.addConstr( self.X[i, g, v, n] == self.is_group_active_var[i, g], name=f"fix_d_node_{i}_{g}")

    # Capacity Constraints
    # Expression calculating the end-to-end delay for a placement group in a slice
    def calculate_delay_slice_e2e(self):
        return {(i, g): 
                    gp.quicksum(
                        self.Y[i, g, u, v, n, m] * self.phys_delay[n, m]
                        for (u, v) in self.slice_requests[i].edges for (n, m) in self.phys_links
                    )
                for i in self.slices for g in range(1, self.max_groups[i] + 1)}


    def latency_rule(self):
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                LHS = self.delay_slice_e2e[i, g]
                self.model.addConstr(LHS <= self.delay_req[i], name=f"latency_{i}_{g}")

    def slice_bw_rule(self):
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                for (u, v) in self.slice_requests[i].edges:
                    self.model.addConstr( (self.is_group_active_var[i, g] == 1) >> (self.y[i, g, u, v] >= self.v_bandwidth_demand[i, u, v]), 
                                    name=f"slice_bw_constraint_{i}_{g}_{u}_{v}")
                    

    def slice_node_res_rule(self):
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                for v in self.slice_requests[i].nodes:
                    for p in self.resources:
                        self.model.addConstr( (self.is_group_active_var[i, g] == 1) >> (self.x[i, g, v, p] >= self.v_resource_demand[i, v, p]), 
                                    name=f"slice_node_res_constraint_{i}_{g}_{v}_{p}")



    ### shared_node_res_capacity_rule
    def shared_node_res_capacity_rule(self):
        # Set allocated resource for each v_node to phys_node mapping (where it is not mapped it is 0)
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                for v in self.slice_requests[i].nodes:
                    for n in self.phys_nodes:
                        for p in self.resources:
                            self.model.addConstr(self.node_allocated_res_var[i, g, v, n, p] == self.x[i, g, v, p]*self.X[i, g, v, n], 
                                        name=f"set_node_allocated_res_var_{i}_{g}_{v}_{p}_{n}")
        
        # Create and set up an auxiliary variable that stores the amount of resource that 
        #    a given virtual node occupies on a given slice for a given physical node
        for n in self.phys_nodes:
            for i in self.slices:
                for v in self.slice_requests[i].nodes:
                    for p in self.resources:
                        max_val = gp.max_(
                            self.node_allocated_res_var[i, g, v, n, p]
                            for g in range(1, self.max_groups[i] + 1)
                        )
                        self.model.addConstr(self.max_node_allocated_res_per_slice_var[i, v, n, p] == max_val, 
                                        name=f'set_max_node_allocated_res_per_slice_var_{i}_{v}_{n}_{p}')
        
        # Create the constraint for every phyiscal node
        for n in self.phys_nodes:
            for p in self.resources:
                LHS = gp.quicksum(
                    self.max_node_allocated_res_per_slice_var[i, v, n, p]
                    for i in self.slices
                    for v in self.slice_requests[i].nodes
                )
                RHS = self.phys_resource_capacity[n, p]
                self.model.addConstr(LHS <= RHS, name=f'phys_node_resource_constr_{n}_{p}')


    ### shared_link_capacity_rule
    def shared_link_capacity_rule(self):
        # Set allocated bw for each v_link to phys_link mapping (where it is not mapped it is 0)
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                for u, v in self.slice_requests[i].edges:
                    for n, m in self.phys_links:
                        self.model.addConstr(self.link_allocated_bw_var[i, g, u, v, n, m] == self.y[i, g, u, v]*self.Y[i, g, u, v, n, m], 
                                        name=f"set_link_allocated_bw_var_{i}_{g}_{u}_{v}_{n}_{m}")
        
        # Create and set up an auxiliary variable that stores the amount of bandwidth that 
        #    a given virtual link occupies on a given slice for a given physical link
        for n, m in self.phys_links:
            for i in self.slices:
                for u, v in self.slice_requests[i].edges:
                    max_val = gp.max_(
                        self.link_allocated_bw_var[i, g, u, v, n, m]
                        for g in range(1, self.max_groups[i] + 1)
                    )
                    self.model.addConstr(self.max_link_allocated_bw_per_slice_var[i, u, v, n, m] == max_val, 
                                    name=f'set_max_link_allocated_bw_per_slice_var_{i}_{u}_{v}_{n}_{m}')
        
        # Create the constraint for every phyiscal link
        for n, m in self.phys_links:
            LHS = gp.quicksum(
                self.max_link_allocated_bw_per_slice_var[i, u, v, n, m]
                for i in self.slices
                for u, v in self.slice_requests[i].edges
            )
            RHS = self.phys_bandwidth_capacity[n, m]
            self.model.addConstr(LHS <= RHS, name=f'phys_link_bandwidth_constr_{n}_{m}')


    def deactivate_phantom_resource_allocation(self):
        """
        If (i, g) is not placed, i.e not active, then x and y values should be 0.
        """
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                # Constraint for bandwidth resources
                for u, v in self.slice_requests[i].edges:
                    self.model.addGenConstrIndicator(self.is_group_active_var[i, g], False, self.y[i, g, u, v] == 0, 
                                                name=f"deactivate_phantom_link_res_alloc_constr_{i}_{g}_{u}_{v}")

                # Constraint for node resources
                for v in self.slice_requests[i].nodes:
                    for p in self.resources:
                        self.model.addGenConstrIndicator(self.is_group_active_var[i, g], False, self.x[i, g, v, p] == 0, 
                                                    name=f"deactivate_phantom_node_res_alloc_constr_{i}_{g}_{v}_{p}")

    def set_min_node_availability_element(self):
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                for v in self.slice_requests[i].nodes:
                    for n in self.phys_nodes:
                        self.model.addConstr(self.min_node_availability_element[i, g, v, n] == (1 - self.X[i, g, v, n] * (1 - self.availability_node[n])),
                                    name=f"min_node_availability_element_rule_{i}_{g}_{v}_{n}")
                        
    def set_min_link_availability_element(self):
        for i in self.slices:
            for g in range(1, self.max_groups[i] + 1):
                for u, v in self.slice_requests[i].edges:
                    for n, m in self.phys_links:
                        self.model.addConstr(self.min_link_availability_element[i, g, u, v, n, m] == (1 - self.Y[i, g, u, v, n, m]*(1 - self.availability_link[n, m])),
                                    name=f"min_link_availability_element_rule_{i}_{g}_{u}_{v}_{n}_{m}")

    def min_node_availability_rule(self):
        for i in self.slices:
            for S in self.placement_group_subsets[i]:
                for n in self.phys_nodes:
                    min_node_avail = gp.min_(
                        self.min_node_availability_element[i, g, v, n]
                        for g in S
                        for v in self.slice_requests[i].nodes
                    )
                    self.model.addConstr(self.min_node_availability[i, S, n] == min_node_avail, 
                                        name=f"min_node_availability_rule_{i}_{str(set(S)).replace(' ', '')}_{n}")
    def min_link_availability_rule(self):
        for i in self.slices:
            for S in self.placement_group_subsets[i]:
                for n, m in self.phys_links:
                    min_link_avail = gp.min_(
                        self.min_link_availability_element[i, g, u, v, n, m]
                        for g in S
                        for u, v in self.slice_requests[i].edges
                    )
                    self.model.addConstr(self.min_link_availability[i, S, n, m] == min_link_avail, 
                                        name=f"min_link_availability_rule_{i}_{str(set(S)).replace(' ', '')}_{n}_{m}")

    def availability_rule(self):
        for i in self.slices:
            LHS_elements = []
            for S in self.placement_group_subsets[i]:
                sign = (-1)**(len(S) + 1)
                # Node availability product using auxiliary variables
                for idx, n in enumerate(self.phys_nodes):
                    if idx == 0:
                        self.model.addConstr(self.avail_node_aux_var[i, S, n] == self.min_node_availability[i, S, n], 
                                        name=f"avail_node_aux_var_constr_{i}_{str(set(S)).replace(' ', '')}_{n}")
                    else:
                        prev_n = self.phys_nodes[idx-1]
                        quad_term = self.avail_node_aux_var[i, S, prev_n] * self.min_node_availability[i, S, n]
                        self.model.addConstr(self.avail_node_aux_var[i, S, n] == quad_term, 
                                        name=f"avail_node_aux_var_constr_{i}_{str(set(S)).replace(' ', '')}_{n}")

                last_n = self.phys_nodes[-1]
                node_avail_prod = self.avail_node_aux_var[i, S, last_n]
                
                # Link availability product using auxiliary variables
                for idx, (n, m) in enumerate(self.phys_links):
                    if idx == 0:
                        self.model.addConstr(self.avail_link_aux_var[i, S, n, m] == self.min_link_availability[i, S, n, m], 
                                        name=f"avail_link_aux_var_constr_{i}_{str(set(S)).replace(' ', '')}_{n}_{m}")
                    else:
                        prev_n, prev_m = self.phys_links[idx-1]
                        quad_term = self.avail_link_aux_var[i, S, prev_n, prev_m] * self.min_link_availability[i, S, n, m]
                        self.model.addConstr(self.avail_link_aux_var[i, S, n, m] == quad_term,
                                    name=f"avail_link_aux_var_constr_{i}_{str(set(S)).replace(' ', '')}_{n}_{m}")

                last_n, last_m = self.phys_links[-1]
                link_avail_prod = self.avail_link_aux_var[i, S, last_n, last_m]
                # Calculate the +- node_availability * link_availability
                #    and store it in a variable
                node_link_prod = sign * node_avail_prod * link_avail_prod
                self.model.addConstr(self.avail_node_link_aux_var[i, S] == node_link_prod, 
                                        name=f"avail_node_link_aux_var_constr_{i}_{str(set(S)).replace(' ', '')}")
                
                LHS_elements.append(self.is_S_active_var[i, S]*self.avail_node_link_aux_var[i, S])
            LHS = gp.quicksum(LHS_elements)
            self.model.addConstr(LHS >= self.availability_req[i], name=f"availability_{i}")

    # Variable that stores the QoS value for each resource type (and bw and delay)
    def create_qos_res_i_g_var(self):
        all_resources = [*self.resources, 'bw', 'delay']
        QoS_res_i_g_var = self.model.addVars(( (i, g, p) for i in self.slices 
                                        for g in range(1, self.max_groups[i]+1) for p in all_resources), 
                                        vtype=GRB.CONTINUOUS, lb=0, name=f"qos_res_i_g_var")
        for i, g, p in QoS_res_i_g_var.keys():
            if p == 'bw':
                constr = QoS_res_i_g_var[i, g, p]  == gp.quicksum( self.y[i, g, u, v]  for u, v in self.slice_requests[i].edges) / \
                                        sum( self.v_bandwidth_demand[i, u, v]  for u, v in self.slice_requests[i].edges)
            elif p == 'delay':
                constr = QoS_res_i_g_var[i, g, p]  == self.delay_slice_e2e[i, g] / self.delay_req[i] 
            else:
                constr = QoS_res_i_g_var[i, g, p]  == gp.quicksum( self.x[i, g, v, p]  for v in self.slice_requests[i].nodes) / \
                                        sum( self.v_resource_demand[i, v, p]  for v in self.slice_requests[i].nodes)
            self.model.addConstr(constr, name=f"set_qos_res_i_g_var_{i}_{g}_{p}")
        return QoS_res_i_g_var

    # def create_QoE_res_i_g_var(self):
    #     # Since Gurobi does not allow to create logarithm or exponential functions
    #     #   using expressions only variables. We need to create aux variables that denote the
    #     #   1) input to the nonlinear function
    #     #   2) output of the nonlinear function
    #     #   3) the Lin expression using the output of the nonlinear function
    #     # NOTE: We have to set the lower bound of the continous variables, since the default is 0!!!
    #     # NOTE: The input for the log/exp function should be in the range of input_aux_var

    #     # 1) Create the input of the nonlinear function
    #     input_aux_var = self.model.addVars(self.QoS_res_i_g.keys(), vtype=GRB.CONTINUOUS, lb=-100, ub=+100,
    #                                 name=f"qoe_res_i_g_func_input_var")
    #     for i, g, p in input_aux_var.keys():
    #         gamma, lambda_ = self.object_coeffs['gamma'][i, p], self.object_coeffs['lambda'][i, p]
    #         linexpr = gamma*self.QoS_res_i_g[i, g, p] + lambda_
    #         # If (i, g) is not active, set the input for the log/exp function to 1
    #         self.model.addConstr(input_aux_var[i, g, p] == (linexpr - 1)*self.is_group_active_var[i, g] + 1, 
    #                         name=f"set_qoe_res_i_g_func_input_var_{i}_{g}_{p}")

    #     # 2 and 3) Create the output of the nonlinear function and create the final QoE_res_i_g_var
    #     output_aux_var = self.model.addVars(self.QoS_res_i_g.keys(), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY,
    #                                 name=f"qoe_res_i_g_func_output_var")
    #     qoe_res_i_g_var = self.model.addVars(self.QoS_res_i_g.keys(), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY,
    #                                     name=f"qoe_res_i_g_var")
    #     for i, g, p in self.QoS_res_i_g.keys():
    #         # is p subject to IQX or WFL?
    #         if p == 'delay':
    #             # IQX
    #             self.model.addGenConstrExp(input_aux_var[i, g, p], output_aux_var[i, g, p], 
    #                                 name=f"set_qoe_res_i_g_func_output_var_{i}_{g}_{p}")
    #         else:
    #             # WFL
    #             self.model.addGenConstrLog(input_aux_var[i, g, p], output_aux_var[i, g, p], 
    #                                 name=f"set_qoe_res_i_g_func_output_var_{i}_{g}_{p}")

    #         beta, mu = self.object_coeffs['beta'][i, p], self.object_coeffs['mu'][i, p]
    #         linexpr = beta*output_aux_var[i, g, p] + mu
    #         self.model.addConstr(qoe_res_i_g_var[i, g, p] == (linexpr)*self.is_group_active_var[i, g], 
    #                         name=f"set_qoe_res_i_g_var_{i}_{g}_{p}")
    #     return qoe_res_i_g_var
    

    # def create_QoE_res_i_g_var(self):
    #     # Create the QoS --> QoE variable for every i, g, p
    #     qoe_res_i_g_var = self.model.addVars(self.QoS_res_i_g.keys(), vtype=GRB.CONTINUOUS, 
    #                                          lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"qoe_res_i_g_var")
    #     for i, g, p in self.QoS_res_i_g.keys():
    #         # Extract the coefficients for the nonlinear function
    #         _beta = self.object_coeffs['beta'][i, p]
    #         _gamma = self.object_coeffs['gamma'][i, p]
    #         _lambda = self.object_coeffs['lambda'][i, p]
    #         _mu = self.object_coeffs['mu'][i, p]
    #         # Define the WFL and IQX functions
    #         def WFL(x):
    #             return _beta * np.log(_gamma * x + _lambda) + _mu
    #         def IQX(x):
    #             return _beta * np.exp(_gamma * x + _lambda) + _mu
    #         # Depending on whether the p resource is subject to the WFL or IQX we create the PWL approximation
    #         if p == 'delay':
    #             # IQX
    #             # Create the PWL function from QoS = 1 (barely satisfied delay requirement)
    #             # With pwl_step width steps, until we reach QoE = 5 then it will become constant
    #             PWL_points = []
    #             pwl_step = 0.2

    #             qos = 1.0
    #             while True:
    #                 y = IQX(qos)
    #                 PWL_points.append( (qos, y) )
    #                 qos -= pwl_step
    #                 if y >= 5:
    #                     break

    #             # Insert first and last element to make it constant outside the QoS bound
    #             PWL_points.append( (qos, y) )   
    #             PWL_points.insert(0, (1+pwl_step, PWL_points[0][1]))
    #             PWL_points = sorted(PWL_points, key=lambda elem: elem[0], reverse=False)
    #             x_points = [p[0] for p in PWL_points]
    #             y_points = [p[1] for p in PWL_points]
    #         else:
    #             # WFL
    #             PWL_points = []
    #             pwl_step = 0.2
    #             # Create the PWL function from QoS = 1 (minimally satisfied resources)
    #             # With pwl_step width steps, until we reach QoE = 5 then it will become almost constant
    #             qos = 1.0
    #             while True:
    #                 y = WFL(qos)
    #                 PWL_points.append( (qos, y) )
    #                 qos += pwl_step
    #                 if y >= _mu:
    #                     break

    #             # Insert first and last element to make it constant outside the QoS bound
    #             PWL_points.append( (qos, y) )   
    #             PWL_points.insert(0, (0, PWL_points[0][1]))
    #             x_points = [p[0] for p in PWL_points]
    #             y_points = [p[1] for p in PWL_points]
    #         # print('-'*5, i, g, p, '-'*5)
    #         # print(_beta, _gamma, _lambda, _mu)
    #         # print(x_points, y_points)
    #         self.model.addGenConstrPWL(xvar=self.QoS_res_i_g[i, g, p], 
    #                                    yvar=qoe_res_i_g_var[i, g, p], 
    #                                    xpts=x_points, 
    #                                    ypts=y_points, 
    #                                    name=f"set_qoe_res_i_g_var_{i}_{g}_{p}")
    #     return qoe_res_i_g_var

    def _get_PWL_points(self, func, _beta, _gamma, _lambda, _mu):
        if func == "WFL":
            if (_beta, _gamma, _lambda, _mu) == (1.8, 1, -0.93, 4.8):
                x_points = [0, 1, 1.054,1.1460000000000001,1.3060000000000003,1.5840000000000005,2.025999999999998, 10]
                y_points = [0.06644290468303993, 0.06644290468303993, 1.0929236484501832, 2.092069370851765, 3.089839199714571,
                            4.079255467658234, 5.000982800992693, 5.000982800992693]
            elif (_beta, _gamma, _lambda, _mu) == (0.1, 1, 0, 5):
                x_points = [1, 2]
                y_points = [5, 5]
            else:
                raise ValueError()
        elif func == "IQX":
            if (_beta, _gamma, _lambda, _mu) == (1, -3.9, 1.6, -0.1):
                x_points = [-1, 0, 0.21800000000000017, 0.3720000000000003, 0.5900000000000004, 0.8, 1, 2]
                y_points = [4.801704346288444, 4.801704346288444, 1.9665461168943048, 1.0106913835328302, 0.3523945388956018, 0.1187, 0, 0]
            elif (_beta, _gamma, _lambda, _mu) == (1, -1, -2, 4.9):
                x_points = [0, 1]
                y_points = [5, 5]
            else:
                raise ValueError()
        else:
            raise ValueError()
        return x_points, y_points

    def create_QoE_res_i_g_var(self):
        # Create the QoS --> QoE variable for every i, g, p
        qoe_res_i_g_var = self.model.addVars(self.QoS_res_i_g.keys(), vtype=GRB.CONTINUOUS, 
                                             lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"qoe_res_i_g_var")
        for i, g, p in self.QoS_res_i_g.keys():
            # Extract the coefficients for the nonlinear function
            _beta = self.object_coeffs['beta'][i, p]
            _gamma = self.object_coeffs['gamma'][i, p]
            _lambda = self.object_coeffs['lambda'][i, p]
            _mu = self.object_coeffs['mu'][i, p]

            # Depending on whether the p resource is subject to the WFL or IQX we create the PWL approximation
            if p == 'delay':
                # IQX
                x_points, y_points = self._get_PWL_points("IQX", _beta, _gamma, _lambda, _mu)
            else:
                # WFL
                x_points, y_points = self._get_PWL_points("WFL", _beta, _gamma, _lambda, _mu)
            # print('-'*5, i, g, p, '-'*5)
            # print(_beta, _gamma, _lambda, _mu)
            # print(x_points, y_points)
            self.model.addGenConstrPWL(xvar=self.QoS_res_i_g[i, g, p], 
                                       yvar=qoe_res_i_g_var[i, g, p], 
                                       xpts=x_points, 
                                       ypts=y_points, 
                                       name=f"set_qoe_res_i_g_var_{i}_{g}_{p}")
        return qoe_res_i_g_var
    
    # def create_QoE_res_i_g_var(self):
    #     # Create the QoS --> QoE variable for every i, g, p
    #     qoe_res_i_g_var = self.model.addVars(self.QoS_res_i_g.keys(), vtype=GRB.CONTINUOUS, 
    #                                          lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"qoe_res_i_g_var")
    #     for i, g, p in self.QoS_res_i_g.keys():
    #         # Extract the coefficients for the nonlinear function
    #         _beta = self.object_coeffs['beta'][i, p]
    #         _gamma = self.object_coeffs['gamma'][i, p]
    #         _lambda = self.object_coeffs['lambda'][i, p]
    #         _mu = self.object_coeffs['mu'][i, p]

    #         # Depending on whether the p resource is subject to the WFL or IQX we create the PWL approximation
    #         if p == 'delay':
    #             # IQX
    #             if (_beta, _gamma, _lambda, _mu) == (1, -3.9, 1.6, -0.1):
    #                 self.model.addConstr(qoe_res_i_g_var[i, g, p] == 1 - self.QoS_res_i_g[i, g, p], 
    #                                      name=f"set_qoe_res_i_g_var_{i}_{g}_{p}")
    #             elif (_beta, _gamma, _lambda, _mu) == (1, -1, -2, 4.9):
    #                 self.model.addConstr(qoe_res_i_g_var[i, g, p] == 5, name=f"set_qoe_res_i_g_var_{i}_{g}_{p}")
    #             else:
    #                 raise ValueError()
    #         else:
    #             # WFL
    #             if (_beta, _gamma, _lambda, _mu) == (1.8, 1, -0.93, 4.8):
    #                 self.model.addConstr(qoe_res_i_g_var[i, g, p] == 5*self.QoS_res_i_g[i, g, p] - 5, 
    #                                      name=f"set_qoe_res_i_g_var_{i}_{g}_{p}")
    #             elif (_beta, _gamma, _lambda, _mu) == (0.1, 1, 0, 5):
    #                 self.model.addConstr(qoe_res_i_g_var[i, g, p] == 5, name=f"set_qoe_res_i_g_var_{i}_{g}_{p}")
    #             else:
    #                 raise ValueError()
    #     return qoe_res_i_g_var

    # Create the inverse of this var
    def create_active_groups_per_slice_inverse_var(self):
        active_groups_per_slice_inverse_var = self.model.addVars(self.active_groups_per_slice_var.keys(),
                                    vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"active_groups_per_slice_inverse_var")
        for i in active_groups_per_slice_inverse_var:
            self.model.addConstr(active_groups_per_slice_inverse_var[i] * self.active_groups_per_slice_var[i] == 1, 
                            name=f"active_groups_per_slice_inverse_constr_{i}")
        return active_groups_per_slice_inverse_var

    def O2_expression(self):
        expression = gp.quicksum(
            gp.quicksum(
                self.max_node_allocated_res_per_slice_var[i, v, n, p]
                for v in self.slice_requests[i].nodes
                for p in self.resources
                for n in self.phys_nodes
            )
            +
            gp.quicksum(
                self.max_link_allocated_bw_per_slice_var[i, u, v, n, m]
                for u, v in self.slice_requests[i].edges
                for n, m in self.phys_links
            )
            for i in self.slices
        )
        return expression

    def O3_expression(self):
        # Create aux vars that stores whether a node or link is used for hosting virtual node/link
        is_n_node_used = self.model.addVars(self.phys_nodes, 
                    vtype=GRB.BINARY, name=f"is_n_node_used")
        is_nm_link_used = self.model.addVars(self.phys_links, 
                        vtype=GRB.BINARY, name=f"is_nm_link_used")
        # Setting the values of these vars
        for n in self.phys_nodes:
            max_val = gp.max_(
                self.X[i, g, v, n]
                for i in self.slices for g in range(1, self.max_groups[i]+1) for v in self.slice_requests[i].nodes
            )
            self.model.addConstr(is_n_node_used[n] == max_val, name=f"set_is_n_node_used_{n}")
        for n, m in self.phys_links:
            max_val = gp.max_(
                self.Y[i, g, u, v, n, m]
                for i in self.slices for g in range(1, self.max_groups[i]+1) for u, v in self.slice_requests[i].edges
            )
            self.model.addConstr(is_nm_link_used[n, m] == max_val, name=f"set_is_nm_link_used_{n}_{m}")
        # Calculating the 3rd part of the objective function the number of used nodes and links
        used_nodes = gp.quicksum(
            is_n_node_used[n]
            for n in self.phys_nodes
        )
        used_links = gp.quicksum(
            is_nm_link_used[n, m]
            for n, m in self.phys_links
        )
        return used_nodes + used_links

    def O4_expression(self):
        """
        Since it is possible to create a placement group which is redundant to a previous one without decreasing the Obj value.
        Because it will not use more nodes and links, and more resources (due to link sharing).
        Due to the precision errors the QoE value could be higher.
        """
        return self.is_group_active_var.sum()


    def optimize(self):
        # Optimize the model
        self.model.optimize()
    
    def save(self, prefix):
        # Retrieve model status, and check if optimal
        model_status = self.model.status
        is_feasible = hasattr(self.model, 'ObjVal') and self.model.ObjVal != -float('inf') and self.model.ObjVal != float('inf')

        elapsed_time = self.model.Runtime
        objective_value = self.model.getObjective().getValue() if is_feasible else None
        objective_1_value = self.O1_expr.getValue() if is_feasible else None
        objective_2_value = self.O2_expr.getValue() if is_feasible else None
        objective_3_value = self.O3_expr.getValue() if is_feasible else None
        objective_4_value = self.O4_expr.getValue() if is_feasible else None
        stats = {"elapsed_time" : elapsed_time, "model_status" : model_status, 
                 "objective_value" : objective_value, 
                 "objective_1_value" : objective_1_value, "objective_2_value" : objective_2_value, 
                 "objective_3_value" : objective_3_value, "objective_4_value" : objective_4_value}

        with open(f'{prefix}_stats.json', 'w') as f:
            json.dump(stats, f)
        self.model.write(f"{prefix}_model.json")
        self.model.write(f"{prefix}_model.mps")
        if is_feasible:
            # Solution can only be saved if it was optimal
            self.model.write(f"{prefix}_model.sol")
