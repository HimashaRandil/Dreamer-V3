import grid2op
from grid2op import Environment
import numpy as np

class ActionConverter:
    def __init__(self, env:Environment) -> None:
        self.action_space = env.action_space
        self.env = env
        self.sub_mask = []
        self.init_sub_topo()
        self.init_action_converter()

    def init_sub_topo(self):
        self.subs = np.flatnonzero(self.action_space.sub_info)
        self.sub_to_topo_begin, self.sub_to_topo_end = [], [] # These lists will eventually store the starting and ending indices, respectively, for each actionable substation's topology data within the environment's overall topology information.
        idx = 0 # This variable will be used to keep track of the current position within the overall topology data
        
        for num_topo in self.action_space.sub_info: # The code can efficiently extract the relevant portion of the overall topology data that specifically applies to the given substation
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)

    def init_action_converter(self):
        self.actions = [self.env.action_space({})]
        self.n_sub_actions = np.zeros(len(self.action_space.sub_info), dtype=int)
        for i, sub in enumerate(self.subs):
            
            # Generating Topology Actions
            topo_actions = self.action_space.get_all_unitary_topologies_set(self.action_space, sub) # retrieves all possible topology actions for the current substation using the get_all_unitary_topologies_set method of the action_space object
            self.actions += topo_actions  # Appends the topology actions for the current substation to the actions list.
            self.n_sub_actions[i] = len(topo_actions) # Stores the number of topology actions for the current substation in the n_sub_actions array
            self.sub_mask.extend(range(self.sub_to_topo_begin[sub], self.sub_to_topo_end[sub])) # Extends the sub_mask list with indices corresponding to the topologies of the current substation.
        
        self.sub_pos = self.n_sub_actions.cumsum() 
        self.n = sum(self.n_sub_actions)

    def act(self, action:int):
        return self.actions[action]
    
    def action_idx(self, action):
        return self.actions.index(action)