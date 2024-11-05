import os
import numpy
import grid2op
import random
import pandas as pd
import time



def topology_selective_action(env, substation_ids):
    """
    Attempts to stabilize the grid by applying topology actions on specified substations.
    
    Parameters:
    - env: Grid2Op environment instance
    - substation_ids: List of IDs of substations to apply topology actions on

    Returns:
    - action_chosen: The best topology action found to stabilize the grid, or a "do nothing" action if none is found
    """
    actions = []
    obs = env.get_obs()
    min_rho = obs.rho.max()
    action_chosen = env.action_space({})  # start with a "do nothing" action

    actions_1 = env.action_space.get_all_unitary_topologies_change(env.action_space, sub_id=substation_ids[0])
    actions_2 = env.action_space.get_all_unitary_topologies_change(env.action_space, sub_id=substation_ids[1])
    all_actions = actions_1 + actions_2

    
    # Start time tracking for performance measurement
    start_time = time.time()

    # Loop through all possible unitary topology changes
    for action in all_actions:
        
        # Check if the action is legal
        if not env._game_rules(action, env):
            continue

        # Simulate the action
        obs_, _, done, _ = obs.simulate(action)
        actions.append(action)
        
        # If the grid is not in a failed state and the action reduces congestion
        if (not done) and (obs_.rho.max() < min_rho):
            min_rho = obs_.rho.max()
            action_chosen = action  # Update the best action

    # Logging the best action found
    print("Greedy topology action found with max rho reduced to %.5f, search duration: %.2f seconds" %
          (min_rho, time.time() - start_time))
    
    return action_chosen, len(actions)