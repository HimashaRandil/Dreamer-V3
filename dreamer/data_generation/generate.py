import os
import numpy as np
import pandas as pd
import grid2op
from lightsim2grid import LightSimBackend
from dreamer.Utils.customReward import LossReward, MarginReward
from dreamer.Utils.converter import ActionConverter
#from .topology_agent import TopologyGreedy, TopologyRandom
from grid2op.Agent import TopologyGreedy
from grid2op.Reward import L2RPNSandBoxScore
from grid2op.Exceptions import *
import time
import random
from collections import defaultdict



class DataGeneration:
    def __init__(self, config) -> None:
        self.env = grid2op.make(config.env_name, reward_class=L2RPNSandBoxScore,
                                backend=LightSimBackend())
        self.config = config
        self.action_converter = ActionConverter(self.env)
        self.agent = TopologyGreedy(self.env.action_space)
        #self.agent = TopologyRandom(self.env)

    def topology_data_generation(self):
        num_episodes = len(self.env.chronics_handler.subpaths)
        
        steps = []
        obs_data = []
        action_data = []
        obs_next_data = []
        reward_data = []
        done_data = []

        for episode_id in range(num_episodes):
            print(f"Episode ID : {episode_id}")
            self.env.set_id(episode_id)
            obs = self.env.reset()
            reward = self.env.reward_range[0]
            done = False
            

            for i in range (self.env.max_episode_duration()):
                
                action = self.agent.act(obs, reward, done) #self.agent.act() #
                obs_, reward, done, _ = self.env.step(self.env.action_space({}))
                action_idx = self.action_converter.action_idx(action)

                # Append data for this step
                obs_data.append(obs.to_vect())
                action_data.append(action_idx)
                obs_next_data.append(obs_.to_vect())
                reward_data.append(reward)
                done_data.append(done)
                steps.append(i)

                # Update observation for the next step
                obs = obs_

                

                if done:
                    self.env.set_id(episode_id)
                    
                    obs = self.env.reset()
                    done = False
                    reward = self.env.reward_range[0]

                    self.env.fast_forward_chronics(i - 1)
                    
                    action = self.agent.act(obs, reward, done)
                    obs_, reward, done, _ = self.env.step(self.env.action_space({}))
                    action_idx = self.action_converter.action_idx(action)

                    obs_data.append(obs.to_vect())
                    action_data.append(action_idx)
                    obs_next_data.append(obs_.to_vect())
                    reward_data.append(reward)
                    done_data.append(done)
                    steps.append(i)

                    obs = obs_
                

        # Convert lists to np.array
        obs_data = np.array(obs_data, dtype=object)
        action_data = np.array(action_data, dtype=int)
        obs_next_data = np.array(obs_next_data, dtype=object)
        reward_data = np.array(reward_data, dtype=float)
        done_data = np.array(done_data, dtype=bool)
        steps_data = np.array(steps, dtype=int)

        # Save to .npz file
        np.savez("dreamer\\data_generation\\data_generation_output.npz", obs=obs_data, action=action_data, 
                 obs_next=obs_next_data, reward=reward_data, done=done_data, steps=steps_data)
            

    

    def generate_random_data(self):
        obs_data = []
        action_data = []
        obs_next_data = []
        reward_data = []
        done_data = []

        for i in range(self.config.episode_num):
            done = False
            reward = self.env.reward_range[0]
            obs = self.env.reset()

            while not done:
                action = self.agent.act(obs, reward, done)
                obs_, reward, done, _ = self.env.step(action)

                action_idx = self.action_converter.action_idx(action)

                # Append data for this step
                obs_data.append(obs.to_vect())
                action_data.append(action_idx)
                obs_next_data.append(obs_.to_vect())
                reward_data.append(reward)
                done_data.append(done)

                if done:
                    print("break")
                    break
        # Convert lists to np.array
        obs_data = np.array(obs_data, dtype=object)
        action_data = np.array(action_data, dtype=int)
        obs_next_data = np.array(obs_next_data, dtype=object)
        reward_data = np.array(reward_data, dtype=float)
        done_data = np.array(done_data, dtype=bool)

        # Save to .npz file
        np.savez("dreamer\\data_generation\\data_generation_output.npz", obs=obs_data, action=action_data, 
                 obs_next=obs_next_data, reward=reward_data, done=done_data)
        


    def topology_search(self, dst_step):
        obs = self.env.get_obs()
        min_rho, overflow_id = obs.rho.max(), obs.rho.argmax()
        print("step-%s, line-%s(from bus-%d to bus-%d) overflows, max rho is %.5f" %
            (dst_step, overflow_id, self.env.line_or_to_subid[overflow_id],
            self.env.line_ex_to_subid[overflow_id], obs.rho.max()))
        all_actions = self.env.action_space.get_all_unitary_topologies_change(self.env.action_space)
        action_chosen = self.env.action_space({})
        tick = time.time()
        for action in all_actions:
            if not self.env._game_rules(action, self.env):
                continue
            obs_, _, done, _ = obs.simulate(action)
            if (not done) and (obs_.rho.max() < min_rho):
                min_rho = obs_.rho.max()
                action_chosen = action
        print("find a greedy action and max rho decreases to %.5f, search duration: %.2f" %
            (min_rho, time.time() - tick))
        return action_chosen
    


    def get_substation_connections(env):
        """
        Returns a dictionary with each substation ID and the number of powerlines connected to it.

        Parameters:
        - env: Grid2Op environment instance

        Returns:
        - substation_connections: Dictionary with substation ID as key and connection count as value
        """
        # Initialize a dictionary to store the number of connections for each substation
        substation_connections = defaultdict(int)

        # Retrieve powerline-to-substation mappings
        line_or_to_subid = env.line_or_to_subid  # Array of origin substations for each powerline
        line_ex_to_subid = env.line_ex_to_subid  # Array of extremity substations for each powerline

        # Count connections for each substation
        for line_id in range(env.n_line):
            origin_substation = line_or_to_subid[line_id]
            extremity_substation = line_ex_to_subid[line_id]
            
            # Increment the connection count for each substation
            substation_connections[origin_substation] += 1
            substation_connections[extremity_substation] += 1

        return substation_connections


    def find_powerlines_connected_to_substations(env, target_substations):
        """
        Finds all powerline IDs that are connected to the given substations.

        Parameters:
        - env: Grid2Op environment instance
        - target_substations: List of substation IDs to check for connections

        Returns:
        - connected_powerlines: Dictionary where each substation ID is a key and 
        the value is a list of powerline IDs connected to that substation
        """
        connected_powerlines = {sub: [] for sub in target_substations}

        # Retrieve powerline-to-substation mappings
        line_or_to_subid = env.line_or_to_subid  # Origin substation for each powerline
        line_ex_to_subid = env.line_ex_to_subid  # Extremity substation for each powerline

        # Loop through each powerline to find connections to target substations
        for line_id in range(env.n_line):
            origin_substation = line_or_to_subid[line_id]
            extremity_substation = line_ex_to_subid[line_id]

            # Check if origin or extremity matches any target substation
            if origin_substation in target_substations:
                connected_powerlines[origin_substation].append(line_id)
            if extremity_substation in target_substations:
                connected_powerlines[extremity_substation].append(line_id)

        return connected_powerlines


    def teacher_generation(self):
        DATA_PATH = f'C:\\Users\\Ernest\\data_grid2op\\{self.config.env_name}'  # for demo only, use your own dataset
        SCENARIO_PATH = f'C:\\Users\\Ernest\\data_grid2op\\{self.config.env_name}'
        SAVE_PATH = 'dreamer\\data_generation'
        LINES2ATTACK = [0, 1, 5, 7, 9, 16, 17, 4, 9, 13, 14, 18]
        NUM_EPISODES = 1  # each scenario runs 100 times for each attack (or to say, sample 100 points)

        for episode in range(NUM_EPISODES):
            # traverse all attacks
            for line_to_disconnect in LINES2ATTACK:
                try:
                    # if lightsim2grid is available, use it.
                    from lightsim2grid import LightSimBackend
                    backend = LightSimBackend()
                    env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend)
                except:
                    env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH)
                env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
                # traverse all scenarios
                for chronic in range(len(os.listdir(SCENARIO_PATH))):
                    env.reset()
                    dst_step = episode * 72 + random.randint(0, 72)  # a random sampling every 6 hours
                    print('\n\n' + '*' * 50 + '\nScenario[%s]: at step[%d], disconnect line-%d(from bus-%d to bus-%d]' % (
                        env.chronics_handler.get_name(), dst_step, line_to_disconnect,
                        env.line_or_to_subid[line_to_disconnect], env.line_ex_to_subid[line_to_disconnect]))
                    # to the destination time-step
                    env.fast_forward_chronics(dst_step - 1)
                    obs, reward, done, _ = env.step(env.action_space({}))
                    if done:
                        break
                    # disconnect the targeted line
                    new_line_status_array = np.zeros(obs.rho.shape, dtype=np.int32)
                    new_line_status_array[line_to_disconnect] = -1
                    action = env.action_space({"set_line_status": new_line_status_array})
                    obs, reward, done, _ = env.step(action)
                    if obs.rho.max() < 1:
                        # not necessary to do a dispatch
                        continue
                    else:
                        # search a greedy action
                        action = self.topology_search(env)
                        obs_, reward, done, _ = env.step(action)