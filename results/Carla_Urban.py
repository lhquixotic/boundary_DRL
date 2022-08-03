import os
import sys
from os.path import dirname,abspath
import copy
import random
sys.path.append(dirname(dirname(abspath(__file__))))

import gym_carla
import gym
import carla

from utilities.data_structures.Config import Config
import matplotlib.pyplot as plt

from agents.Trainer import Trainer
from agents.DQN_agents.DQN import DQN
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.AIf_agents.AIf import AIf
from agents.DQN_agents.DRQN import DRQN


env_params = {
'number_of_vehicles': 100,
'number_of_walkers': 0,
'display_size': 256,  # screen size of bird-eye render
'max_past_step': 1,  # the number of past steps to draw
'dt': 0.1,  # time interval between two frames
'discrete': True,  # whether to use discrete control space
# 'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
# 'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
'discrete_acc': [-3.0, 0.0,1.0,2.0],  # discrete value of accelerations
'discrete_steer': [-0.2, -0.15,-0.1,-0.05, 0.0,0.05, 0.1,0.15, 0.2],  # discrete value of steering angles
'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
'port': 2000,  # connection port
'town': 'Town03',  # which town to simulate
'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
'max_time_episode': 500,  # maximum timesteps per episode
'max_waypt': 12,  # maximum number of waypoints
'obs_range': 32,  # observation range (meter)
'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
'd_behind': 12,  # distance behind the ego vehicle (meter)
'out_lane_thres': 2.0,  # threshold for out of lane
'desired_speed': 8,  # desired speed (m/s)
'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
'display_route': True,  # whether to render the desired route
'pixor_size': 64,  # size of the pixor labels
'pixor': False,  # whether to output PIXOR observation
'use_boundary': True, # whether to use boundary
'boundary_dist' : 12, # if use boundary, boundary dist is the detected dist
'boundary_size' : 360, # points on the boundary
'lane_boundary_dist' : 12, # lane boundary distance
'no_rendering': True, # no rendering mode
}

config = Config()
config.seed = 1
config.environment = gym.make('carla-v0', params=env_params)
config.num_episodes_to_run = 450
config.file_to_save_data_results = "results/data_and_graphs/Carla_Env_Results_Data.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/Carla_Env_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "buffer_size": 40000,
        "epsilon": 0.1,
        "epsilon_decay_rate_denominator": 0.999,
        "discount_rate": 0.99,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 25, 
        "linear_hidden_units": [128,64],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 1,
        "clip_rewards": False
    },
    "AIf_Agents":{
        "batch_size": 64,
        "buffer_size": 65536,
        "tra_lr": 0.001,
        "pol_lr": 0.001,
        "val_lr": 0.001,
        "gamma" : 1.0,
        "beta" : 0.99,
        "update_every_n_steps": 25,
        "clip_rewards": False
    },
    "Actor_Critic_Agents":  {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

class CarlaEnvTrainer(Trainer):
    def run_games_for_agent(self, agent_number, agent_class):
        """Runs a set of games for a given agent, saving the results in self.results"""
        agent_results = []
        agent_name = agent_class.agent_name
        agent_group = self.agent_to_agent_group[agent_name]
        agent_round = 1
        for run in range(self.config.runs_per_agent):
            agent_config = self.config
            # if self.environment_has_changeable_goals(agent_config.environment) and self.agent_cant_handle_changeable_goals_without_flattening(agent_name):
            #     print("Flattening changeable-goal environment for agent {}".format(agent_name))
            #     agent_config.environment = gym.wrappers.FlattenDictWrapper(agent_config.environment,
                                                                        #    dict_keys=["observation", "desired_goal"])

            if self.config.randomise_random_seed: agent_config.seed = random.randint(0, 2**32 - 2)
            if run == 0:agent_config.hyperparameters = agent_config.hyperparameters[agent_group]
            print("AGENT NAME: {}".format(agent_name))
            print("\033[1m" + "{}.{}: {}".format(agent_number, agent_round, agent_name) + "\033[0m", flush=True)
            agent = agent_class(agent_config)
            self.environment_name = agent.environment_title
            print(agent.hyperparameters)
            print("RANDOM SEED " , agent_config.seed)
            game_scores, rolling_scores, time_taken = agent.run_n_episodes()
            print("Time taken: {}".format(time_taken), flush=True)
            self.print_two_empty_lines()
            agent_results.append([game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores), time_taken])
            if self.config.visualise_individual_results:
                self.visualise_overall_agent_results([rolling_scores], agent_name, show_each_run=True)
                plt.show()
            agent_round += 1
        self.results[agent_name] = agent_results

if __name__ == "__main__":
    AGENTS = [DRQN]
    trainer =  CarlaEnvTrainer(config,AGENTS)
    trainer.run_games_for_agents()
