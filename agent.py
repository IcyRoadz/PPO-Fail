from atexit import register
import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray import train, tune

from env import SimpleSim

ray.init()

def env_creator(env_config):
    """
    This function creates our custom environment.
    """
    return SimpleSim()

register_env("ActionMatcherSim", env_creator)

config = PPOConfig().training(lr=tune.grid_search([1e-2, 1e-3, 1e-4, 1e-5])).environment(env="ActionMatcherSim")#.resources(num_gpus=1)
# Stuff I've tried tweaking:
#exploration_config={
                                  #"type": "EpsilonGreedy",
                                  #"initial_epsilon": 1.0,
                                  #"final_epsilon": 0.00,
                                  #"epsilon_timesteps": 10000,  # Number of steps over which epsilon is reduced
                              #}
#}).training(train_batch_size=1000, sgd_minibatch_size=35, model={"fcnet_hiddens": [5]}, lr=tune.grid_search([0.1, 0.01, 1e-8, 1e-10])).environment(env="PackageMatcherSim").resources(num_gpus=1)
#rain_batch_size=30, sgd_minibatch_size=15, 

# Configure checkpointing
checkpoint_dir = "/mnt/cluster_storage/ppo/action_matcher"
os.makedirs(checkpoint_dir, exist_ok=True)
tuner = tune.Tuner(
    "PPO",
    run_config=train.RunConfig(
        stop={"episode_reward_mean": 1},
        local_dir=checkpoint_dir,
        verbose=1
    ),
    param_space=config.to_dict(),
)

tuner.fit()
