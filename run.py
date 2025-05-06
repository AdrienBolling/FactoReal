import argparse
import os

from src.config.config import make_config
from src.utils.wandb_utils import init_wandb

from stable_baselines3 import A2C, PPO, DQN
from src.envs.FactoReal import FactoReal

import wandb

# This script is used to run the training and evaluation of the model.
# It will parse the arguments and run the training and evaluation.

def parse_args():
    """
    Parse the base arguments to run the job.
    """
    
    parser = argparse.ArgumentParser(description="Run the training and evaluation of the model.")
    parser.add_argument(
        "--config_type",
        type=str,
        default="default",
        help="Type of configuration to use. Options: default, simple.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="FactoReal",
        help="Name of the wandb project.",
    )
    parser.add_argument(
        "--entity_name",
        type=str,
        default="bolling.adrien",
        help="Name of the wandb entity.",
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        default="A2C",
        help="Baseline to use. Options: A2C, PPO, DQN.",
    )
    parser.add_argument(
        "--num_runs_eval",
        type=int,
        default=1,
        help="Number of runs to perform.",
    )
    parser.add_argument(
        "--num_episodes_train",
        type=int,
        default=100,
        help="Number of episodes to run.",
    )
    
    # Add any other arguments you need for your training and evaluation here
    return parser.parse_args()
def main():
    """
    Main function to run the training and evaluation of the model.
    """
    
    args = parse_args()
    
    # Set the configuration based on the provided config type
    config, config_dict = make_config(args.config_type)
    # Update the configuration with the parsed arguments
    config_dict.update({
        "baseline": args.baseline,
        "num_runs_eval": args.num_runs,
        "num_episodes_train": args.num_episodes,
    })
    
    #### Train the agent and test it consecutively
    env = FactoReal(ARGS=config)
    # Train the agent
    policy_kwargs = dict(
    net_arch=[64, 128, 96, 64]
    )
    config_dict.update({
        "policy_kwargs": policy_kwargs,
    })
    # Init wandb with training job_type
    init_wandb(config_dict, args.project_name, args.entity_name, job_type="train")
    # Train the agent here
    if args.baseline == "A2C":
        model = A2C("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
    elif args.baseline == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
    elif args.baseline == "DQN":
        model = DQN("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")
    
    model.learn(total_timesteps=args.num_episodes_train*config.episode_length, progress_bar=True)
    
    # Finish wandb training run
    wandb.finish()
    # Save the model
    model.save(os.path.join("models", f"{args.baseline}_{args.config_type}"))
    
    
    # Test the agent
    # Init wandb with testing job_type
    init_wandb(config_dict, args.project_name, args.entity_name, job_type="test")
    # Test the agent here
    obs = env.reset()
    done = False
    truncated = False
    for i in range(args.num_runs_eval):
        
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break