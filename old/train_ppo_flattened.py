"""
PPO Training Script with flattened observation space to avoid dictionary issues.

This script creates a wrapper that flattens the dictionary observation space
to work with the new RLlib API while preserving the underlying problem structure.
"""

import sys
import os

import ray
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np
import json
import gymnasium as gym
from gymnasium import spaces
from test_environment import TemporalDictEnv, FlattenedDictEnv


def train_ppo_with_flattened_env():
    """Train PPO with flattened environment to avoid dictionary space issues."""
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    try:
        print("=== Setting up PPO Training with Flattened Environment ===")
        
        # Configure PPO with flattened environment
        config = PPOConfig()
        config = config.environment(FlattenedDictEnv)
        config = config.framework("torch")
        config = config.env_runners(num_env_runners=1, rollout_fragment_length=200)
        config = config.training(train_batch_size=1000, lr=3e-4)
        config = config.resources(num_gpus=0)
        
        print("PPO Config created successfully")
        print(f"Config: {config}")
        
        # Build the algorithm
        print("\n=== Building PPO Algorithm ===")
        ppo = config.build_algo()
        print("PPO Algorithm built successfully")
        
        # Test environment creation
        print("\n=== Testing Flattened Environment ===")
        test_env = FlattenedDictEnv()
        obs, info = test_env.reset()
        print(f"Test environment reset successful")
        print(f"Flattened observation shape: {obs.shape}")
        print(f"Flattened observation: {obs}")
        print(f"Original observation from info: {info['original_obs']}")
        
        # Test a few steps
        for i in range(3):
            action = test_env.action_space.sample()
            obs, reward, terminated, truncated, info = test_env.step(action)
            print(f"Step {i+1}: reward={reward}, terminated={terminated}, truncated={truncated}")
            if terminated or truncated:
                obs, info = test_env.reset()
                print("Environment reset due to episode end")
        
        print("\n=== Starting PPO Training ===")
        
        # Train for a few iterations with detailed logging
        for iteration in range(3):
            print(f"\n--- Training Iteration {iteration + 1} ---")
            
            # Train one iteration
            result = ppo.train()
            
            # Extract key metrics
            episode_reward_mean = result.get('episode_reward_mean', 'N/A')
            episode_reward_min = result.get('episode_reward_min', 'N/A')
            episode_reward_max = result.get('episode_reward_max', 'N/A')
            episode_len_mean = result.get('episode_len_mean', 'N/A')
            episodes_this_iter = result.get('episodes_this_iter', 'N/A')
            num_env_steps_sampled = result.get('num_env_steps_sampled', 'N/A')
            
            print(f"Results for iteration {iteration + 1}:")
            print(f"  Episode Reward Mean: {episode_reward_mean}")
            print(f"  Episode Reward Min: {episode_reward_min}")
            print(f"  Episode Reward Max: {episode_reward_max}")
            print(f"  Episode Length Mean: {episode_len_mean}")
            print(f"  Episodes This Iter: {episodes_this_iter}")
            print(f"  Env Steps Sampled: {num_env_steps_sampled}")
            
            # Check for NaN values
            metrics_with_nan = []
            if isinstance(episode_reward_mean, float) and np.isnan(episode_reward_mean):
                metrics_with_nan.append('episode_reward_mean')
            if isinstance(episode_reward_min, float) and np.isnan(episode_reward_min):
                metrics_with_nan.append('episode_reward_min')
            if isinstance(episode_reward_max, float) and np.isnan(episode_reward_max):
                metrics_with_nan.append('episode_reward_max')
            if isinstance(episode_len_mean, float) and np.isnan(episode_len_mean):
                metrics_with_nan.append('episode_len_mean')
            
            if metrics_with_nan:
                print(f"  🚨 NaN DETECTED in metrics: {metrics_with_nan}")
                print("  This reproduces the issue described by your colleague!")
                
                # Let's also check if there are any episodes at all
                print(f"  Debugging: episodes_this_iter = {episodes_this_iter}")
                print(f"  Debugging: num_env_steps_sampled = {num_env_steps_sampled}")
                
            else:
                print("  ✅ No NaN values detected in main metrics")
            
            # Additional debugging
            if 'sampler_results' in result:
                sampler_results = result['sampler_results']
                print(f"  Sampler Results Keys: {list(sampler_results.keys())}")
            
            # Debug: Print more details about what's in result
            print(f"  Result keys: {list(result.keys())}")
            
            # Look for any episode-related info
            episode_keys = [k for k in result.keys() if 'episode' in k.lower()]
            print(f"  Episode-related keys: {episode_keys}")
            
            # Check env runner results specifically
            if 'env_runners' in result:
                env_runner_results = result['env_runners']
                print(f"  Env runner results type: {type(env_runner_results)}")
                print(f"  Env runner results: {env_runner_results}")
            
            # Print some actual metric values to see what's happening
            print(f"  Raw episode_reward_mean: {result.get('episode_reward_mean')} (type: {type(result.get('episode_reward_mean'))})")
            print(f"  Raw episode_len_mean: {result.get('episode_len_mean')} (type: {type(result.get('episode_len_mean'))})")
            
            # Save detailed results (with better serialization)
            json_result = {}
            for key, value in result.items():
                try:
                    if value is None:
                        json_result[key] = None
                    elif isinstance(value, np.ndarray):
                        json_result[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating, np.float32, np.float64)):
                        json_result[key] = float(value)
                    elif isinstance(value, dict):
                        # Recursively handle nested dicts
                        json_result[key] = {}
                        for k2, v2 in value.items():
                            try:
                                if isinstance(v2, (np.integer, np.floating, np.float32, np.float64)):
                                    json_result[key][k2] = float(v2)
                                elif isinstance(v2, np.ndarray):
                                    json_result[key][k2] = v2.tolist()
                                else:
                                    json_result[key][k2] = v2
                            except:
                                json_result[key][k2] = str(v2)
                    else:
                        json_result[key] = value
                except Exception as e:
                    json_result[key] = f"Error serializing: {str(e)}"
            
            try:
                with open(f'/Users/varnie/Projects/rllib_test/flattened_training_result_iter_{iteration + 1}.json', 'w') as f:
                    json.dump(json_result, f, indent=2)
            except Exception as e:
                print(f"  Warning: Could not save JSON results: {e}")
            
            print(f"  Detailed results saved to flattened_training_result_iter_{iteration + 1}.json")
        
        print("\n=== Training Complete ===")
        
        # Final evaluation with policy
        print("\n=== Final Policy Test ===")
        final_env = FlattenedDictEnv()
        obs, info = final_env.reset()
        
        for step in range(10):
            # Get action from trained policy
            action = ppo.compute_single_action(obs)
            print(f"Step {step + 1}: Action from policy: {action}")
            
            obs, reward, terminated, truncated, info = final_env.step(action)
            print(f"  Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            print(f"  Original obs: {info['original_obs']}")
            
            if terminated or truncated:
                print("  Episode ended, resetting...")
                obs, info = final_env.reset()
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'ppo' in locals():
            ppo.stop()
        ray.shutdown()


if __name__ == "__main__":
    print("Starting PPO training with flattened environment to reproduce/fix NaN metrics issue...")
    train_ppo_with_flattened_env()
    print("Training script completed.")
