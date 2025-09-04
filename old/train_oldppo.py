"""
PPO Training Script to reproduce NaN metrics issue.

This script sets up PPO training with the custom dictionary environment
and includes extensive debugging to track metrics and episode attributes.


THIS DOESNT WORK!
"""

import sys
import os


import ray
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np
import json
from test_environment import TemporalDictEnv


def train_ppo_with_debugging():
    """Train PPO with extensive debugging to identify NaN metrics issue."""
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    try:
        print("=== Setting up PPO Training ===")
        
        # Configure PPO with new API and custom model for dictionary observation spaces
        config = PPOConfig()
        config = config.environment(TemporalDictEnv)
        config = config.framework("torch")
        config = config.env_runners(num_env_runners=1, rollout_fragment_length=200)
        config = config.training(train_batch_size=1000, lr=3e-4)
        config = config.resources(num_gpus=0)
        
        # Add model configuration to handle dictionary observation spaces
        from ray.rllib.core.models.configs import MLPEncoderConfig
        config = config.rl_module(
            model_config_dict={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            }
        )
        
        print("PPO Config created successfully")
        print(f"Config: {config}")
        
        # Build the algorithm
        print("\n=== Building PPO Algorithm ===")
        ppo = config.build_algo()
        print("PPO Algorithm built successfully")
        
        # Test environment creation
        print("\n=== Testing Environment Creation ===")
        test_env = TemporalDictEnv()
        obs, info = test_env.reset()
        print(f"Test environment reset successful")
        print(f"Initial observation: {obs}")
        print(f"Initial info: {info}")
        
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
            else:
                print("  ✅ No NaN values detected in main metrics")
            
            # Additional debugging - check sampler results
            if 'sampler_results' in result:
                sampler_results = result['sampler_results']
                print(f"  Sampler Results Keys: {list(sampler_results.keys())}")
                
                # Check episode-related sampler metrics
                for key in ['episode_reward_mean', 'episode_reward_min', 'episode_reward_max', 'episode_len_mean']:
                    if key in sampler_results:
                        value = sampler_results[key]
                        print(f"  Sampler {key}: {value}")
                        if isinstance(value, float) and np.isnan(value):
                            print(f"    🚨 NaN in sampler {key}")
            
            # Check evaluation results if available
            if 'evaluation' in result:
                eval_results = result['evaluation']
                print(f"  Evaluation Results: {eval_results}")
            
            # Print subset of available keys for debugging (not all to avoid spam)
            important_keys = [k for k in result.keys() if 'episode' in k.lower() or 'reward' in k.lower() or 'len' in k.lower()]
            print(f"  Important result keys: {important_keys}")
            
            # Save detailed results
            with open(f'/Users/varnie/Projects/rllib_test/training_result_iter_{iteration + 1}.json', 'w') as f:
                # Convert numpy types to regular Python types for JSON serialization
                json_result = {}
                for key, value in result.items():
                    try:
                        if isinstance(value, np.ndarray):
                            json_result[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            json_result[key] = value.item()
                        else:
                            json_result[key] = value
                    except:
                        json_result[key] = str(value)
                
                json.dump(json_result, f, indent=2)
            
            print(f"  Detailed results saved to training_result_iter_{iteration + 1}.json")
        
        print("\n=== Training Complete ===")
        
        # Final evaluation
        print("\n=== Final Environment Test ===")
        final_env = TemporalDictEnv()
        obs, info = final_env.reset()
        
        for step in range(10):
            # Get action from trained policy
            action = ppo.compute_single_action(obs)
            print(f"Step {step + 1}: Action from policy: {action}")
            
            obs, reward, terminated, truncated, info = final_env.step(action)
            print(f"  Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            
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
    print("Starting PPO training to reproduce NaN metrics issue...")
    train_ppo_with_debugging()
    print("Training script completed.")
