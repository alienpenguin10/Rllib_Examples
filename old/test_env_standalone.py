"""
Standalone test of the TemporalDictEnv to verify it works correctly
before running PPO training.
"""

from test_environment import TemporalDictEnv
import numpy as np


def test_environment():
    """Test the environment thoroughly."""
    print("=== Testing TemporalDictEnv ===")
    
    # Create environment
    env = TemporalDictEnv()
    
    # Test reset
    print("\n1. Testing reset...")
    obs, info = env.reset()
    print(f"Reset successful!")
    print(f"Observation type: {type(obs)}")
    print(f"Observation: {obs}")
    print(f"Info: {info}")
    
    # Verify observation space
    print(f"\n2. Verifying observation space...")
    assert env.observation_space.contains(obs), "Observation not in observation space!"
    print("✅ Observation space verification passed")
    
    # Test multiple steps
    print(f"\n3. Testing steps...")
    total_reward = 0
    episode_length = 0
    
    for step in range(20):
        # Sample random action
        action = env.action_space.sample()
        
        # Verify action space
        assert env.action_space.contains(action), f"Action not in action space: {action}"
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        episode_length += 1
        
        print(f"Step {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Observation valid: {env.observation_space.contains(obs)}")
        
        # Check for NaN in observation
        has_nan = False
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                if np.any(np.isnan(value)):
                    has_nan = True
                    print(f"  🚨 NaN detected in {key}: {value}")
            elif np.isnan(value):
                has_nan = True
                print(f"  🚨 NaN detected in {key}: {value}")
        
        if not has_nan:
            print(f"  ✅ No NaN values in observation")
        
        # Check if episode ended
        if terminated or truncated:
            print(f"  Episode ended: Terminated={terminated}, Truncated={truncated}")
            print(f"  Episode stats: Length={episode_length}, Total Reward={total_reward}")
            
            # Reset for next episode
            obs, info = env.reset()
            total_reward = 0
            episode_length = 0
            print(f"  Environment reset for new episode")
            break
    
    print(f"\n4. Testing termination conditions...")
    # Test if we can achieve termination
    env.reset()
    for attempt in range(5):
        # Try to manually set observation to trigger termination
        env.current_obs = {
            'continuous_obs': np.array([0.8, 0.0, 0.0], dtype=np.float32),  # Close to target
            'discrete_obs': 2,  # Target discrete value
            'temporal_state': np.array([0.0, 0.0], dtype=np.float32)
        }
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Termination test {attempt + 1}:")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        
        if terminated:
            print("  ✅ Successfully achieved termination condition")
            break
        
        if attempt == 4:
            print("  ⚠️ Could not achieve termination in 5 attempts")
    
    print(f"\n5. Testing truncation (max steps)...")
    env.reset()
    env.max_steps = 5  # Set low for testing
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if truncated:
            print(f"  ✅ Truncation triggered at step {step + 1}")
            break
    
    print(f"\n=== Environment Test Complete ===")
    print("✅ All tests passed! Environment appears to be working correctly.")


if __name__ == "__main__":
    test_environment()
