"""
Custom Gym Environment to reproduce PPO NaN metrics issue.

This environment simulates the conditions described:
- Dictionary observation and action spaces (discrete + continuous)
- Temporal dependence between steps
- Reward based on specific observation values
- Truncation after 1000+ steps or NaN observations
- Termination only when target is found

Episode Lifecycle:
Episode Start
    ↓
TemporalDictEnv.reset()
    ↓
- Clears step_count and temporal_history
    ↓
TemporalDictEnv._generate_observation()
    ↓
- Creates initial observation with temporal influence
    ↓
Returns (observation, info)

Main Loop:
For each step until termination/truncation:
    ↓
TemporalDictEnv.step(action)
    ↓
- Validates action
- Stores temporal information
    ↓
TemporalDictEnv._update_observation(action)
    ↓
- Applies action effects
- Incorporates temporal dependencies
    ↓
TemporalDictEnv._calculate_reward(obs)
    ↓
- Computes reward based on targets
    ↓
TemporalDictEnv._check_termination(obs)
    ↓
- Checks if targets achieved
    ↓
Returns (observation, reward, terminated, truncated, info)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple


class TemporalDictEnv(gym.Env):
    """
    Environment with dictionary obs/action spaces and temporal dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        '''
        - Sets up observation_space (Dict with continuous, discrete, temporal)
        - Sets up action_space (Dict with continuous, discrete)
        - Initializes state variables (step_count, temporal_history, current_obs)
        '''
        super().__init__()

        # Environment configuration
        self.max_steps = 1000
        self.target_continuous_value = 0.8 # Target value for continuous obs
        self.target_discrete_value = 2 # Target value for discrete obs
        
        # Define observation space (dictionary with discrete and continuous)
        self.observation_space = spaces.Dict({
            'continuous_obs': spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
            'discrete_obs': spaces.Discrete(5),
            'temporal_state': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })
        # Example observation for each:
        # 'continuous_obs': np.array([0.0, 1.0, -1.0], dtype=np.float32)
        # 'discrete_obs': 2
        # 'temporal_state': np.array([0.5, -0.5], dtype=np.float32)
        '''
        Current position = continuous_obs
        Current gear = discrete_obs
        Speed and direction history = temporal_state
        Without knowing your previous speed/direction, you can't make good decisions about acceleration or steering.
        '''

        # Define action space (dictionary with discrete and continuous)
        self.action_space = spaces.Dict({
            'continuous_action': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'discrete_action': spaces.Discrete(3)
        })

        # State variables
        self.step_count = 0
        self.temporal_history = []
        self.current_obs = None

        print("Environment initialized successfully")
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.step_count = 0
        self.temporal_history = []

        # Initialize observation
        self.current_obs = self._generate_observation()

        info = {'step_count': self.step_count, 'temporal_history_length':len(self.temporal_history)}

        print(f"Environment reset - Step: {self.step_count}")
        print(f"Initial observation types: {type(self.current_obs)}")

        for key, value in self.current_obs.items():
            print(f"{key}: {type(value)} - {value}")
        
        return self.current_obs, info
    
    def step(self, action):
        """Execute one environment step"""
        self.step_count += 1

        # Validate action space
        if not self.action_space.contains(action):
            print(f"WARNING: Invalid action received: {action}")
            # Force truncation due to invalid action
            return self.current_obs, 0.0, False, True, {'error': 'invalid_action'}


        print(f"\nStep {self.step_count}")
        print(f"Action received: {action}")
        print(f"Action types: {type(action)}")
        for key, value in action.items():
            print(f"  {key}: {type(value)} - {value}")
        
        # Store temporal information
        self.temporal_history.append({
            'step': self.step_count,
            'action': action.copy(),
            'prev_obs': self.current_obs.copy()
        })

        # Update observation based on action and temporal dependencies
        self.current_obs = self._update_observation(action)

        # Calculate reward based on observation values
        reward = self._calculate_reward(self.current_obs)
        
        # Check termination conditions
        terminated = self._check_termination(self.current_obs)
        
        # Check truncation conditions
        truncated = self.step_count >= self.max_steps
        
        info = {
            'step_count': self.step_count,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'temporal_history_length': len(self.temporal_history),
            'target_continuous': self.target_continuous_value,
            'target_discrete': self.target_discrete_value
        }
        
        print(f"Observation: {self.current_obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        
        return self.current_obs, reward, terminated, truncated, info
    
    
    def _generate_observation(self):
        """Generate a new observation"""
        # Add some randomness with temporal influence
        temporal_influence = 0.0
        if len(self.temporal_history) > 0:
            # Use previous action to influence current observation
            last_action = self.temporal_history[-1]['action']
            temporal_influence = np.mean(last_action['continuous_action']) * 0.1
        
        obs = {
            'continuous_obs': np.random.uniform(-1.5, 1.5, 3).astype(np.float32) + temporal_influence,
            'discrete_obs': np.random.randint(0,5),
            'temporal_state': np.random.uniform(-0.5, 0.5, 2).astype(np.float32)
        }

        # Ensure observation is within bounds
        obs['continuous_obs'] = np.clip(obs['continuous_obs'], -2.0, 2.0)
        obs['temporal_state'] = np.clip(obs['temporal_state'], -1.0, 1.0)

        return obs
    
    def _update_observation(self, action):
        """Update observation based on action and temporal dependencies."""
        # Temporal influence from previous observations
        temporal_factor = 0.0
        if len(self.temporal_history) > 1:
            # Use history of actions to influence observations
            recent_actions = self.temporal_history[-2:]
            temporal_factor = np.mean([np.mean(h['action']['continuous_action']) for h in recent_actions]) * 0.05
        
        # Update based on action
        continuous_change_2d = action['continuous_action'] * 0.3 + temporal_factor
        # Extend to 3D for continuous_obs
        continuous_change_3d = np.array([continuous_change_2d[0], continuous_change_2d[1], 0.0])
        discrete_influence = action['discrete_action'] * 0.1
        
        new_obs = {
            'continuous_obs': (self.current_obs['continuous_obs'] + 
                             continuous_change_3d + 
                             np.random.normal(0, 0.1, 3)).astype(np.float32),
            'discrete_obs': (self.current_obs['discrete_obs'] + 
                           np.random.randint(-1, 2)) % 5,
            'temporal_state': (self.current_obs['temporal_state'] + 
                             continuous_change_2d * 0.5 + 
                             np.random.normal(0, 0.05, 2)).astype(np.float32)
        }
        
        # Ensure bounds
        new_obs['continuous_obs'] = np.clip(new_obs['continuous_obs'], -2.0, 2.0)
        new_obs['temporal_state'] = np.clip(new_obs['temporal_state'], -1.0, 1.0)

        # Occasionally introduce NaN (uncomment to test NaN handling)
        # if np.random.random() < 0.01:  # 1% chance
        #     new_obs['continuous_obs'][0] = np.nan
        
        return new_obs
    
    def _calculate_reward(self, obs):
        """Calculate reward based on observation values."""
        reward = 0.0
        
        # Reward for getting close to target continuous value
        continuous_distance = np.abs(obs['continuous_obs'][0] - self.target_continuous_value)
        if continuous_distance < 0.1:
            reward += 0.5
        elif continuous_distance < 0.3:
            reward += 0.2
        
        # Reward for hitting the target discrete value
        if obs['discrete_obs'] == self.target_discrete_value:
            reward += 0.3
        
        # Big reward for achieving both targets (termination condition)
        if (continuous_distance < 0.1 and 
            obs['discrete_obs'] == self.target_discrete_value):
            reward = 1.0
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.001
        
        return float(reward)
    
    def _check_termination(self, obs):
        """Check if episode should terminate (target achieved)."""
        continuous_distance = np.abs(obs['continuous_obs'][0] - self.target_continuous_value)
        target_achieved = (continuous_distance < 0.1 and obs['discrete_obs'] == self.target_discrete_value)
        return target_achieved
    
    def _check_for_nan(self, obs):
        """Check if observation contains NaN values."""
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                if np.any(np.isnan(value)):
                    return True
            elif np.isnan(value):
                return True
        return False


class FlattenedDictEnv(gym.Wrapper):
    """
    Wrapper that flattens dictionary observation and action spaces to work with RLlib.
    This preserves the same underlying problem while making it compatible with RLlib's new API.

    ### Initialization Chain   ###
    RLlib creates FlattenedDictEnv
        ↓
    FlattenedDictEnv.__init__()
        ↓
    - Creates TemporalDictEnv() as base_env
        ↓
    TemporalDictEnv.__init__()
        ↓
    - Sets up original observation/action spaces
    - Initializes state variables
        ↓
    FlattenedDictEnv sets up flattened spaces
        ↓
    Returns ready-to-use environment

    ### Step Flow ###
    RLlib calls FlattenedDictEnv.step(flattened_action)
        ↓
    FlattenedDictEnv calls _unflatten_action(flattened_action)
        ↓
    FlattenedDictEnv calls base_env.step(action_dict) (TemporalDictEnv.step())
        ↓
    TemporalDictEnv.step() calls:
        - _update_observation()
        - _calculate_reward()
        - _check_termination()
        ↓
    Returns (obs_dict, reward, terminated, truncated, info) to FlattenedDictEnv
        ↓
    FlattenedDictEnv calls _flatten_observation(obs_dict)
        ↓
    Returns (flattened_obs, reward, terminated, truncated, info) to RLlib

    ### Observation Flattening ###
    Original Dict Observation:
    {
        'continuous_obs': [0.1, -0.5, 0.8],     # 3 elements
        'discrete_obs': 2,                      # 1 element → one-hot
        'temporal_state': [0.3, -0.2]           # 2 elements
    }
        ↓
    Flattened Array Observation: 
    [0.1, -0.5, 0.8, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3, -0.2]
    # 3 continuous + 5 one-hot + 2 temporal = 10 elements

    ### Action Unflattening ###
    Flattened Array Action:
    [0.5, -0.3, 0.8]  # 3 elements
        ↓
    Dictionary Action:
    {
        'continuous_action': [0.5, -0.3],  # First 2 elements
        'discrete_action': 2               # Last element converted to discrete
    }

    ### Episode Start ###
    FlattenedDictEnv.reset() 
        → TemporalDictEnv.reset() 
        → TemporalDictEnv._generate_observation()
        → FlattenedDictEnv._flatten_observation()
        → Return flattened obs to RLlib

    ### Each Step ###
    FlattenedDictEnv.step(flattened_action)
        → FlattenedDictEnv._unflatten_action()
        → TemporalDictEnv.step(action_dict)
        → TemporalDictEnv._update_observation()
        → TemporalDictEnv._calculate_reward()
        → TemporalDictEnv._check_termination()
        → FlattenedDictEnv._flatten_observation()
        → Return flattened obs, reward, done to RLlib
    """
    
    def __init__(self, config=None):
        self.base_env = TemporalDictEnv()
        super().__init__(self.base_env)
        
        # Flatten observation space: continuous_obs(3) + discrete_obs(1) + temporal_state(2) = 6 elements
        # We'll convert discrete_obs to one-hot encoding (5 elements) for a total of 10 elements
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(10,), dtype=np.float32)

        # Flatten action space: continuous_action(2) + discrete_action(1) = 3 elements
        # We'll use continuous actions for everything and convert the discrete action
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        print("Flattened Environment initialized successfully")
        print(f"Original obs space: {self.base_env.observation_space}")
        print(f"Flattened obs space: {self.observation_space}")
        print(f"Original action space: {self.base_env.action_space}")
        print(f"Flattened action space: {self.action_space}")
    
    def _flatten_observation(self, obs_dict):
        """Convert dictionary observation to flattened array."""
        # continuous_obs: 3 elements
        continuous_obs = obs_dict['continuous_obs'].astype(np.float32)
        
        # discrete_obs: convert to one-hot (5 elements)
        discrete_onehot = np.zeros(5, dtype=np.float32)
        discrete_onehot[obs_dict['discrete_obs']] = 1.0

        # temporal_state: 2 elements
        temporal = obs_dict['temporal_state'].astype(np.float32)

        # concatenate all: 3 + 5 + 2 = 10 elements
        flattened = np.concatenate([continuous_obs, discrete_onehot, temporal])

        # ensure bounds
        flattened = np.clip(flattened, -2.0, 2.0)

        return flattened

    def _unflatten_action(self, action_array):
        """Convert flattened action to dictionary."""
        # continuous_action: first 2 elements
        continuous_action = action_array[:2].astype(np.float32)

        # discrete_action: convert last element to discrete choice
        discrete_action = int(np.clip(np.round((action_array[2] + 1) * 1.5), 0, 2))

        return {
            'continuous_action': continuous_action,
            'discrete_action': discrete_action
        }
    
    def reset(self, **kwargs):
        """Reset environment and return flattened observation."""
        obs_dict, info = self.base_env.reset(**kwargs)
        flattened_obs = self._flatten_observation(obs_dict)
        
        # Store original observation for debugging
        info['original_obs'] = obs_dict
        
        print(f"Reset: Original obs = {obs_dict}")
        print(f"Reset: Flattened obs = {flattened_obs}")
        
        return flattened_obs, info
    

    def step(self, action):
        """Take step with flattened action and return flattened observation."""
        # Convert flattened action to dictionary
        action_dict = self._unflatten_action(action)
        
        print(f"Step: Flattened action = {action}")
        print(f"Step: Dict action = {action_dict}")
        
        # Take step in base environment
        obs_dict, reward, terminated, truncated, info = self.base_env.step(action_dict)
        
        # Flatten observation
        flattened_obs = self._flatten_observation(obs_dict)
        
        # Store original data for debugging
        info['original_obs'] = obs_dict
        info['original_action'] = action_dict
        
        print(f"Step: Original obs = {obs_dict}")
        print(f"Step: Flattened obs = {flattened_obs}")
        print(f"Step: Reward = {reward}, Terminated = {terminated}, Truncated = {truncated}")
        
        return flattened_obs, reward, terminated, truncated, info


if __name__ == "__main__":
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
    
    print("\n=== END of Flattened Environment ===")