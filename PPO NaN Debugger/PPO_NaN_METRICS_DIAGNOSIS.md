# PPO NaN Metrics Issue - Diagnosis and Solutions

## Root Cause Identified

**Issue**: The new RLlib API (2.49.0) changed how episode metrics are stored and accessed. Episode metrics are no longer directly available in the top-level training result dictionary.

### What We Found:

1. **Old API Expected Location**: `result['episode_reward_mean']` → Returns `None`
2. **New API Actual Location**: `result['env_runners']['episode_return_mean']` → Contains actual values

3. **Episodes ARE being tracked correctly**, but metrics are nested differently:
   ```python
   # OLD WAY (returns None/NaN):
   episode_reward = result.get('episode_reward_mean')
   
   # NEW WAY (correct):
   episode_reward = result['env_runners']['episode_return_mean']
   episode_length = result['env_runners']['episode_len_mean']
   num_episodes = result['env_runners']['num_episodes']
   ```

###  Update Metric Access 
Update your training code to access metrics from the new location:

```python
def get_episode_metrics(result):
    """Extract episode metrics from RLlib 2.49.0 result."""
    env_runners = result.get('env_runners', {})
    
    return {
        'episode_reward_mean': env_runners.get('episode_return_mean'),
        'episode_reward_min': env_runners.get('episode_return_min'),
        'episode_reward_max': env_runners.get('episode_return_max'),
        'episode_len_mean': env_runners.get('episode_len_mean'),
        'episodes_this_iter': env_runners.get('num_episodes'),
        'num_env_steps_sampled': env_runners.get('num_env_steps_sampled')
    }

# Usage:
result = ppo.train()
metrics = get_episode_metrics(result)
print(f"Episode reward: {metrics['episode_reward_mean']}")
```
