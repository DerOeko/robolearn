# RLWM Model configurations for Controllability GoNoGo Task
import numpy as np
from .rw_agent import RWConfig


def _default_action_probs_fn(agent, logits):
    diff = agent.beta * (logits[0] - logits[1])
    if diff > 700:
        p0 = 1.0
    elif diff < -700:
        p0 = 0.0
    else:
        p0 = 1.0 / (1.0 + np.exp(-diff))
    return np.array([p0, 1.0 - p0])


def _default_choose_action_fn(agent, probs, log_history=False):
    action = agent._rng.choice(agent.n_actions, p=probs)
    if log_history:
        agent.history['actions'].append(action)
    return action


def rlwm_logits_fn(agent, observation):
    """Compute logits as mixture of RL and WM components"""
    # Get Q-values from both systems
    q_rl = agent.q_values[observation].copy()
    q_wm = agent.q_wm[observation].copy()

    # Compute mixture based on current weight
    mixed_q = (1 - agent.w[observation]) * q_rl + agent.w[observation] * q_wm

    # Add instrumental weight and biases
    logits = mixed_q * agent.instrumental_weight
    logits[1] += agent.go_bias

    # Add pavlovian component if using it
    if hasattr(agent, 'v_values'):
        pav_bias = agent.pavlovian_weight * agent.v_values[observation]
        logits[1] += pav_bias
        logits[0] -= pav_bias

    return logits


def rlwm_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """Update both RL and WM systems, plus mixture weights"""

    # 1. Update RL component (slow learning)
    q_pe = reward - agent.q_values[observation, action]
    agent.q_values[observation, action] += agent.alpha * q_pe

    # 2. Update WM component (fast learning if within capacity)
    # Check if this stimulus is currently in WM
    if agent._is_in_wm(observation):
        # Perfect learning for WM
        agent.q_wm[observation, action] = reward

    # 3. Decay WM toward uniform for all states
    uniform_q = 0  # for 2 actions
    agent.q_wm += agent.wm_decay * (uniform_q - agent.q_wm)

    # 4. Update Pavlovian values if using them
    if hasattr(agent, 'v_values'):
        v_pe = reward - agent.v_values[observation]
        agent.v_values[observation] += agent.alpha * v_pe

    # 5. Update mixture weights using reliability
    agent._update_mixture_weights(observation, action, reward)

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(q_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['q_wm'].append(agent.q_wm.copy())
        agent.history['w'].append(agent.w.copy())
        agent.history['observations'].append(observation)

    return q_pe


def rlwm_reset_fn(agent, seed=None, reset_history: bool = False):
    """Reset agent state including WM components"""
    if seed is not None:
        agent._rng = np.random.default_rng(seed)

    # Reset Q-values
    agent.q_values = np.full(
        (agent.n_states, agent.n_actions), agent.config.initial_q_value, dtype=np.float64
    )

    # Reset WM Q-values to uniform
    agent.q_wm = np.full(
        (agent.n_states, agent.n_actions), 0.0, dtype=np.float64
    )

    # Reset mixture weights
    agent.w = np.full(agent.n_states, agent.w0, dtype=np.float64)

    # Reset WM tracking
    agent._wm_contents = set()
    agent._last_seen = {}

    if hasattr(agent, 'v_values'):
        agent.v_values = np.full(agent.n_states, agent.config.initial_v_value, dtype=np.float64)

    if reset_history:
        agent.history = {key: [] for key in agent.history.keys()}
        agent.history['q_wm'] = []
        agent.history['w'] = []


def controllability_rlwm_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """RLWM update with dynamic capacity based on inferred controllability"""

    # Track outcomes to infer controllability
    if not hasattr(agent, '_control_tracker'):
        agent._control_tracker = {'correct_rewards': 0, 'total_trials': 0}

    # Update controllability estimate
    optimal_action = agent._get_optimal_action(observation)
    if action == optimal_action:
        agent._control_tracker['correct_rewards'] += (reward > 0)
    agent._control_tracker['total_trials'] += 1

    # Adjust effective capacity based on inferred control
    if agent._control_tracker['total_trials'] > 10:
        control_estimate = agent._control_tracker['correct_rewards'] / agent._control_tracker['total_trials']

        # High control: use full capacity
        # Low control: reduce effective capacity (harder to maintain hypotheses)
        if control_estimate > 0.7:
            agent.effective_capacity = agent.wm_capacity
        else:
            agent.effective_capacity = max(2, agent.wm_capacity // 2)

    # Continue with normal RLWM update
    return rlwm_update_fn(agent, observation, action, reward, next_observation, terminated, log_history)


def build_rlwm_model(base_name, **components):
    """Build RLWM model with specified components"""

    config = {
        'included': True,
        # RL parameters
        'alpha': 0.3,
        'beta': 5.0,
        # WM parameters
        'wm_capacity': 4,  # Can hold 4 stimulus-action mappings
        'wm_decay': 0.1,   # Decay rate toward uniform
        'w0': 0.8,         # Initial mixture weight for WM
        # Additional parameters
        'initial_q_value': 0,
        'n_states': 4,
        'n_actions': 2,
        'reset_after_block': True,
        # Functions
        'logits_fn': rlwm_logits_fn,
        'action_probs_fn': _default_action_probs_fn,
        'choose_action_fn': _default_choose_action_fn,
        'update_fn': rlwm_update_fn,
        'reset_fn': rlwm_reset_fn,
        'fit_parameters': ["alpha", "beta", "wm_capacity", "wm_decay"],
        'parameter_bounds': {
            'alpha': (0.0, 1.0),
            'beta': (0.0, 10.0),
            'wm_capacity': (1, 8),
            'wm_decay': (0.0, 0.5)
        },
        'plausible_bounds': {
            'alpha': (0.1, 0.8),
            'beta': (1.0, 5.0),
            'wm_capacity': (2, 6),
            'wm_decay': (0.01, 0.3)
        }
    }

    # Add optional components
    if 'go_bias' in components:
        config['go_bias'] = components['go_bias']
        config['fit_parameters'].append('go_bias')
        config['parameter_bounds']['go_bias'] = (0.0, 1.0)
        config['plausible_bounds']['go_bias'] = (0.01, 0.5)

    if 'pavlovian' in components:
        config['pavlovian_weight'] = components['pavlovian']
        config['initial_v_value'] = 0.51
        config['fit_parameters'].append('pavlovian_weight')
        config['parameter_bounds']['pavlovian_weight'] = (0.0, 1.0)
        config['plausible_bounds']['pavlovian_weight'] = (0.01, 0.75)

    if 'controllability_sensitive' in components:
        config['controllability_sensitive'] = True

    config['name'] = f"{base_name}"

    return RWConfig(**config)


# RLWM model configurations
rlwm_models = {
    'rlwm_base': build_rlwm_model('rlwm_base'),
    'rlwm_bias': build_rlwm_model('rlwm_bias', go_bias=0.05),
    'rlwm_pav': build_rlwm_model('rlwm_pav', go_bias=0.05, pavlovian=0.5),
    'rlwm_ctrl': build_rlwm_model('rlwm_ctrl', go_bias=0.05, controllability_sensitive=True),
}