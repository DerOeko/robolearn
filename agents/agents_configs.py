# %% Standard Rescorla Wagner model configurations
import numpy as np
from .rw_agent import RWConfig
from .rlwm_configs import rlwm_models


def default_logits_fn(agent, observation):
    return agent.q_values[observation].copy()


def go_bias_logits_fn(agent, observation):
    logits = agent.q_values[observation].copy()
    logits[1] += agent.go_bias
    return logits


def pav_logits_fn(agent, observation):
    logits = agent.q_values[observation] * agent.instrumental_weight
    logits[1] += agent.go_bias

    # pavlovian: positive v_value biases toward go, negative toward nogo
    pav_bias = agent.pavlovian_weight * agent.v_values[observation]
    logits[1] += pav_bias  # add to go action
    logits[0] -= pav_bias  # subtract from nogo action

    return logits


def reward_rate_go_bias_logits_fn(agent, observation):
    """modulate go bias by reward rate"""
    logits = agent.q_values[observation].copy()
    
    # modulate go bias by reward rate
    if hasattr(agent, 'reward_rate'):
        effective_go_bias = agent.go_bias * (1 + agent.reward_rate_mod * (agent.reward_rate - 0.5))
    else:
        effective_go_bias = agent.go_bias
    
    logits[1] += effective_go_bias
    return logits


def bayesian_control_logits_fn(agent, observation):
    """combine q-learning and model-based bayesian control"""
    
    # model-free component (q-values)
    mf_logits = agent.q_values[observation].copy()
    mf_logits[1] += agent.go_bias
    
    # model-based component using bayesian probabilities
    if hasattr(agent, 'msas'):
        omega_state = 1 / (1 + np.exp(-agent.omega_bias - agent.lstate[observation]))
        
        # expected values for each action based on bayesian probabilities
        mb_go = agent.msas[observation, 1] * 1 + (1 - agent.msas[observation, 1]) * (-1)
        mb_nogo = agent.msas[observation, 0] * 1 + (1 - agent.msas[observation, 0]) * (-1)
        mb_logits = np.array([mb_nogo, mb_go])
        
        # weighted combination
        logits = (1 - omega_state) * mf_logits + omega_state * mb_logits
    else:
        logits = mf_logits
    
    return logits


def bayesian_pav_logits_fn(agent, observation):
    """bayesian control + pavlovian bias"""
    
    # model-free component with pavlovian bias
    mf_logits = agent.q_values[observation] * agent.instrumental_weight
    mf_logits[1] += agent.go_bias
    
    # pavlovian component
    pav_bias = agent.pavlovian_weight * agent.v_values[observation]
    mf_logits[1] += pav_bias
    mf_logits[0] -= pav_bias
    
    # model-based component
    if hasattr(agent, 'msas'):
        omega_state = 1 / (1 + np.exp(-agent.omega_bias - agent.lstate[observation]))
        
        mb_go = agent.msas[observation, 1] * 1 + (1 - agent.msas[observation, 1]) * (-1)
        mb_nogo = agent.msas[observation, 0] * 1 + (1 - agent.msas[observation, 0]) * (-1)
        mb_logits = np.array([mb_nogo, mb_go])
        
        logits = (1 - omega_state) * mf_logits + omega_state * mb_logits
    else:
        logits = mf_logits
    
    return logits


def omega_control_logits_fn(agent, observation):
    """simple omega control of Q-instrumental influence"""
    
    # q-values with reduced influence based on omega
    if hasattr(agent, 'omega'):
        q_influence = 1.0 / (1.0 + np.exp(agent.beta_control * agent.omega))
        logits = agent.q_values[observation] * q_influence
    else:
        logits = agent.q_values[observation].copy()
    
    logits[1] += agent.go_bias
    
    # pavlovian component (v-values with separate temperature)
    pav_logits = agent.v_values[observation]
    logits[1] += pav_logits
    logits[0] -= pav_logits
    
    return logits

def default_action_probs_fn(agent, logits):
    diff = agent.beta * (logits[0] - logits[1])
    if diff > 700:
        p0 = 1.0
    elif diff < -700:
        p0 = 0.0
    else:
        p0 = 1.0 / (1.0 + np.exp(-diff))
    return np.array([p0, 1.0 - p0])


def epsilon_greedy_action_probs_fn(agent, logits):
    n_actions = len(logits)
    logits = logits + agent._rng.normal(0, 1e-8, size=logits.shape)
    greedy_action = np.argmax(logits)

    # start with epsilon/n_actions for all actions
    probs = np.full(n_actions, agent.epsilon / n_actions)
    # add the remaining (1-epsilon) to greedy action
    probs[greedy_action] += 1.0 - agent.epsilon

    return probs


def noisy_action_probs_fn(agent, logits):
    # get base sigmoid probs
    diff = agent.beta * (logits[0] - logits[1])
    if diff > 700:
        p0 = 1.0
    elif diff < -700:
        p0 = 0.0
    else:
        p0 = 1.0 / (1.0 + np.exp(-diff))

    base_probs = np.array([p0, 1.0 - p0])

    # add decision noise
    noise_level = agent.decision_noise  # this is ξ
    noisy_probs = base_probs * (1 - noise_level) + noise_level / 2

    return noisy_probs


def default_choose_action_fn(agent, probs, log_history=False):
    action = agent._rng.choice(agent.n_actions, p=probs)
    if log_history:
        agent.history['actions'].append(action)
    return action


def default_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """Update Q-values and return prediction error"""

    if not (0 <= observation < agent.n_states):
        raise ValueError(f"Observation {observation} out of bounds.")
    if not (0 <= action < agent.n_actions):
        raise ValueError(f"Action {action} out of bounds.")

    prediction_error = reward - agent.q_values[observation, action]
    agent.q_values[observation, action] += agent.alpha * prediction_error

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['prediction_errors'].append(prediction_error)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return prediction_error


def pav_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """Update Q-values and return prediction error"""

    if not (0 <= observation < agent.n_states):
        raise ValueError(f"Observation {observation} out of bounds.")
    if not (0 <= action < agent.n_actions):
        raise ValueError(f"Action {action} out of bounds.")

    Q_prediction_error = reward - agent.q_values[observation, action]
    V_prediction_error = reward - agent.v_values[observation]
    agent.q_values[observation, action] += agent.alpha * Q_prediction_error
    agent.v_values[observation] += agent.alpha * V_prediction_error

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(Q_prediction_error)
        agent.history['v_prediction_errors'].append(V_prediction_error)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return Q_prediction_error


def asym_q_update_pe_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """q update with asymmetrical learning rates based on prediction error sign"""

    if not (0 <= observation < agent.n_states):
        raise ValueError(f"observation {observation} out of bounds.")
    if not (0 <= action < agent.n_actions):
        raise ValueError(f"action {action} out of bounds.")

    q_prediction_error = reward - agent.q_values[observation, action]

    # asymmetrical learning rates
    alpha = agent.alpha_pos if q_prediction_error >= 0 else agent.alpha_neg
    agent.q_values[observation, action] += alpha * q_prediction_error

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['prediction_errors'].append(q_prediction_error)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)

    return q_prediction_error

def asym_q_update_reward_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """q update with asymmetrical learning rates based on reward sign"""

    if not (0 <= observation < agent.n_states):
        raise ValueError(f"observation {observation} out of bounds.")
    if not (0 <= action < agent.n_actions):
        raise ValueError(f"action {action} out of bounds.")

    q_prediction_error = reward - agent.q_values[observation, action]

    # asymmetrical learning rates
    alpha = agent.alpha_pos if reward >= 0 else agent.alpha_neg
    agent.q_values[observation, action] += alpha * q_prediction_error

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['prediction_errors'].append(q_prediction_error)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)

    return q_prediction_error


def asym_pav_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """pavlovian + asymmetrical q learning rates"""

    if not (0 <= observation < agent.n_states):
        raise ValueError(f"observation {observation} out of bounds.")
    if not (0 <= action < agent.n_actions):
        raise ValueError(f"action {action} out of bounds.")

    q_prediction_error = reward - agent.q_values[observation, action]
    v_prediction_error = reward - agent.v_values[observation]

    # asymmetrical q learning, regular v learning
    alpha_q = agent.alpha_pos if q_prediction_error >= 0 else agent.alpha_neg
    agent.q_values[observation, action] += alpha_q * q_prediction_error
    agent.v_values[observation] += agent.alpha * v_prediction_error

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(q_prediction_error)
        agent.history['v_prediction_errors'].append(v_prediction_error)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)

    return q_prediction_error


def decay_q_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """Update Q-values and return prediction error"""

    if not (0 <= observation < agent.n_states):
        raise ValueError(f"Observation {observation} out of bounds.")
    if not (0 <= action < agent.n_actions):
        raise ValueError(f"Action {action} out of bounds.")

    Q_prediction_error = reward - agent.q_values[observation, action]
    V_prediction_error = reward - agent.v_values[observation]

    agent.q_values[observation, action] += agent.alpha * Q_prediction_error
    agent.v_values[observation] += agent.alpha * V_prediction_error

    agent.q_values = (1-agent.decay_rate) * agent.q_values + \
        agent.decay_rate * agent._init_q

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(Q_prediction_error)
        agent.history['v_prediction_errors'].append(V_prediction_error)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return Q_prediction_error

def collins_decay_q_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """Update Q-values and return prediction error"""

    if not (0 <= observation < agent.n_states):
        raise ValueError(f"Observation {observation} out of bounds.")
    if not (0 <= action < agent.n_actions):
        raise ValueError(f"Action {action} out of bounds.")

    Q_prediction_error = reward - agent.q_values[observation, action]
    V_prediction_error = reward - agent.v_values[observation]

    agent.q_values[observation, action] += agent.alpha * Q_prediction_error
    agent.v_values[observation] += agent.alpha * V_prediction_error

    agent.q_values += agent.decay_rate * (agent._init_q - agent.q_values)

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(Q_prediction_error)
        agent.history['v_prediction_errors'].append(V_prediction_error)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return Q_prediction_error

def collins_decay_both_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """Update Q-values and return prediction error"""

    if not (0 <= observation < agent.n_states):
        raise ValueError(f"Observation {observation} out of bounds.")
    if not (0 <= action < agent.n_actions):
        raise ValueError(f"Action {action} out of bounds.")

    Q_prediction_error = reward - agent.q_values[observation, action]
    V_prediction_error = reward - agent.v_values[observation]

    agent.q_values[observation, action] += agent.alpha * Q_prediction_error
    agent.v_values[observation] += agent.alpha * V_prediction_error

    agent.q_values += agent.decay_rate * (agent._init_q - agent.q_values)
    agent.v_values += agent.decay_rate * (agent._init_v - agent.v_values)

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(Q_prediction_error)
        agent.history['v_prediction_errors'].append(V_prediction_error)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return Q_prediction_error

def decay_v_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """Update Q-values and return prediction error"""

    if not (0 <= observation < agent.n_states):
        raise ValueError(f"Observation {observation} out of bounds.")
    if not (0 <= action < agent.n_actions):
        raise ValueError(f"Action {action} out of bounds.")

    Q_prediction_error = reward - agent.q_values[observation, action]
    V_prediction_error = reward - agent.v_values[observation]
    agent.q_values[observation, action] += agent.alpha * Q_prediction_error
    agent.v_values[observation] += agent.alpha * V_prediction_error

    agent.v_values = (1-agent.decay_rate) * agent.v_values + \
        agent.decay_rate * agent._init_v
    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(Q_prediction_error)
        agent.history['v_prediction_errors'].append(V_prediction_error)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return Q_prediction_error


def decay_both_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """Update Q-values and return prediction error"""

    if not (0 <= observation < agent.n_states):
        raise ValueError(f"Observation {observation} out of bounds.")
    if not (0 <= action < agent.n_actions):
        raise ValueError(f"Action {action} out of bounds.")

    Q_prediction_error = reward - agent.q_values[observation, action]
    V_prediction_error = reward - agent.v_values[observation]
    agent.q_values[observation, action] += agent.alpha * Q_prediction_error
    agent.v_values[observation] += agent.alpha * V_prediction_error

    agent.q_values = (1-agent.decay_rate) * agent.q_values + \
        agent.decay_rate * agent._init_q
    agent.v_values = (1-agent.decay_rate) * agent.v_values + \
        agent.decay_rate * agent._init_v

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(Q_prediction_error)
        agent.history['v_prediction_errors'].append(V_prediction_error)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return Q_prediction_error

def choice_kernel_decay_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """update with choice kernel decay - frequent choices persist, rare ones fade"""

    # normal q and v updates
    q_pe = reward - agent.q_values[observation, action]
    v_pe = reward - agent.v_values[observation]
    agent.q_values[observation, action] += agent.alpha * q_pe
    agent.v_values[observation] += agent.alpha * v_pe

    # track choice frequency
    if not hasattr(agent, 'visit_counts'):
        agent.visit_counts = np.zeros((agent.n_states, agent.n_actions))

    agent.visit_counts[observation, action] += 1

    # choice kernel decay
    for state in range(agent.n_states):
        total_visits = agent.visit_counts[state].sum()
        if total_visits > 15:  # need some history first
            choice_probs = agent.visit_counts[state] / total_visits
            decay_weights = 1 - choice_probs  # rare choices decay more
            agent.q_values[state] *= (1 - agent.decay_rate * decay_weights)

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(q_pe)
        agent.history['v_prediction_errors'].append(v_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)

    return q_pe


def asym_decay_q_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """asymmetrical learning + q decay"""

    q_pe = reward - agent.q_values[observation, action]
    alpha_q = agent.alpha_pos if q_pe >= 0 else agent.alpha_neg
    agent.q_values[observation, action] += alpha_q * q_pe

    # decay q toward baseline
    agent.q_values = (1-agent.decay_rate) * agent.q_values + \
        agent.decay_rate * agent._init_q

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['prediction_errors'].append(q_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return q_pe


def asym_pav_decay_both_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """asymmetrical q learning + pavlovian + decay both"""

    q_pe = reward - agent.q_values[observation, action]
    v_pe = reward - agent.v_values[observation]

    # asymmetrical q, regular v
    alpha_q = agent.alpha_pos if q_pe >= 0 else agent.alpha_neg
    agent.q_values[observation, action] += alpha_q * q_pe
    agent.v_values[observation] += agent.alpha * v_pe

    # decay both
    agent.q_values = (1-agent.decay_rate) * agent.q_values + \
        agent.decay_rate * agent._init_q
    agent.v_values = (1-agent.decay_rate) * agent.v_values + \
        agent.decay_rate * agent._init_v

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(q_pe)
        agent.history['v_prediction_errors'].append(v_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return q_pe


def asym_choice_kernel_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """asymmetrical learning + choice kernel decay"""

    q_pe = reward - agent.q_values[observation, action]
    alpha_q = agent.alpha_pos if q_pe >= 0 else agent.alpha_neg
    agent.q_values[observation, action] += alpha_q * q_pe

    # track visits
    if not hasattr(agent, 'visit_counts'):
        agent.visit_counts = np.zeros((agent.n_states, agent.n_actions))
    agent.visit_counts[observation, action] += 1

    # choice kernel decay
    for state in range(agent.n_states):
        total_visits = agent.visit_counts[state].sum()
        if total_visits > 15:
            choice_probs = agent.visit_counts[state] / total_visits
            decay_weights = 1 - choice_probs
            agent.q_values[state] *= (1 - agent.decay_rate * decay_weights)

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['prediction_errors'].append(q_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return q_pe


def asym_pav_choice_kernel_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """asymmetrical q + pavlovian + choice kernel decay"""

    q_pe = reward - agent.q_values[observation, action]
    v_pe = reward - agent.v_values[observation]

    alpha_q = agent.alpha_pos if q_pe >= 0 else agent.alpha_neg
    agent.q_values[observation, action] += alpha_q * q_pe
    agent.v_values[observation] += agent.alpha * v_pe

    # choice kernel on q only
    if not hasattr(agent, 'visit_counts'):
        agent.visit_counts = np.zeros((agent.n_states, agent.n_actions))
    agent.visit_counts[observation, action] += 1

    for state in range(agent.n_states):
        total_visits = agent.visit_counts[state].sum()
        if total_visits > 15:
            choice_probs = agent.visit_counts[state] / total_visits
            decay_weights = 1 - choice_probs
            agent.q_values[state] *= (1 - agent.decay_rate * decay_weights)

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(q_pe)
        agent.history['v_prediction_errors'].append(v_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return q_pe


def reward_rate_go_bias_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """modulate go bias based on running reward rate"""
    
    # track running reward rate
    if not hasattr(agent, 'reward_history'):
        agent.reward_history = []
        agent.reward_rate = 0.5
    
    agent.reward_history.append(1 if reward > 0 else 0)
    if len(agent.reward_history) > 20:  # sliding window
        agent.reward_history.pop(0)
    agent.reward_rate = np.mean(agent.reward_history)
    
    # normal q updates
    q_pe = reward - agent.q_values[observation, action]
    agent.q_values[observation, action] += agent.alpha * q_pe

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['prediction_errors'].append(q_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return q_pe


def context_decay_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """decay is stronger when outcomes are unpredictable"""
    
    # normal updates
    q_pe = reward - agent.q_values[observation, action]
    agent.q_values[observation, action] += agent.alpha * q_pe
    
    # track outcome variance
    if not hasattr(agent, 'outcome_variance'):
        agent.outcome_variance = 0.5
    
    # track variance in outcomes for this state-action
    prediction_error = abs(reward - agent.q_values[observation, action])
    agent.outcome_variance = 0.9 * agent.outcome_variance + 0.1 * prediction_error
    
    # high variance = high decay (uncertain environment)
    effective_decay = agent.decay_rate * (1 + agent.variance_mod * agent.outcome_variance)
    agent.q_values = (1 - effective_decay) * agent.q_values + effective_decay * agent._init_q

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['prediction_errors'].append(q_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return q_pe


def win_stay_boost_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """boost learning for go actions after positive outcomes"""
    
    q_pe = reward - agent.q_values[observation, action]
    v_pe = reward - agent.v_values[observation]
    
    # asymmetric learning
    alpha_q = agent.alpha_pos if q_pe >= 0 else agent.alpha_neg
    
    # boost learning for go actions after positive outcomes
    if action == 1 and reward > 0:  # go action + win
        alpha_q *= (1 + agent.win_go_boost)
    
    agent.q_values[observation, action] += alpha_q * q_pe
    agent.v_values[observation] += agent.alpha * v_pe

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(q_pe)
        agent.history['v_prediction_errors'].append(v_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
    return q_pe


def bayesian_control_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """bayesian controllability inference - reward already coded as 1/-1"""
    
    # initialize bayesian components if needed
    if not hasattr(agent, 'msas'):
        agent.msas = np.full((agent.n_states, agent.n_actions), 0.5)  # p(favorable | state, action)
        agent.mss = np.full(agent.n_states, 0.5)  # p(favorable | state)
        agent.msas_counts = np.full((agent.n_states, agent.n_actions), 1.0)
        agent.mss_counts = np.full(agent.n_states, 1.0)
        agent.lglob = 0.0  # global controllability log-likelihood
        agent.lstate = np.zeros(agent.n_states)  # state-specific controllability
    
    favorable = 1 if reward > 0 else 0
    
    # standard q-learning update
    q_pe = reward - agent.q_values[observation, action]
    agent.q_values[observation, action] += agent.alpha * q_pe
    
    # update controllability likelihood
    if favorable == 1:
        agent.lglob += np.log(agent.msas[observation, action]) - np.log(agent.mss[observation])
        agent.lstate[observation] += np.log(agent.msas[observation, action]) - np.log(agent.mss[observation])
    else:
        agent.lglob += np.log(1 - agent.msas[observation, action]) - np.log(1 - agent.mss[observation])
        agent.lstate[observation] += np.log(1 - agent.msas[observation, action]) - np.log(1 - agent.mss[observation])
    
    # update bayesian transition probabilities
    agent.mss_counts[observation] += 1
    agent.msas_counts[observation, action] += 1
    agent.mss[observation] += (favorable - agent.mss[observation]) / agent.mss_counts[observation]
    agent.msas[observation, action] += (favorable - agent.msas[observation, action]) / agent.msas_counts[observation, action]
    
    # optional counterfactual learning
    if hasattr(agent, 'use_counterfactual') and agent.use_counterfactual:
        omega_glob = 1 / (1 + np.exp(-agent.omega_bias - agent.lglob))
        counterfactual_action = 1 - action  # 0->1, 1->0
        agent.msas[observation, counterfactual_action] = (
            omega_glob * (1 - agent.msas[observation, action]) + 
            (1 - omega_glob) * agent.msas[observation, counterfactual_action]
        )
        agent.msas_counts[observation, counterfactual_action] += omega_glob

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['prediction_errors'].append(q_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
        # track omega values for bayesian control models
        if hasattr(agent, 'lglob'):
            omega_glob = 1 / (1 + np.exp(-agent.omega_bias - agent.lglob))
            agent.history.setdefault('omega_glob', []).append(omega_glob)
        if hasattr(agent, 'lstate'):
            omega_state = 1 / (1 + np.exp(-agent.omega_bias - agent.lstate[observation]))
            agent.history.setdefault('omega_state', []).append(omega_state)
    return q_pe


def bayesian_pav_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """bayesian controllability + pavlovian bias updating"""
    
    # initialize components if needed
    if not hasattr(agent, 'msas'):
        agent.msas = np.full((agent.n_states, agent.n_actions), 0.5)
        agent.mss = np.full(agent.n_states, 0.5)
        agent.msas_counts = np.full((agent.n_states, agent.n_actions), 1.0)
        agent.mss_counts = np.full(agent.n_states, 1.0)
        agent.lglob = 0.0
        agent.lstate = np.zeros(agent.n_states)
    
    favorable = 1 if reward > 0 else 0
    
    # q and v updates
    q_pe = reward - agent.q_values[observation, action]
    v_pe = reward - agent.v_values[observation]
    agent.q_values[observation, action] += agent.alpha * q_pe
    agent.v_values[observation] += agent.alpha * v_pe
    
    # bayesian updates (same as above)
    if favorable == 1:
        agent.lglob += np.log(agent.msas[observation, action]) - np.log(agent.mss[observation])
        agent.lstate[observation] += np.log(agent.msas[observation, action]) - np.log(agent.mss[observation])
    else:
        agent.lglob += np.log(1 - agent.msas[observation, action]) - np.log(1 - agent.mss[observation])
        agent.lstate[observation] += np.log(1 - agent.msas[observation, action]) - np.log(1 - agent.mss[observation])
    
    agent.mss_counts[observation] += 1
    agent.msas_counts[observation, action] += 1
    agent.mss[observation] += (favorable - agent.mss[observation]) / agent.mss_counts[observation]
    agent.msas[observation, action] += (favorable - agent.msas[observation, action]) / agent.msas_counts[observation, action]
    
    if hasattr(agent, 'use_counterfactual') and agent.use_counterfactual:
        omega_glob = 1 / (1 + np.exp(-agent.omega_bias - agent.lglob))
        counterfactual_action = 1 - action
        agent.msas[observation, counterfactual_action] = (
            omega_glob * (1 - agent.msas[observation, action]) + 
            (1 - omega_glob) * agent.msas[observation, counterfactual_action]
        )
        agent.msas_counts[observation, counterfactual_action] += omega_glob

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(q_pe)
        agent.history['v_prediction_errors'].append(v_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
        # track omega values for bayesian control models
        if hasattr(agent, 'lglob'):
            omega_glob = 1 / (1 + np.exp(-agent.omega_bias - agent.lglob))
            agent.history.setdefault('omega_glob', []).append(omega_glob)
        if hasattr(agent, 'lstate'):
            omega_state = 1 / (1 + np.exp(-agent.omega_bias - agent.lstate[observation]))
            agent.history.setdefault('omega_state', []).append(omega_state)
    return q_pe


def omega_control_update_fn(agent, observation, action, reward, next_observation, terminated, log_history=True):
    """simple omega control that modulates Q-instrumental influence"""
    
    # initialize omega if needed
    if not hasattr(agent, 'omega'):
        agent.omega = getattr(agent.config, 'omega_init', 0.0)
    
    # q and v updates
    q_pe = reward - agent.q_values[observation, action]
    v_pe = reward - agent.v_values[observation]
    
    agent.q_values[observation, action] += agent.alpha * q_pe
    agent.v_values[observation] += agent.alpha * v_pe
    
    # update omega based on difference between v_pe and q_pe magnitudes
    omega_pe = abs(v_pe) - abs(q_pe) - agent.omega
    agent.omega += agent.alpha * omega_pe

    if log_history:
        agent.history['rewards'].append(reward)
        agent.history['q_prediction_errors'].append(q_pe)
        agent.history['v_prediction_errors'].append(v_pe)
        agent.history['q_values'].append(agent.q_values.copy())
        agent.history['observations'].append(observation)
        # track simple omega value for omega control models
        if hasattr(agent, 'omega'):
            agent.history.setdefault('omega', []).append(agent.omega)
    return q_pe


def default_reset_fn(agent, seed=None, reset_history: bool = False) -> None:
    """Reset agent state"""
    if seed is not None:
        agent._rng = np.random.default_rng(seed)

    if agent.config.initial_q_values is not None:
        agent.q_values = agent.config.initial_q_values.copy()
    else:
        agent.q_values = np.full(
            (agent.n_states, agent.n_actions), agent.config.initial_q_value, dtype=np.float64
        )

    if agent.config.initial_v_values is not None:
        agent.v_values = agent.config.initial_v_values.copy()
    else:
        agent.v_values = np.array([agent.config.initial_v_value, -agent.config.initial_v_value, agent.config.initial_v_value, -agent.config.initial_v_value], dtype=np.float64)

    # reset additional attributes for new update functions
    if hasattr(agent, 'reward_history'):
        agent.reward_history = []
        agent.reward_rate = 0.5
    
    if hasattr(agent, 'outcome_variance'):
        agent.outcome_variance = 0.5
    
    if hasattr(agent, 'visit_counts'):
        agent.visit_counts = np.zeros((agent.n_states, agent.n_actions))
    
    if hasattr(agent, 'msas'):
        agent.msas = np.full((agent.n_states, agent.n_actions), 0.5)
        agent.mss = np.full(agent.n_states, 0.5)
        agent.msas_counts = np.full((agent.n_states, agent.n_actions), 1.0)
        agent.mss_counts = np.full(agent.n_states, 1.0)
        agent.lglob = 0.0
        agent.lstate = np.zeros(agent.n_states)
    
    if hasattr(agent, 'omega'):
        agent.omega = getattr(agent.config, 'omega_init', 0.0)

    if reset_history:
        agent.history = {key: [] for key in agent.history.keys()}


def build_model(base_name, action_type="sigmoid", **components):
    """build models by stacking components"""

    # base defaults
    config = {
        'included': True,
        'alpha': 0.3, 'beta': 5.0,
        'initial_q_value': 0,
        'n_states': 4, 'n_actions': 2,
        'reset_after_block': True,
        'logits_fn': default_logits_fn,
        'action_probs_fn': default_action_probs_fn if action_type == "sigmoid" else epsilon_greedy_action_probs_fn,
        'choose_action_fn': default_choose_action_fn,
        'update_fn': default_update_fn,
        'reset_fn': default_reset_fn,
        'fit_parameters': ["alpha", "beta"],
        'parameter_bounds': {'alpha': (0.0, 1.0), 'beta': (0.0, 10.0)},
        'plausible_bounds': {'alpha':(0.1, 0.8), 'beta': (1.0, 5.0)},
    }

    # add components
    if 'decision_noise' in components:
        config['decision_noise'] = components['decision_noise']
        config['action_probs_fn'] = noisy_action_probs_fn
        config['fit_parameters'].append('decision_noise')
        config['parameter_bounds']['decision_noise'] = (0.0, 1.0)
        config['plausible_bounds']['decision_noise'] = (0.05, 0.5)


    if 'go_bias' in components:
        config['go_bias'] = components['go_bias']
        config['logits_fn'] = go_bias_logits_fn
        config['fit_parameters'].append('go_bias')
        config['parameter_bounds']['go_bias'] = (0.0, 1.0)
        config['plausible_bounds']['go_bias'] = (0.01, 0.5)

    if 'pavlovian' in components:
        config['pavlovian_weight'] = components['pavlovian']
        config['initial_v_value'] = 0.51
        config['logits_fn'] = pav_logits_fn
        config['fit_parameters'].append('pavlovian_weight')
        config['parameter_bounds']['pavlovian_weight'] = (0.0, 1.0)
        config['plausible_bounds']['pavlovian_weight'] = (0.01, 0.75)

    if 'dynamic_pav' in components and components['dynamic_pav']:
        config['update_fn'] = pav_update_fn

    if 'decay' in components and components['decay'] == 'q':
        config['decay_rate'] = 0.1
        config['update_fn'] = decay_q_update_fn
        config['fit_parameters'].append('decay_rate')
        config['parameter_bounds']['decay_rate'] = (0.0, 1.0)
        config['plausible_bounds']['decay_rate'] = (0.01, 0.75)
    elif 'decay' in components and components['decay'] == 'v':
        config['decay_rate'] = 0.1
        config['update_fn'] = decay_v_update_fn
        config['fit_parameters'].append('decay_rate')
        config['parameter_bounds']['decay_rate'] = (0.0, 1.0)
        config['plausible_bounds']['decay_rate'] = (0.01, 0.75)
    elif 'decay' in components and components['decay'] == 'both':
        config['decay_rate'] = 0.1
        config['update_fn'] = decay_both_update_fn
        config['fit_parameters'].append('decay_rate')
        config['parameter_bounds']['decay_rate'] = (0.0, 1.0)
        config['plausible_bounds']['decay_rate'] = (0.01, 0.75)
    elif 'decay' in components and components['decay'] == 'choice_kernel':
        config['decay_rate'] = 0.05  # lower rate for choice kernel
        config['update_fn'] = choice_kernel_decay_update_fn
        config['fit_parameters'].append('decay_rate')
        config['parameter_bounds']['decay_rate'] = (0.0, 1.0)
        config['plausible_bounds']['decay_rate'] = (0.01, 0.75)
    elif 'decay' in components and components['decay'] == 'collins_q':
        config['decay_rate'] = 0.15
        config['update_fn'] = collins_decay_q_update_fn
        config['fit_parameters'].append('decay_rate')
        config['parameter_bounds']['decay_rate'] = (0.0, 1.0)
        config['plausible_bounds']['decay_rate'] = (0.01, 0.75)
    elif 'decay' in components and components['decay'] == 'collins_both':
        config['decay_rate'] = 0.15
        config['update_fn'] = collins_decay_both_update_fn
        config['fit_parameters'].append('decay_rate')
        config['parameter_bounds']['decay_rate'] = (0.0, 1.0)
        config['plausible_bounds']['decay_rate'] = (0.01, 0.75)
    if 'asym_lr' in components:
        # Remove alpha and replace with alpha_pos, alpha_neg
        config['fit_parameters'].remove('alpha')
        config['alpha_pos'] = components.get('alpha_pos', 0.99)
        config['alpha_neg'] = components.get('alpha_neg', 0.01)
        config['fit_parameters'].extend(["alpha_pos", "alpha_neg"])
        
        # Add other parameters that might have been added before asym_lr
        existing_params = config['fit_parameters'].copy()
        for param in ['decision_noise', 'go_bias', 'pavlovian_weight', 'decay_rate']:
            if param in config and param not in existing_params:
                config['fit_parameters'].append(param)
                
        config['parameter_bounds'].update({
            'alpha_pos': (0.0, 1.0),
            'alpha_neg': (0.0, 1.0),
        })

        config['plausible_bounds'].update({
            'alpha_pos': (0.1, 0.8),
            'alpha_neg': (0.1, 0.8),
        })

        # override update function based on what's already set
        if 'decay' in components:
            if components['decay'] == 'q':
                config['update_fn'] = asym_decay_q_update_fn
            elif components['decay'] == 'choice_kernel':
                if 'dynamic_pav' in components:
                    config['update_fn'] = asym_pav_choice_kernel_update_fn
                else:
                    config['update_fn'] = asym_choice_kernel_update_fn
            elif components['decay'] == 'both' and 'dynamic_pav' in components:
                config['update_fn'] = asym_pav_decay_both_update_fn

        elif 'dynamic_pav' in components:
            config['update_fn'] = asym_pav_update_fn
        elif 'asym_based_on_reward' in components and components['asym_based_on_reward']:
            config['update_fn'] = asym_q_update_reward_fn
        elif 'asym_based_on_pe' in components and components['asym_based_on_pe']:
            config['update_fn'] = asym_q_update_pe_fn
        else:
            config['update_fn'] = asym_q_update_pe_fn

    # reward rate modulated go bias
    if 'reward_rate_go_bias' in components:
        config['reward_rate_mod'] = components['reward_rate_go_bias']
        config['logits_fn'] = reward_rate_go_bias_logits_fn
        config['update_fn'] = reward_rate_go_bias_update_fn
        config['fit_parameters'].append('reward_rate_mod')
        config['parameter_bounds']['reward_rate_mod'] = (-2.0, 2.0)
        config['plausible_bounds']['reward_rate_mod'] = (-1.0, 1.0)

    # context-dependent decay
    if 'context_decay' in components:
        config['variance_mod'] = components['context_decay']
        config['update_fn'] = context_decay_update_fn
        config['fit_parameters'].append('variance_mod')
        config['parameter_bounds']['variance_mod'] = (0.0, 2.0)
        config['plausible_bounds']['variance_mod'] = (0.1, 1.0)

    # win-stay boost
    if 'win_stay_boost' in components:
        config['win_go_boost'] = components['win_stay_boost']
        config['update_fn'] = win_stay_boost_update_fn
        config['fit_parameters'].append('win_go_boost')
        config['parameter_bounds']['win_go_boost'] = (0.0, 2.0)
        config['plausible_bounds']['win_go_boost'] = (0.1, 1.0)

    # bayesian controllability
    if 'bayesian_control' in components:
        config['omega_bias'] = components.get('omega_bias', 0.0)
        config['use_counterfactual'] = components.get('use_counterfactual', False)
        config['logits_fn'] = bayesian_control_logits_fn
        config['update_fn'] = bayesian_control_update_fn
        config['fit_parameters'].append('omega_bias')
        config['parameter_bounds']['omega_bias'] = (-4.0, 4.0)
        config['plausible_bounds']['omega_bias'] = (-2.0, 2.0)

    # bayesian + pavlovian
    if 'bayesian_pav' in components:
        config['omega_bias'] = components.get('omega_bias', 0.0)
        config['use_counterfactual'] = components.get('use_counterfactual', False)
        config['pavlovian_weight'] = components['bayesian_pav']
        config['initial_v_value'] = 0.0  # different from regular pav
        config['logits_fn'] = bayesian_pav_logits_fn
        config['update_fn'] = bayesian_pav_update_fn
        config['fit_parameters'].extend(['omega_bias', 'pavlovian_weight'])
        config['parameter_bounds'].update({
            'omega_bias': (-4.0, 4.0),
            'pavlovian_weight': (0.0, 1.0)
        })
        config['plausible_bounds'].update({
            'omega_bias': (-2.0, 2.0),
            'pavlovian_weight': (0.01, 0.75)
        })

    # simple omega control
    if 'omega_control' in components:
        config['omega_init'] = components.get('omega_init', 0.0)
        config['beta_control'] = components.get('beta_control', 1.0)
        config['initial_v_value'] = 0.0
        config['logits_fn'] = omega_control_logits_fn
        config['update_fn'] = omega_control_update_fn
        config['fit_parameters'].append('beta_control')
        config['parameter_bounds']['beta_control'] = (0.1, 10.0)
        config['plausible_bounds']['beta_control'] = (0.5, 5.0)

    if action_type == "epsilon":
        config['epsilon'] = 0.2
        config['fit_parameters'].append('epsilon')
        config['parameter_bounds']['epsilon'] = (0.0, 1.0)
        config['plausible_bounds']['epsilon'] = (0.05, 0.999)
    suffix = "_eps" if action_type == "epsilon" else ""
    config['name'] = f"{base_name}{suffix}"

    return RWConfig(**config)


# guitart-masip models + epsilon variants + decay variants
all_models = {
    # base models (no pavlovian)
    'rw': build_model('rw'),
    'rw_noise': build_model('rw_noise', decision_noise=0.1),
    'rw_noise_bias': build_model('rw_noise_bias', decision_noise=0.1, go_bias=0.1),

    # dynamic pavlovian (v_values update, can decay q, v, or both)
    'rw_noise_bias_pav_dynamic': build_model('rw_noise_bias_pav_dynamic', decision_noise=0.1, go_bias=0.1, pavlovian=0.1, dynamic_pav=True),
    'rw_noise_bias_asym_reward': build_model('rw_noise_bias_asym_reward', decision_noise=0.1, go_bias=0.1, asym_lr=True, asym_based_on_reward=True),
    'rw_noise_bias_pav_dynamic_asym_decay_both': build_model('rw_noise_bias_pav_dynamic_asym_decay_both', decision_noise=0.2, go_bias=0.05, pavlovian=0.07, dynamic_pav=True, decay='both', asym_lr=True),

    'rw_noise_bias_pav_dynamic_collins_decay_q': build_model('rw_noise_bias_pav_dynamic_collins_decay_q', decision_noise=0.2, go_bias=0.05, pavlovian=0.07, dynamic_pav=True, decay='collins_q'),
    'rw_noise_bias_pav_dynamic_collins_decay_both': build_model('rw_noise_bias_pav_dynamic_collins_decay_both', decision_noise=0.2, go_bias=0.05, pavlovian=0.1, dynamic_pav=True, decay='collins_both'),
    
    # new bayesian and omega control models
    'rw_bayesian_control': build_model('rw_bayesian_control', bayesian_control=True, go_bias=0.1),
    'rw_bayesian_control_counterfactual': build_model('rw_bayesian_control_counterfactual', bayesian_control=True, go_bias=0.1, use_counterfactual=True),
    'rw_bayesian_pav': build_model('rw_bayesian_pav', bayesian_pav=0.1, go_bias=0.1),
    'rw_bayesian_pav_counterfactual': build_model('rw_bayesian_pav_counterfactual', bayesian_pav=0.1, go_bias=0.1, use_counterfactual=True),
    'rw_omega_control': build_model('rw_omega_control', omega_control=True, go_bias=0.1, beta_control=2.0),
    
    # other new variants
    'rw_reward_rate_bias': build_model('rw_reward_rate_bias', reward_rate_go_bias=0.5, go_bias=0.1),
    'rw_context_decay': build_model('rw_context_decay', context_decay=0.5, decay='q'),
    'rw_win_stay_boost': build_model('rw_win_stay_boost', win_stay_boost=0.3, asym_lr=True, go_bias=0.1),
}

def return_configs():
    return [model for model in all_models.values() if model.included]

def get_models_for_fitting(include_names=None, exclude_names=None, only_included=True):
    """
    Get models for fitting based on flexible filtering criteria.
    
    Args:
        include_names: List of model name patterns to include (None = all)
        exclude_names: List of model name patterns to exclude (None = none)
        only_included: If True, only return models with included=True
    
    Returns:
        List of model configs matching the criteria
    """
    models = all_models.copy()
    
    # Add RLWM models to the pool
    models.update(rlwm_models)
    
    # Filter by included property
    if only_included:
        models = {k: v for k, v in models.items() if v.included}
    
    # Filter by include patterns
    if include_names:
        filtered = {}
        for pattern in include_names:
            for name, model in models.items():
                if pattern in name:
                    filtered[name] = model
        models = filtered
    
    # Filter out exclude patterns
    if exclude_names:
        for pattern in exclude_names:
            models = {k: v for k, v in models.items() if pattern not in k}
    
    return list(models.values())

def toggle_model_inclusion(model_names, include=True):
    """
    Easily toggle the included property for specific models.
    
    Args:
        model_names: String or list of model names to toggle
        include: True to include, False to exclude
    """
    if isinstance(model_names, str):
        model_names = [model_names]
    
    for name in model_names:
        if name in all_models:
            all_models[name].included = include
        if name in rlwm_models:
            rlwm_models[name].included = include

# Add RLWM models to the main model registry
all_models.update(rlwm_models)

# %%