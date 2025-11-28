#%% flexible_rw_agent.py
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from .agent import Agent

from .subject_data import SubjectData

@dataclass
class RWConfig:
    included: bool = True
    # kitchen sink approach - add whatever needed. the functions grab the values they need!
    alpha: Optional[float] = 0.1
    alpha_pos: Optional[float] = 0.1
    alpha_neg: Optional[float] = 0.1
    beta: Optional[float] = 1.0
    epsilon: Optional[float] = 0.1
    go_bias: Optional[float] = 0.0
    instrumental_weight: Optional[float] = 1.0
    pavlovian_weight: Optional[float] = 0.0
    decision_noise: Optional[float] = 0.0
    decay_rate: Optional[float] = 0.0
    omega_bias: Optional[float] = 0.0
    use_counterfactual: Optional[bool] = False
    omega_init: Optional[float] = 0.5
    beta_control: Optional[float] = 1.0
    reward_rate_mod: Optional[float] = 0.0
    variance_mod: Optional[float] = 0.0
    win_go_boost: Optional[float] = 0.0

    initial_q_value: Optional[float] = 0.0
    initial_q_values: Optional[np.ndarray] = None
    initial_v_value: Optional[float] = 0.0
    initial_v_values: Optional[np.ndarray] = None
    name: str = "rw"
    seed: Optional[int] = None
    reset_after_block: bool = True
    n_states: int = 4
    n_actions: int = 2
    wm_capacity: int = 4
    wm_decay: float = 0.1
    w0: float = 0.8
    controllability_sensitive: bool = False
    # functions - set defaults later
    logits_fn: Callable = None
    action_probs_fn: Callable = None
    update_fn: Callable = None
    choose_action_fn: Callable = None
    reset_fn: Callable = None

    # Fitting parameters
    fit_parameters: List[str] = field(
        default_factory=lambda: ["alpha", "beta"])
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "alpha": (0.01, 0.999),
        "alpha_pos": (0.01, 0.999),
        "alpha_neg": (0.01, 0.999),
        "beta": (0.1, 10.0),
        "epsilon": (0.01, 0.999),
        "go_bias": (-10.0, 10.0),
        "instrumental_weight": (0.0, 1.0),
        "pavlovian_weight": (0.0, 1.0),
        "decision_noise": (0.0, 1.0),
        "decay_rate": (0.0, 1.0),
    })
    # Plausible bounds for parameter values
    plausible_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "alpha": (0.1, 0.8),
        "alpha_pos": (0.1, 0.8),
        "alpha_neg": (0.1, 0.8),
        "beta": (1.0, 8.0),
        "epsilon": (0.01, 0.5),
        "go_bias": (-1.0, 1.0),
        "instrumental_weight": (0.1, 1.0),
        "pavlovian_weight": (0.0, 1.0),
        "decision_noise": (0.0, 0.5),
        "decay_rate": (0.0, 0.5),
    })



class RWAgent(Agent):
    def __init__(self, config: RWConfig):
        super().__init__(config.n_actions, config.n_states, config.name)
        self.config = config

        # rng
        self._rng = np.random.default_rng(config.seed)

        # copy all config params to self for easy access
        for key, value in config.__dict__.items():
            setattr(self, key, value)

        # q-values
        if config.initial_q_values is not None:
            self.q_values = config.initial_q_values.copy()
        else:
            self.q_values = np.full((config.n_states, config.n_actions),
                                  config.initial_q_value, dtype=np.float64)

        self.q_values += self._rng.normal(0, 0.01, self.q_values.shape)
        self._init_q = self.q_values.copy()

        if config.initial_v_values is not None:
            self.v_values = config.initial_v_values.copy()
        else:
            self.v_values = np.array([config.initial_v_value, -config.initial_v_value, config.initial_v_value, -config.initial_v_value], dtype=np.float64)

        self.v_values += self._rng.normal(0, 0.01, self.v_values.shape)
        self._init_v = self.v_values.copy()

        self.visit_counts = np.zeros((self.n_states, self.n_actions))

        # history
        self.history = {
            'observations': [], 'actions': [], 'rewards': [],
            'prediction_errors': [], 'q_values': [], 'action_probabilities': []
        }

        self.q_wm = np.full((self.n_states, self.n_actions), 0.5, dtype=np.float64)
        self.w = np.full(self.n_states, self.w0, dtype=np.float64)
        self._wm_contents = set()
        self._last_seen = {}
        self._trial_count = 0
        self._control_tracker = {'correct_rewards': 0, 'total_trials': 0}
    def get_logits(self, observation: int) -> np.ndarray:
        return self.logits_fn(self, observation)

    def get_action_probs(self, logits: np.ndarray) -> np.ndarray:
        return self.action_probs_fn(self, logits)

    def choose_action(self, probs: np.ndarray, log_history: bool) -> int:
        return self.choose_action_fn(self, probs, log_history=log_history)

    def update(self, observation: int, action: int, reward: float,
               next_observation: int, terminated: bool, log_history:bool) -> float:
        return self.update_fn(self, observation, action, reward,
                                   next_observation, terminated, log_history)
    def reset(self, seed:int, reset_history:bool=False) -> None:
        self.reset_fn(self, seed, reset_history)

    def _get_name(self) -> str:
        return self.name

    def _get_description(self) -> str:
        return f"{self.name}: alpha={self.alpha}, beta={self.beta}"

    def _is_in_wm(self, stimulus):
        """Check if stimulus is currently in WM based on capacity"""
        # Simple capacity model: most recently seen stimuli up to capacity
        if len(self._wm_contents) < self.wm_capacity:
            self._wm_contents.add(stimulus)
            return True
        elif stimulus in self._wm_contents:
            return True
        else:
            # Need to replace something - use LRU
            if len(self._wm_contents) >= self.wm_capacity:
                # Remove least recently used
                lru_stim = min(self._wm_contents,
                             key=lambda s: self._last_seen.get(s, -1))
                self._wm_contents.remove(lru_stim)
            self._wm_contents.add(stimulus)
            return True

    def _update_mixture_weights(self, stimulus, action, reward):
        """Update mixture weights based on prediction accuracy"""
        # Get predictions from both systems
        q_rl_pred = self.q_values[stimulus, action]
        q_wm_pred = self.q_wm[stimulus, action]

        # Compute likelihoods (simplified - could use full Bayesian)
        # Higher likelihood for system that better predicted the reward
        rl_error = abs(reward - q_rl_pred)
        wm_error = abs(reward - q_wm_pred)

        # Convert errors to likelihoods (smaller error = higher likelihood)
        rl_likelihood = np.exp(-self.beta * rl_error)
        wm_likelihood = np.exp(-self.beta * wm_error)

        # Update weight using Bayes rule
        prior_wm = self.w[stimulus]
        prior_rl = 1 - prior_wm

        posterior_wm = (wm_likelihood * prior_wm) / (
            wm_likelihood * prior_wm + rl_likelihood * prior_rl + 1e-10
        )

        # Smooth update
        self.w[stimulus] = 0.9 * self.w[stimulus] + 0.1 * posterior_wm

        # Track last seen for LRU
        self._last_seen[stimulus] = self._trial_count
        self._trial_count += 1

    def compute_log_likelihood(self, data: SubjectData = None) -> float:
        """
        Compute the log likelihood of the agent's actions given a single subject's data.
        """
        
        observations, actions, rewards = data.observations, data.actions, data.rewards
        log_likelihood = 0.0
        n_blocks = observations.shape[0]
        n_trials = observations.shape[1]
        
        for block_idx in range(n_blocks):
            for trial_idx in range(n_trials):
                obs = observations[block_idx, trial_idx]
                action = actions[block_idx, trial_idx]
                reward = rewards[block_idx, trial_idx]

                if obs is None or action is None or reward is None:
                    continue

                logits = self.get_logits(obs)
                action_probs = self.get_action_probs(logits)

                # Compute log likelihood of the chosen action
                log_prob = np.log(action_probs[action])
                log_likelihood += log_prob

                pred_error = self.update(obs, action, reward, obs, terminated=False, log_history=False)


        return log_likelihood



# %%
