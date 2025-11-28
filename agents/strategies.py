from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


class ActionSelectionStrategy(ABC):
    @abstractmethod
    def select_action(self, q_values: np.ndarray, rng: np.random.Generator, **kwargs) -> int:
        """Select an action based on Q-values and the current state."""
        pass

    @abstractmethod
    def _get_action_probabilities(self, q_values: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate action probabilities based on Q-values."""
        pass


class SigmoidStrategy(ActionSelectionStrategy):
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def select_action(self, q_values: np.ndarray, rng: np.random.Generator, **kwargs) -> int:
        if len(q_values) != 2:
            raise ValueError("Sigmoid strategy only works with 2 actions")
        probabilities = self._get_action_probabilities(q_values)
        return rng.choice(2, p=probabilities)

    def _get_action_probabilities(self, q_values: np.ndarray, **kwargs) -> np.ndarray:
        if len(q_values) != 2:
            raise ValueError("Sigmoid strategy only works with 2 actions")

        diff_q_scaled = self.beta * (q_values[0] - q_values[1])

        if diff_q_scaled < -700:
            prob_action0 = 0.0
        elif diff_q_scaled > 700:
            prob_action0 = 1.0
        else:
            prob_action0 = 1.0 / (1.0 + np.exp(-diff_q_scaled))

        return np.array([prob_action0, 1.0 - prob_action0])


class EpsilonGreedyStrategy(ActionSelectionStrategy):
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def select_action(self, q_values: np.ndarray, rng: np.random.Generator, **kwargs) -> int:
        if rng.random() < self.epsilon or q_values[0] == q_values[1]:
            return rng.choice(len(q_values))
        else:
            return np.argmax(q_values)

    def _get_action_probabilities(self, q_values: np.ndarray, **kwargs) -> np.ndarray:
        n_actions = len(q_values)
        probs = np.full(n_actions, self.epsilon / n_actions)
        best_action = np.argmax(q_values)
        probs[best_action] += 1.0 - self.epsilon
        return probs


class GreedyStrategy(ActionSelectionStrategy):
    def select_action(self, q_values: np.ndarray, rng: np.random.Generator, **kwargs) -> int:
        if q_values[0] == q_values[1]:
            return rng.choice(len(q_values))
        else:
            return np.argmax(q_values)

    def _get_action_probabilities(self, q_values: np.ndarray, **kwargs) -> np.ndarray:
        probs = np.zeros(len(q_values))
        probs[np.argmax(q_values)] = 1.0
        return probs

# Update Rule Strategies
class UpdateStrategy(ABC):
    @abstractmethod
    def update(self, q_values: np.ndarray, observation: int, action: int,
               reward: float, next_observation: Optional[int],
               terminated: bool, **kwargs) -> float:
        """Update Q-values and return prediction error"""
        pass

class RescorlaWagnerUpdate(UpdateStrategy):
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def update(self, q_values: np.ndarray, observation: int, action: int,
               reward: float, next_observation: Optional[int],
               terminated: bool, **kwargs) -> float:
        prediction = q_values[observation, action]
        target = reward
        prediction_error = target - prediction
        q_values[observation, action] += self.alpha * prediction_error
        return prediction_error