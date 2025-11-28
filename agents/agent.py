from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class Agent(ABC):
    def __init__(self, n_actions: int, n_states: int, name: str = "Agent"):
        self.n_actions = n_actions
        self.n_states = n_states
        self.name = name

    @abstractmethod
    def _get_name(self) -> str:
        pass

    @abstractmethod
    def _get_description(self) -> str:
        pass

    @abstractmethod
    def get_logits(self, observation: int) -> np.ndarray:
        """Getting action values of action in state `observation`."""
        pass

    @abstractmethod
    def get_action_probs(self, logits: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def choose_action(self, action_probs: np.ndarray, log_history: bool) -> int:
        pass

    @abstractmethod
    def update(self, observation: int, action: int, reward: float,
               next_observation: int, terminated: bool, log_history: bool) -> float:
        pass

    @abstractmethod
    def reset(self, seed: int, reset_history: bool) -> None:
        pass
