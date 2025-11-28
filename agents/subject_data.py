from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from numpy.typing import NDArray
import numpy as np

@dataclass
class SubjectData:
    observations: NDArray[int]
    actions: NDArray[int]
    rewards: NDArray[int]
    subject: str = None
    block_type: NDArray[str] = None
    step: NDArray[int] = None
    block_id: NDArray[int] = None
    response_times: NDArray[float] = None
    num_blocks: int = 0

    def __post_init__(self):
        self.transpose()

    def transpose(self):
        if len(self.observations) % 40 != 0:
            raise ValueError(f"Data length {len(self.observations)} not divisible by 40")
        self.observations = np.reshape(self.observations, (-1, 40))
        self.actions = np.reshape(self.actions, (-1, 40))
        self.rewards = np.reshape(self.rewards, (-1, 40))
        self.response_times = np.reshape(self.response_times, (-1, 40))
        self.block_type = np.reshape(self.block_type, (-1, 40))
        self.block_id = np.reshape(self.block_id, (-1, 40))
        self.step = np.reshape(self.step, (-1, 40))
        self.num_blocks = self.block_id.shape[0]
        return self

    def __iter__(self):
        for i in range(self.num_blocks):
            yield dict(
                observations=self.observations[i],
                actions=self.actions[i],
                rewards=self.rewards[i],
                response_times=self.response_times[i],
                block_type=self.block_type[i],
                block_id=self.block_id[i],
                step=self.step[i],
                subject=self.subject
            )