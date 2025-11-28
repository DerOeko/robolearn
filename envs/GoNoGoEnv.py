# %%
import gymnasium as gym
import numpy as np
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class GoNoGoConfig:
    """Simplified configuration for Go-NoGo environment."""
    n_blocks: int = 8
    block_schedule: Optional[List[str]] = None  # ['low', 'high', 'low', ...]
    add_calibration: bool = True  # Whether to add a calibration block at the start
    # Custom sequences for each block, if provided
    predefined_blocks: Optional[List[Dict[str, Any]]] = None
    # Number of unique stimuli (4 types) - go2win, nogo2win, go2avoid, nogo2avoid
    n_states: int = 4
    # Index for predefined controllability schedule
    c_schedule_idx: Optional[int] = None
    n_yoked: int = 2

class GoNoGoEnv(gym.Env):
    """
    Simplified Go-NoGo environment for studying controllability and decision-making.

    Features:
    - Two main block types: low control (20% control rate) and high control (80% control rate)
    - 4 stimulus types per block with different valences and optimal actions
    - Optional calibration block at the start
    - Configurable stimulus sequences and block schedules
    - Yoked blocks that use reward rates from calibration performance
    """

    # Predefined block schedules for consistency across experiments
    BLOCK_SCHEDULES = [
        ['low', 'high', 'high', 'low', 'high', 'low', 'low', 'high'],
        ['high', 'low', 'high', 'high',
         'low', 'high', 'low', 'low'],
        ['low', 'high', 'low', 'high', 'high', 'low', 'high', 'low'],
        ['low', 'low', 'high', 'low', 'high', 'high', 'low', 'high'],
        ['high', 'low', 'low', 'high', 'low', 'high', 'high', 'low'],
        ['low', 'high', 'low', 'low', 'high', 'low', 'high', 'high'],
        ['high', 'low', 'high', 'low', 'low', 'high', 'low', 'high'],
        ['high', 'high', 'low', 'high',
         'low', 'low', 'high', 'low'],
        ['low', 'low', 'high', 'high', 'low', 'high', 'low', 'high'],
        ['high', 'high', 'low', 'low', 'high', 'low', 'high', 'low'],
        ['low', 'high', 'low', 'high', 'low', 'high', 'low', 'high']
    ]

    metadata = {"render.modes": [], "name": "GoNoGoEnv-v0"}

    def __init__(self, config: GoNoGoConfig = None):
        super().__init__()

        self.config = config or GoNoGoConfig()

        if self.config.c_schedule_idx is None and self.config.block_schedule is None:
            self.config.c_schedule_idx = np.random.randint(0, 11)  # Default to first schedule if none provided

        assert (self.config.c_schedule_idx is not None) != (self.config.block_schedule is not None), \
            "Provide either c_schedule_idx OR block_schedule, but not both or neither."

        if self.config.add_calibration:
            self.calibration_data = {
                "total": 0,
                "correct": 0,
                "reward_rate": 0.8
            }

            self.yoked_block_indices = []

        if self.config.predefined_blocks:
            assert self.config.block_schedule is not None, \
                "Custom blocks cannot be used without a predefined block schedule."
            assert len(self.config.predefined_blocks) == len(self.config.block_schedule), \
                "Num of custom blocks must match the number of blocks in the predefined schedule."
            self.blocks = self.config.predefined_blocks
        else:
            self.blocks = []
            self._setup_blocks()

        self.current_block = 0
        self.current_trial = 0

        self.action_space = gym.spaces.Discrete(2)  # 0: No-Go, 1: Go
        self.observation_space = gym.spaces.Discrete(
            self.config.n_states
        )

    def _setup_blocks(self):
        self.block_types = {
            'low': {'control_rate': 0.2, 'reward_rate': 0.8},
            'high': {'control_rate': 0.8, 'reward_rate': 0.8},
            'calibration': {'control_rate': 0.8, 'reward_rate': 0.8},
            'yoked': {'control_rate': 0.2, 'reward_rate': None}
        }

        if self.config.block_schedule:
            schedule = self.config.block_schedule.copy()
        elif self.config.c_schedule_idx is not None:
            schedule = self.BLOCK_SCHEDULES[self.config.c_schedule_idx].copy()
        else:
            schedule = self.BLOCK_SCHEDULES[0].copy()

        # Randomly select blocks to be yoked (replace 'low' with 'yoked')
        low_indices = [i for i, block_type in enumerate(schedule) if block_type == 'low']
        if len(low_indices) >= self.config.n_yoked:
            self.yoked_block_indices = random.sample(low_indices, self.config.n_yoked)
            for idx in self.yoked_block_indices:
                schedule[idx] = 'yoked'

        if self.config.add_calibration:
            schedule = ['calibration'] + schedule
            self.yoked_block_indices = [idx + 1 for idx in self.yoked_block_indices]

        self.schedule = schedule

        for i, block_type in enumerate(schedule):
            block = self._create_block(block_type, i)
            self.blocks.append(block)

    def _create_block(self, block_type: str, index: int) -> Dict[str, Any]:

        config = self.block_types.get(block_type, {})

        if block_type == "yoked":
            reward_rate = self.calibration_data['reward_rate']
        else:
            reward_rate = config['reward_rate']

        stimuli = list(range(self.config.n_states))

        trials = self._generate_trials(
            stimuli, config['control_rate'], reward_rate)

        return {
            'type': block_type,
            'index': index,
            'control_rate': config['control_rate'],
            'reward_rate': reward_rate,
            'stimuli': stimuli,
            'trials': trials,
        }

    def _generate_trials(self, stimuli: List[int], control_rate: float, reward_rate: float) -> Dict[str, List]:
        n_per_stimulus = 10  # Number of repetitions per stimulus
        total_trials = len(stimuli) * n_per_stimulus

        stimulus_seq = []
        control_seq = []
        reward_seq = []

        for stimulus in stimuli:
            stimulus_seq.extend([stimulus] * n_per_stimulus)

            n_control = int(n_per_stimulus * control_rate)
            n_reward = int(n_per_stimulus * reward_rate)
            control_seq.extend([1] * n_control + [0] *
                               (n_per_stimulus - n_control))
            reward_seq.extend([1] * n_reward + [0] *
                              (n_per_stimulus - n_reward))

        # Shuffle the trials
        indices = list(range(total_trials))
        random.shuffle(indices)

        return {
            'stimulus': [stimulus_seq[i] for i in indices],
            'is_control': [control_seq[i] for i in indices],
            'is_rewarded': [reward_seq[i] for i in indices],
        }

    def _update_calibration_data(self, stimulus: int, action: int):
        """Update calibration performance tracking."""
        if self.blocks[self.current_block]['type'] == 'calibration':
            self.calibration_data['total_trials'] += 1
            if self._is_optimal_action(stimulus, action):
                self.calibration_data['correct_trials'] += 1

    def _finalize_calibration(self):
            """Calculate final reward rate from calibration and update yoked blocks."""
            if self.calibration_data['total_trials'] > 0:
                performance_rate = self.calibration_data['correct_trials'] / self.calibration_data['total_trials']
                self.calibration_data['reward_rate'] = performance_rate

                # Update yoked blocks with calibration-derived reward rate
                for block_idx in self.yoked_block_indices:
                    if block_idx < len(self.blocks):
                        self.blocks[block_idx]['reward_rate'] = performance_rate
                        # Regenerate trials with new reward rate
                        self.blocks[block_idx]['trials'] = self._generate_trials(
                            self.blocks[block_idx]['stimuli'],
                            self.blocks[block_idx]['control_rate'],
                            performance_rate
                        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.config.c_schedule_idx = np.random.randint(0, 11) if self.config.c_schedule_idx is None else self.config.c_schedule_idx

        if self.config.predefined_blocks:
            assert self.config.block_schedule is not None, \
                "Custom blocks cannot be used without a predefined block schedule."
            assert len(self.config.predefined_blocks) == len(self.config.block_schedule), \
                "Num of custom blocks must match the number of blocks in the predefined schedule."
            self.blocks = self.config.predefined_blocks
        else:
            self.blocks = []
            self._setup_blocks()

        # Reset calibration data
        self.calibration_data = {
            'total_trials': 0,
            'correct_trials': 0,
            'reward_rate': 0.8
        }

        self.current_block = 0
        self.current_trial = 0

        if len(self.blocks) > 0:
            block = self.blocks[self.current_block]
            return block['trials']['stimulus'][self.current_trial], {}
        else:
            return 0, {}

    def step(self, action):
        if self.current_block >= len(self.blocks):
            return 0, 0, True, True, {}

        current_block_data = self.blocks[self.current_block]
        current_trials = current_block_data['trials']

        # Move to next block if all trials in the current block are done
        if self.current_trial >= len(current_trials['stimulus']):
            self.current_block += 1
            self.current_trial = 0

            if current_block_data['type'] == 'calibration':
                self._finalize_calibration()

            if self.current_block >= len(self.blocks):
                return 0, 0, True, True, {}

            current_block_data = self.blocks[self.current_block]
            current_trials = current_block_data['trials']

        # Get current stimulus and its properties
        stimulus = current_trials['stimulus'][self.current_trial]
        is_control = current_trials['is_control'][self.current_trial]
        is_rewarded = current_trials['is_rewarded'][self.current_trial]

        # Update calibration data if in calibration block
        self._update_calibration_data(stimulus, action)

        reward = 0

        if is_control:
            if is_rewarded:
                # In control trials, check if the action is correct
                reward = 1 if self._is_optimal_action(stimulus, action) else -1
            else:
                reward = -1 if self._is_optimal_action(stimulus, action) else 1
        else:
            reward = 1 if is_rewarded else -1

        self.current_trial += 1

        terminated = False
        truncated = False

        if self.current_trial >= len(current_trials['stimulus']):
            if self.current_block >= len(self.blocks) - 1:
                terminated = True
                next_obs = 0
            else:
                next_obs = self.blocks[self.current_block +
                                       1]['trials']['stimulus'][0]
        else:
            next_obs = current_trials['stimulus'][self.current_trial]

        return next_obs, reward, terminated, truncated, {"block_type": current_block_data['type'], "block_index": current_block_data['index'], "is_control": is_control, "is_rewarded": is_rewarded}

    def _is_optimal_action(self, stimulus: int, action: int) -> bool:
        optimal_actions = {
            0: 1,  # go2win
            1: 1,  # go2avoid  
            2: 0,  # nogo2win
            3: 0   # nogo2avoid
        }
        return action == optimal_actions.get(stimulus, 0)

    def get_yoked_block_info(self) -> Dict[str, Any]:
        """Return information about yoked blocks and calibration performance."""
        return {
            'yoked_block_indices': self.yoked_block_indices,
            'calibration_reward_rate': self.calibration_data['reward_rate'],
            'calibration_performance': self.calibration_data['correct_trials'] / max(1, self.calibration_data['total_trials'])
        }
# %%