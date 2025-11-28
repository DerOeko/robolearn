# simulation_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from agents.rw_agent import RWAgent, RWConfig

class SimulationRunner:
    """Class for running agent simulations with block-trial structure"""

    def __init__(self, environment, agent: RWAgent):
        self.env = environment
        self.agent = agent
        self.calibration_stats = {"correct": 0, "total": 0}

    def _run_block(self, block_idx: int, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run a single block and return trial-by-trial data"""

        block_trials = []
        trials = block['trials']
        n_trials = len(trials['stimulus'])

        for trial_idx in range(n_trials):
            # Get trial information
            stimulus = trials['stimulus'][trial_idx]
            is_control = trials['is_control'][trial_idx]
            is_rewarded = trials['is_rewarded'][trial_idx]

            # clean pipeline: obs -> logits -> probs -> action
            current_logits = self.agent.get_logits(stimulus)
            action_probs = self.agent.get_action_probs(current_logits)
            action = self.agent.choose_action(action_probs, log_history=False)

            # Compute reward based on trial properties
            reward = self._compute_reward(stimulus, action, is_control, is_rewarded)

            # Agent learns from trial
            # For trial-based learning, next_obs is typically same as current
            pred_error = self.agent.update(stimulus, action, reward, stimulus,
                                         terminated=False, log_history=False)

            # Store trial data
            trial_data = {
                'block_idx': block_idx,
                'trial_idx': trial_idx,
                'stimulus': stimulus,
                'action': action,
                'reward': reward,
                'prediction_error': pred_error,
                'logits': current_logits.copy(),
                'action_probabilities': action_probs.copy(),
                'block_type': block['type'],
                'control_rate': block['control_rate'],
                'reward_rate': block['reward_rate'],
                'is_control': is_control,
                'is_rewarded': is_rewarded,
                'optimal_action': self.env._is_optimal_action(stimulus, action)
            }

            if block['type'] == 'calibration':
                self.calibration_stats['total'] += 1
                if trial_data['optimal_action']:
                    self.calibration_stats['correct'] += 1

            block_trials.append(trial_data)

        return block_trials
    def get_calibration_rate(self):
        if self.calibration_stats['total'] == 0:
            return 0.8  # default
        return self.calibration_stats['correct'] / self.calibration_stats['total']

    def _compute_reward(self, stimulus: int, action: int, is_control: bool, is_rewarded: bool) -> float:
        """Compute reward based on trial properties"""
        if is_control:
            if is_rewarded:
                return 1 if self.env._is_optimal_action(stimulus, action) else -1
            else:
                return -1 if self.env._is_optimal_action(stimulus, action) else 1
        else:
            return 1 if is_rewarded else -1

    def run_experiment(self, reset_agent: bool = True,
                                 reset_seed: Optional[int] = None,
                                 agent_seed: Optional[int] = None) -> pd.DataFrame:
        """Run experiment and return as structured DataFrame"""

        if reset_agent:
            if agent_seed is not None:
                self.agent.reset(seed=agent_seed)
            elif reset_seed is not None:
                # Fallback to reset_seed if no agent_seed provided (backward compatibility)
                self.agent.reset(seed=reset_seed)
            else:
                self.agent.reset()

        # Reset environment (always use reset_seed for environment)
        if reset_seed is not None:
            obs, info = self.env.reset(seed=reset_seed)
        else:
            obs, info = self.env.reset()

        all_trial_data = []

        # Process each block
        for block_idx, block in enumerate(self.env.blocks):
            block_trials = self._run_block(block_idx, block)
            if reset_agent:
                if agent_seed is not None:
                    self.agent.reset(seed=agent_seed)
                elif reset_seed is not None:
                    # Fallback to reset_seed if no agent_seed provided (backward compatibility)
                    self.agent.reset(seed=reset_seed)
                else:
                    self.agent.reset()
            all_trial_data.extend(block_trials)

        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(all_trial_data)
        return df

def compare_agents(configs: List[RWConfig],
                                          environment,
                                          n_simulations: int = 10,
                                          seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare multiple agent configurations using structured analysis with multiple runs"""

    config_names = [config.name for config in configs]
    
    # Generate seeds for reproducible comparisons
    if seeds is None:
        np.random.seed(42)  # For reproducible seed generation
        env_seeds = np.random.randint(0, 10000, n_simulations).tolist()
    else:
        env_seeds = seeds
        if len(env_seeds) != n_simulations:
            raise ValueError(f"Number of seeds ({len(env_seeds)}) must match n_simulations ({n_simulations})")
    
    # Generate separate agent seeds for each configuration to ensure different action randomness
    # Each agent config gets a different seed range to avoid overlap
    np.random.seed(42)  # Reset for consistent agent seed generation
    agent_seed_ranges = {}
    for i, name in enumerate(config_names):
        # Each agent gets seeds from a different range: [10000-20000), [20000-30000), etc.
        base_seed = 10000 + (i * 10000)
        agent_seed_ranges[name] = np.random.randint(base_seed, base_seed + 10000, n_simulations).tolist()

    results = {}

    for config, name in zip(configs, config_names):
        print(f"Running {n_simulations} simulations for {name}...")

        all_dataframes = []
        summary_stats = []
        agent_seeds = agent_seed_ranges[name]

        for sim_idx, (env_seed, agent_seed) in enumerate(zip(env_seeds, agent_seeds)):

            agent = RWAgent(
                config=config
            )

            runner = SimulationRunner(environment, agent)
            df = runner.run_experiment(reset_seed=env_seed, agent_seed=agent_seed)
            df['simulation_id'] = sim_idx  # Track simulation ID
            df['env_seed'] = env_seed  # Track environment seed used
            df['agent_seed'] = agent_seed  # Track agent seed used
            all_dataframes.append(df)

            # Compute summary stats for this simulation
            sim_stats = {
                'simulation_id': sim_idx,
                'total_reward': df['reward'].sum(),
                'mean_reward': df['reward'].mean(),
                'accuracy': df['optimal_action'].mean(),
                'mean_abs_pe': df['prediction_error'].abs().mean(),
                'final_entropy': np.mean([
                    -np.sum(probs * np.log(probs + 1e-10))
                    for probs in df['action_probabilities'].iloc[-10:]  # Last 10 trials
                ])
            }

            # Block-specific performance
            for block_type in df['block_type'].unique():
                block_data = df[df['block_type'] == block_type]
                sim_stats[f'{block_type}_mean_reward'] = block_data['reward'].mean()
                sim_stats[f'{block_type}_accuracy'] = block_data['optimal_action'].mean()

            summary_stats.append(sim_stats)

        combined_df = pd.concat(all_dataframes, ignore_index=True)
        summary_df = pd.DataFrame(summary_stats)

        aggregate_stats = {
            'mean_total_reward': summary_df['total_reward'].mean(),
            'std_total_reward': summary_df['total_reward'].std(),
            'mean_accuracy': summary_df['accuracy'].mean(),
            'std_accuracy': summary_df['accuracy'].std(),
            'mean_abs_pe': summary_df['mean_abs_pe'].mean(),
            'std_abs_pe': summary_df['mean_abs_pe'].std(),
        }
    # Add block-specific aggregate stats
        for block_type in combined_df['block_type'].unique():
            if f'{block_type}_mean_reward' in summary_df.columns:
                aggregate_stats[f'{block_type}_reward_mean'] = summary_df[f'{block_type}_mean_reward'].mean()
                aggregate_stats[f'{block_type}_reward_std'] = summary_df[f'{block_type}_mean_reward'].std()
                aggregate_stats[f'{block_type}_accuracy_mean'] = summary_df[f'{block_type}_accuracy'].mean()
                aggregate_stats[f'{block_type}_accuracy_std'] = summary_df[f'{block_type}_accuracy'].std()

        results[name] = {
            'config': config,
            'combined_data': combined_df,
            'simulation_summaries': summary_df,
            'aggregate_stats': aggregate_stats,
            'n_simulations': n_simulations
        }

    return results