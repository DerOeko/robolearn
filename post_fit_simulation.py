"""
Post-fit simulation module for Rescorla-Wagner modeling project.

Handles simulation from fitted parameters for both PyVBMC (posterior distributions)
and PyBADS (point estimates) with equal and hierarchical sampling strategies.
Compatible with existing SimulationRunner and DataAnalyzer infrastructure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from dataclasses import dataclass, field
import pickle
import dill
from pathlib import Path
import logging

from agents.rw_agent import RWAgent
from simulation_analysis import SimulationRunner
from envs.GoNoGoEnv import GoNoGoEnv, GoNoGoConfig
from fitting import (
    list_all_parallel_fit_results, 
    find_latest_parallel_fit_results,
    load_specific_parallel_fit_results
)


@dataclass
class SimulationConfig:
    """Configuration for post-fit simulations."""
    n_simulations: int = 1000
    sampling_strategy: str = "equal"  # "equal" or "hierarchical"
    random_seed: Optional[int] = None
    environment_config: Optional[Dict] = None
    

@dataclass 
class FitResult:
    """Container for fitted parameters from either PyVBMC or PyBADS."""
    participant_id: str
    parameters: Dict[str, Union[float, np.ndarray]]  # Point estimates or samples
    log_likelihood: float
    fit_type: str  # "pyvbmc" or "pybads" 
    model_name: str
    agent_config: Any  # The actual RWConfig used during fitting
    metadata: Dict[str, Any] = field(default_factory=dict)


class PostFitSimulator:
    """
    Post-fit simulation system for RW models.
    
    Handles both PyVBMC (posterior samples) and PyBADS (point estimates)
    with flexible sampling strategies. Uses existing SimulationRunner infrastructure
    and returns DataFrames compatible with DataAnalyzer.
    
    Features:
    - Equal or hierarchical (likelihood-weighted) participant sampling
    - Automatic detection of PyVBMC vs PyBADS fitted parameters
    - Controllability schedule cycling: each simulation uses a different
      controllability schedule (0-10), ensuring consistent environments
      across agents within the same simulation
    - Compatible with existing analysis pipelines (DataAnalyzer, etc.)
    """
    
    def __init__(self, 
                 fitted_results: Union[Dict, List[FitResult], str, Path],
                 config: Optional[SimulationConfig] = None):
        """
        Initialize post-fit simulator.
        
        Args:
            fitted_results: Fitted parameters (dict, list of FitResult, or path to pickle)
            config: Simulation configuration
        """
        self.logger = self._setup_logger()
        self.config = config or SimulationConfig()
        self.fitted_results = self._load_fitted_results(fitted_results)
        
        # Set up base environment config (will be updated per simulation)
        self.base_env_config = self.config.environment_config or {
            'n_blocks': 8,
            'add_calibration': True,
            'n_states': 4,
            'c_schedule_idx': None  # Will be set per simulation
        }
        
            
        self._validate_fitted_results()
        self._setup_seeds()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for simulation tracking."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)  # INFO level for normal use
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')  # Simplified format
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _setup_seeds(self):
        """Setup seed sequences following simulation_analysis.py pattern."""
        # Generate env seeds (same for all participants for comparability)
        np.random.seed(42)  # For reproducible seed generation
        self.env_seeds = np.random.randint(0, 10000, self.config.n_simulations).tolist()
        
        # Generate separate agent seeds for each participant (different randomness)
        np.random.seed(42)  # Reset for consistent agent seed generation
        self.agent_seed_ranges = {}
        
        for i, result in enumerate(self.fitted_results):
            # Each participant gets seeds from a different range to avoid overlap
            base_seed = 10000 + (i * 10000)
            self.agent_seed_ranges[result.participant_id] = np.random.randint(
                base_seed, base_seed + 10000, self.config.n_simulations
            ).tolist()
    
    def _load_fitted_results(self, fitted_results: Union[Dict, List[FitResult], str, Path]) -> List[FitResult]:
        """Load fitted results from various input formats."""
        if isinstance(fitted_results, (str, Path)):
            fitted_results_str = str(fitted_results)
            
            # Check if it's a timestamp (format: YYYYMMDD_HHMMSS)
            if (len(fitted_results_str) == 15 and 
                fitted_results_str[8] == '_' and 
                fitted_results_str.replace('_', '').isdigit()):
                # Load by timestamp
                data, _ = load_specific_parallel_fit_results(fitted_results_str)
                if data is None:
                    raise ValueError(f"Could not load results for timestamp: {fitted_results_str}")
                return self._parse_parallel_fit_data(data)
            else:
                # Load from file path
                with open(fitted_results, 'rb') as f:
                    try:
                        data = dill.load(f)
                    except Exception:
                        data = pickle.load(f)
                return self._parse_fitted_data(data)
        
        elif isinstance(fitted_results, list) and all(isinstance(r, FitResult) for r in fitted_results):
            return fitted_results
        
        elif isinstance(fitted_results, dict):
            # Check if this is parallel fit data format vs legacy format
            if self._is_parallel_fit_format(fitted_results):
                self.logger.info(f"Detected parallel fit format with {len(fitted_results)} models")
                return self._parse_parallel_fit_data(fitted_results)
            else:
                self.logger.info(f"Detected legacy fit format with {len(fitted_results)} entries")
                return self._parse_fitted_data(fitted_results)
        
        else:
            raise ValueError("fitted_results must be dict, list of FitResult, or path to pickle file")
    
    def _is_parallel_fit_format(self, data: Dict) -> bool:
        """Check if data is in parallel fit format vs legacy format."""
        # Parallel fit format has model names as keys, each with 'individual_results', 'config', etc.
        # Legacy format has participant IDs as keys with direct fit data
        
        for key, value in data.items():
            if isinstance(value, dict):
                value_keys = list(value.keys())
                self.logger.debug(f"Checking format for key '{key}': {value_keys}")
                
                # Check for parallel fit structure indicators
                if 'individual_results' in value and 'config' in value:
                    self.logger.debug(f"Found parallel fit indicators for '{key}'")
                    return True
                # If we find direct fit data indicators, it's legacy format
                if 'parameters' in value or 'x' in value or 'fitted_params' in value:
                    self.logger.debug(f"Found legacy fit indicators for '{key}'")
                    return False
        
        # Default to legacy format if unclear
        self.logger.debug("Format unclear, defaulting to legacy")
        return False
    
    def _parse_fitted_data(self, data: Dict) -> List[FitResult]:
        """Parse fitted data from various formats into FitResult objects."""
        results = []
        
        for participant_id, fit_data in data.items():
            if isinstance(fit_data, dict):
                # Detect fit type based on data structure
                fit_type = self._detect_fit_type(fit_data)
                
                # Extract parameters
                if fit_type == "pyvbmc":
                    parameters = self._extract_pyvbmc_parameters(fit_data)
                else:
                    parameters = self._extract_pybads_parameters(fit_data)
                
                # Extract log likelihood and metadata
                log_likelihood = fit_data.get('log_likelihood', fit_data.get('fval', 0.0))
                if isinstance(log_likelihood, np.ndarray):
                    log_likelihood = float(np.mean(log_likelihood))
                else:
                    log_likelihood = float(log_likelihood) if log_likelihood is not None else 0.0
                
                model_name = fit_data.get('model_name', fit_data.get('agent_name', 'unknown'))
                
                # Try to get config from fit data, fallback to None
                agent_config = data.get('config', None)
                
                results.append(FitResult(
                    participant_id=str(participant_id),
                    parameters=parameters,
                    log_likelihood=log_likelihood,
                    fit_type=fit_type,
                    model_name=model_name,
                    agent_config=agent_config,
                    metadata=fit_data
                ))
        
        return results
    
    def _parse_parallel_fit_data(self, parallel_data: Dict) -> List[FitResult]:
        """Parse parallel fit data from fitting.py into FitResult objects."""
        results = []
        
        for model_name, model_results in parallel_data.items():
            individual_results = model_results.get('individual_results', [])
            # Extract the config used for this model during fitting
            agent_config = model_results.get('config', None)
            
            # Get model-level fit toolbox information
            model_fit_toolbox = model_results.get('fit_toolbox', None)
            
            if agent_config is None:
                self.logger.warning(f"No config found for model {model_name}, skipping")
                continue
            
            for subject_result in individual_results:
                if subject_result.get('failed', False):
                    continue  # Skip failed fits
                
                # Add model-level toolbox info to subject result if not present
                if 'fit_toolbox' not in subject_result and model_fit_toolbox:
                    subject_result['fit_toolbox'] = model_fit_toolbox
                
                # Detect fit type first
                fit_type = self._detect_fit_type_from_subject_result(subject_result)
                
                # Extract fitted parameters based on detected fit type
                fitted_params = self._extract_parameters_from_subject_result(subject_result, fit_type)
                
                # Get log likelihood
                log_likelihood = subject_result.get('log_likelihood', 0.0)
                
                # Create FitResult
                results.append(FitResult(
                    participant_id=str(subject_result.get('subject', 'unknown')),
                    parameters=fitted_params,
                    log_likelihood=float(log_likelihood),
                    fit_type=fit_type,
                    model_name=model_name,
                    agent_config=agent_config,  # Use the actual config from fitting
                    metadata=subject_result
                ))
        
        return results
    
    def _detect_fit_type_from_subject_result(self, subject_result: Dict) -> str:
        """Detect fit type from subject result data structure."""

        pyvbmc_indicators = ['elbo', 'log_model_evidence', 'vposteriors', 'vp', 'variational_posterior']
        
        available_keys = list(subject_result.keys())
        self.logger.debug(f"Subject result keys: {available_keys}")
        
        # Check direct indicators in subject result
        for indicator in pyvbmc_indicators:
            if indicator in subject_result:
                self.logger.debug(f"Found PyVBMC indicator: {indicator}")
                return 'pyvbmc'
        
        # Check metadata for toolbox information
        if 'fit_toolbox' in subject_result:
            toolbox = subject_result['fit_toolbox'].lower()
            self.logger.debug(f"Found fit_toolbox: {toolbox}")
            if 'vbmc' in toolbox:
                return 'pyvbmc'
            elif 'bads' in toolbox:
                return 'pybads'
        
        # Default to pybads if unsure
        self.logger.debug("No clear indicators found, defaulting to pybads")
        return 'pybads'
    
    def _extract_parameters_from_subject_result(self, subject_result: Dict, fit_type: str) -> Dict:
        """Extract parameters from subject result based on fit type."""
        # Try multiple locations for parameters
        potential_locations = [
            'fitted_params',
            'parameters', 
            'x',
            'best_params',
            'xbest'
        ]
        
        fitted_params = {}
        found_location = None
        for location in potential_locations:
            if location in subject_result:
                fitted_params = subject_result[location]
                found_location = location
                break
        
        if not fitted_params:
            # Debug: show what keys are available
            available_keys = list(subject_result.keys())
            self.logger.warning(f"No parameters found in subject result for {fit_type}. Available keys: {available_keys}")
            return {}
        
        self.logger.debug(f"Found parameters at '{found_location}' for {fit_type}: {type(fitted_params)}")
        
        # Return as-is - the _sample_parameters method will handle the format
        return fitted_params
    
    def _detect_fit_type(self, fit_data: Dict) -> str:
        """Detect whether fit data comes from PyVBMC or PyBADS."""
        # PyVBMC indicators
        pyvbmc_keys = ['vp', 'elbo', 'posterior_samples', 'variational_posterior']
        if any(key in fit_data for key in pyvbmc_keys):
            return "pyvbmc"
        
        # Check if parameters contain arrays (samples) vs scalars (point estimates)
        params = fit_data.get('parameters', fit_data.get('x', {}))
        if isinstance(params, dict):
            for value in params.values():
                if isinstance(value, np.ndarray) and len(value) > 1:
                    return "pyvbmc"
        
        return "pybads"
    
    def _extract_pyvbmc_parameters(self, fit_data: Dict) -> Dict[str, np.ndarray]:
        """Extract parameter samples from PyVBMC results."""
        parameters = {}
        
        # Try different possible locations for samples
        if 'posterior_samples' in fit_data:
            samples = fit_data['posterior_samples']
        elif 'vp' in fit_data and hasattr(fit_data['vp'], 'sample'):
            # Sample from variational posterior
            samples = fit_data['vp'].sample(1000)
        elif 'parameters' in fit_data:
            samples = fit_data['parameters']
        else:
            raise ValueError("Could not find posterior samples in PyVBMC results")
        
        # Convert to parameter dictionary
        if isinstance(samples, dict):
            parameters = {k: np.array(v) for k, v in samples.items()}
        elif isinstance(samples, np.ndarray):
            # Assume parameter names from config or metadata
            param_names = fit_data.get('parameter_names', [f'param_{i}' for i in range(samples.shape[1])])
            parameters = {name: samples[:, i] for i, name in enumerate(param_names)}
        
        return parameters
    
    def _extract_pybads_parameters(self, fit_data: Dict) -> Dict[str, float]:
        """Extract point estimates from PyBADS results."""
        parameters = {}
        
        # Try different possible locations for parameters
        if 'parameters' in fit_data:
            params = fit_data['parameters']
        elif 'x' in fit_data:
            params = fit_data['x']
        elif 'best_params' in fit_data:
            params = fit_data['best_params']
        else:
            raise ValueError("Could not find parameters in fit results")
        
        # Convert to parameter dictionary
        if isinstance(params, dict):
            parameters = {k: float(v) for k, v in params.items()}
        elif isinstance(params, (list, np.ndarray)):
            param_names = fit_data.get('parameter_names', [f'param_{i}' for i in range(len(params))])
            parameters = {name: float(params[i]) for i, name in enumerate(param_names)}
        
        return parameters
    
    def _validate_fitted_results(self):
        """Validate fitted results structure."""
        if not self.fitted_results:
            raise ValueError("No fitted results found")
        
        self.logger.info(f"Loaded {len(self.fitted_results)} fitted results")
        
        # Count fit types
        pyvbmc_count = sum(1 for r in self.fitted_results if r.fit_type == "pyvbmc")
        pybads_count = sum(1 for r in self.fitted_results if r.fit_type == "pybads")
        
        self.logger.info(f"PyVBMC results: {pyvbmc_count}, PyBADS results: {pybads_count}")
    
    def _sample_participant_equal(self, sim_idx: int) -> FitResult:
        """Sample participant with equal probability."""
        # Use simulation index to get different random state for each simulation
        base_seed = self.config.random_seed if self.config.random_seed is not None else 42
        temp_rng = np.random.default_rng(base_seed + 999999 + sim_idx)
        return temp_rng.choice(self.fitted_results)
    
    def _sample_participant_hierarchical(self, sim_idx: int) -> FitResult:
        """Sample participant with probability proportional to log likelihood."""
        log_liks = np.array([result.log_likelihood for result in self.fitted_results])
        
        # Handle numerical stability
        log_liks = log_liks - np.max(log_liks)
        weights = np.exp(log_liks * 1/5) # temperature for more dampened hierarchical sampling
        weights = weights + 1e-10  # Avoid zero weights
        weights = weights / weights.sum()
        
        # Use simulation index to get different random state for each simulation
        base_seed = self.config.random_seed if self.config.random_seed is not None else 42
        temp_rng = np.random.default_rng(base_seed + 999999 + sim_idx)
        idx = temp_rng.choice(len(self.fitted_results), p=weights)
        return self.fitted_results[idx]
    
    def _sample_parameters(self, fit_result: FitResult, sim_idx: int) -> Dict[str, float]:
        """Sample parameters from fit result (handles both PyVBMC and PyBADS)."""
        if not fit_result.parameters:
            raise ValueError(f"No parameters found for {fit_result.participant_id} ({fit_result.fit_type})")
        
        if fit_result.fit_type == "pyvbmc":
            # Sample from posterior using simulation-specific random state
            base_seed = self.config.random_seed if self.config.random_seed is not None else 42
            temp_rng = np.random.default_rng(base_seed + 888888 + sim_idx)
            sampled_params = {}
            for param_name, param_samples in fit_result.parameters.items():
                if isinstance(param_samples, np.ndarray) and len(param_samples) > 1:
                    sampled_params[param_name] = temp_rng.choice(param_samples)
                else:
                    sampled_params[param_name] = float(param_samples)
            return sampled_params
        else:
            # Use point estimates - handle different parameter formats
            sampled_params = {}
            for param_name, param_value in fit_result.parameters.items():
                try:
                    if isinstance(param_value, (list, np.ndarray)):
                        # If it's an array, take the first element or mean
                        sampled_params[param_name] = float(param_value[0]) if len(param_value) > 0 else 0.0
                    else:
                        sampled_params[param_name] = float(param_value)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Could not convert parameter {param_name}={param_value} to float: {e}")
                    sampled_params[param_name] = 0.0
            return sampled_params
    
    def _create_agent(self, parameters: Dict[str, float], fit_result: FitResult) -> RWAgent:
        """Create RW agent with given parameters using the original fitting config."""
        if fit_result.agent_config is None:
            raise ValueError(f"No agent config found for {fit_result.model_name}. "
                           "Cannot create agent without the original fitting configuration.")
        
        # Make a copy of the original config to avoid modifying the stored one
        config = fit_result.agent_config.copy() if hasattr(fit_result.agent_config, 'copy') else fit_result.agent_config
        
        # Update config with fitted parameters
        for param_name, param_value in parameters.items():
            if hasattr(config, param_name):
                setattr(config, param_name, param_value)
            else:
                self.logger.warning(f"Parameter {param_name} not found in config for {fit_result.model_name}")
        
        return RWAgent(config)
    
    
    def _create_environment_for_simulation(self, sim_idx: int) -> GoNoGoEnv:
        """Create environment with controllability schedule based on simulation index."""
        # Cycle through the 11 available controllability schedules
        c_schedule_idx = sim_idx % 11
        
        # Create config for this specific simulation
        env_config_dict = self.base_env_config.copy()
        env_config_dict['c_schedule_idx'] = c_schedule_idx
        
        env_config = GoNoGoConfig(**env_config_dict)
        return GoNoGoEnv(config=env_config)
    
    def simulate(self) -> pd.DataFrame:
        """
        Run post-fit simulations and return DataFrame compatible with DataAnalyzer.
        
        Returns:
            Combined DataFrame with all simulation results
        """
        self.logger.info(f"Starting {self.config.n_simulations} simulations with {self.config.sampling_strategy} sampling")
        self.logger.info(f"Using controllability schedules 0-10 (cycling every 11 simulations)")
        
        all_dataframes = []
        
        for sim_idx in range(self.config.n_simulations):
            if sim_idx % 100 == 0:
                self.logger.info(f"Simulation {sim_idx}/{self.config.n_simulations}")
            
            # Sample participant
            if self.config.sampling_strategy == "equal":
                fit_result = self._sample_participant_equal(sim_idx)
            elif self.config.sampling_strategy == "hierarchical":
                fit_result = self._sample_participant_hierarchical(sim_idx)
            else:
                raise ValueError(f"Unknown sampling strategy: {self.config.sampling_strategy}")
            
            # Sample parameters
            parameters = self._sample_parameters(fit_result, sim_idx)
            
            # Create agent
            agent = self._create_agent(parameters, fit_result)
            
            # Create environment with specific controllability schedule for this simulation
            environment = self._create_environment_for_simulation(sim_idx)
            c_schedule_idx = sim_idx % 11
            
            # Get seeds for this simulation
            env_seed = self.env_seeds[sim_idx]
            agent_seed = self.agent_seed_ranges[fit_result.participant_id][sim_idx]
            
            # Run simulation using SimulationRunner
            runner = SimulationRunner(environment, agent)
            df = runner.run_experiment(reset_seed=env_seed, agent_seed=agent_seed)
            
            # Add metadata columns
            df['simulation_id'] = sim_idx
            df['env_seed'] = env_seed
            df['agent_seed'] = agent_seed
            df['c_schedule_idx'] = c_schedule_idx
            df['original_participant'] = fit_result.participant_id
            df['fit_type'] = fit_result.fit_type
            df['model_name'] = fit_result.model_name
            df['sampling_strategy'] = self.config.sampling_strategy
            
            # Add sampled parameter values for tracking
            for param_name, param_value in parameters.items():
                df[f'param_{param_name}'] = param_value
            
            # Add omega tracking if available in agent history
            if hasattr(agent, 'history'):
                if 'omega_glob' in agent.history:
                    df['omega_glob'] = agent.history['omega_glob']
                if 'omega_state' in agent.history:
                    df['omega_state'] = agent.history['omega_state']
                if 'omega' in agent.history:
                    df['omega'] = agent.history['omega']
            
            all_dataframes.append(df)
        
        self.logger.info("Simulations completed")
        
        # Combine all DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get statistics about the fitted results and simulation setup."""
        stats = {
            'n_participants': len(self.fitted_results),
            'fit_types': {},
            'model_types': {},
            'log_likelihood_stats': {},
            'sampling_strategy': self.config.sampling_strategy,
            'n_simulations': self.config.n_simulations
        }
        
        # Count fit types
        for result in self.fitted_results:
            stats['fit_types'][result.fit_type] = stats['fit_types'].get(result.fit_type, 0) + 1
            stats['model_types'][result.model_name] = stats['model_types'].get(result.model_name, 0) + 1
        
        # Log likelihood statistics
        log_liks = [result.log_likelihood for result in self.fitted_results]
        stats['log_likelihood_stats'] = {
            'mean': np.mean(log_liks),
            'std': np.std(log_liks),
            'min': np.min(log_liks),
            'max': np.max(log_liks)
        }
        
        return stats
    
    def get_participant_sampling_weights(self) -> pd.DataFrame:
        """Get sampling weights for each participant (useful for hierarchical sampling)."""
        if self.config.sampling_strategy != "hierarchical":
            # Equal weights for equal sampling
            weights = np.ones(len(self.fitted_results)) / len(self.fitted_results)
        else:
            # Likelihood-based weights
            log_liks = np.array([result.log_likelihood for result in self.fitted_results])
            log_liks = log_liks - np.max(log_liks)
            weights = np.exp(log_liks)
            weights = weights + 1e-10
            weights = weights / weights.sum()
        
        return pd.DataFrame({
            'participant_id': [r.participant_id for r in self.fitted_results],
            'log_likelihood': [r.log_likelihood for r in self.fitted_results],
            'sampling_weight': weights,
            'fit_type': [r.fit_type for r in self.fitted_results],
            'model_name': [r.model_name for r in self.fitted_results]
        })


def simulate_from_fits(fitted_results_path: Union[str, Path],
                      n_simulations: int = 1000,
                      sampling_strategy: str = "equal",
                      random_seed: Optional[int] = None,
                      environment_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function for running post-fit simulations.
    
    Args:
        fitted_results_path: Path to fitted results pickle file
        n_simulations: Number of simulations to run
        sampling_strategy: "equal" or "hierarchical"
        random_seed: Random seed for reproducibility
        environment_config: Environment configuration dict
    
    Returns:
        DataFrame compatible with DataAnalyzer
    """
    config = SimulationConfig(
        n_simulations=n_simulations,
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
        environment_config=environment_config
    )
    
    simulator = PostFitSimulator(fitted_results_path, config=config)
    return simulator.simulate()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run post-fit simulations")
    parser.add_argument("--fitted_results", type=str, required=True, help="Path to fitted results")
    parser.add_argument("--n_simulations", type=int, default=1000, help="Number of simulations")
    parser.add_argument("--sampling", type=str, default="equal", choices=["equal", "hierarchical"])
    parser.add_argument("--output", type=str, help="Output path for simulation results")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Run simulations
    results_df = simulate_from_fits(
        args.fitted_results,
        n_simulations=args.n_simulations,
        sampling_strategy=args.sampling,
        random_seed=args.seed
    )
    
    # Save results if output path provided
    if args.output:
        results_df.to_pickle(args.output)
        print(f"Results saved to {args.output}")
    else:
        print(f"Generated {len(results_df)} simulation trials from {results_df['simulation_id'].nunique()} simulations")
        print(f"Participants sampled: {results_df['original_participant'].nunique()}")
        
        # Quick analysis using DataAnalyzer
        from data_analyzer import DataAnalyzer
        analyzer = DataAnalyzer(results_df, is_subject_data=False)
        analyzer.plot_learning_curves(title="Post-fit Simulation Results")
        
        # Show controllability schedule distribution
        print(f"Controllability schedule distribution:")
        print(results_df['c_schedule_idx'].value_counts().sort_index())


# Convenience functions for notebook usage  
def list_available_fit_results():
    """
    Enhanced listing of available fit results with fit toolbox information.
    
    Returns:
        List of (timestamp, directory_path, has_results, fit_info) tuples
    """
    base_results = list_all_parallel_fit_results()
    enhanced_results = []
    
    for timestamp, dir_path, has_results in base_results:
        fit_info = {}
        if has_results:
            try:
                # Load results to get fit toolbox info
                results, _ = load_specific_parallel_fit_results(timestamp)
                if results:
                    # Extract fit toolbox info from each model
                    model_info = {}
                    for model_name, model_data in results.items():
                        fit_toolbox = model_data.get('fit_toolbox', 'unknown')
                        n_subjects = model_data.get('n_successful', 0)
                        model_info[model_name] = {
                            'fit_toolbox': fit_toolbox,
                            'n_subjects': n_subjects
                        }
                    fit_info = {
                        'n_models': len(results),
                        'models': model_info
                    }
            except Exception as e:
                fit_info = {'error': str(e)}
        
        enhanced_results.append((timestamp, dir_path, has_results, fit_info))
    
    # Print enhanced info
    print(f"Found {len(enhanced_results)} parallel fitting directories:")
    for i, (timestamp, path, has_results, fit_info) in enumerate(enhanced_results):
        if has_results and 'models' in fit_info:
            # Count PyVBMC vs PyBADS models
            toolbox_counts = {}
            for model_name, info in fit_info['models'].items():
                toolbox = info['fit_toolbox']
                toolbox_counts[toolbox] = toolbox_counts.get(toolbox, 0) + 1
            
            toolbox_str = ", ".join([f"{count} {toolbox}" for toolbox, count in toolbox_counts.items()])
            print(f"  {i+1}. {timestamp} - ✓ {fit_info['n_models']} models ({toolbox_str})")
        elif has_results:
            print(f"  {i+1}. {timestamp} - ✓ Has results (info unavailable)")
        else:
            print(f"  {i+1}. {timestamp} - ✗ No results file")
    
    return enhanced_results


def create_simulator_from_latest(config: Optional[SimulationConfig] = None) -> PostFitSimulator:
    """
    Create PostFitSimulator from the most recent parallel fit results.
    
    Args:
        config: Simulation configuration
        
    Returns:
        PostFitSimulator instance
    """
    results, _ = find_latest_parallel_fit_results()
    if results is None:
        raise ValueError("No parallel fit results found")
    
    return PostFitSimulator(results, config=config)


def create_simulator_from_timestamp(timestamp: str, 
                                   config: Optional[SimulationConfig] = None) -> PostFitSimulator:
    """
    Create PostFitSimulator from specific timestamp.
    
    Args:
        timestamp: Timestamp string (e.g., "20241226_143052")
        config: Simulation configuration
        
    Returns:
        PostFitSimulator instance
    """
    return PostFitSimulator(timestamp, config=config)


def get_fit_summary(timestamp: str) -> Dict:
    """
    Get summary statistics for fitted results from a specific timestamp.
    
    Args:
        timestamp: Timestamp string
        
    Returns:
        Dictionary with summary statistics
    """
    results, _ = load_specific_parallel_fit_results(timestamp)
    if results is None:
        raise ValueError(f"Could not load results for timestamp: {timestamp}")
    
    summary = {
        'timestamp': timestamp,
        'n_models': len(results),
        'models': list(results.keys()),
        'model_details': {}
    }
    
    for model_name, model_data in results.items():
        n_subjects = model_data.get('n_subjects', 0)
        n_successful = model_data.get('n_successful', 0)
        fit_toolbox = model_data.get('fit_toolbox', 'unknown')
        
        summary['model_details'][model_name] = {
            'n_subjects': n_subjects,
            'n_successful': n_successful,
            'success_rate': n_successful / n_subjects if n_subjects > 0 else 0,
            'fit_toolbox': fit_toolbox
        }
    
    return summary


def simulate_from_latest(n_simulations: int = 1000,
                        sampling_strategy: str = "equal",
                        random_seed: Optional[int] = None,
                        environment_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function to simulate from the latest parallel fit results.
    
    Args:
        n_simulations: Number of simulations to run
        sampling_strategy: "equal" or "hierarchical"
        random_seed: Random seed for reproducibility
        environment_config: Environment configuration dict
        
    Returns:
        DataFrame compatible with DataAnalyzer
    """
    config = SimulationConfig(
        n_simulations=n_simulations,
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
        environment_config=environment_config
    )
    
    simulator = create_simulator_from_latest(config)
    return simulator.simulate()


def simulate_from_timestamp(timestamp: str,
                           n_simulations: int = 1000,
                           sampling_strategy: str = "equal",
                           random_seed: Optional[int] = None,
                           environment_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function to simulate from specific timestamp results.
    
    Args:
        timestamp: Timestamp string (e.g., "20241226_143052")
        n_simulations: Number of simulations to run
        sampling_strategy: "equal" or "hierarchical"
        random_seed: Random seed for reproducibility
        environment_config: Environment configuration dict
        
    Returns:
        DataFrame compatible with DataAnalyzer
    """
    config = SimulationConfig(
        n_simulations=n_simulations,
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
        environment_config=environment_config
    )
    
    simulator = create_simulator_from_timestamp(timestamp, config)
    return simulator.simulate()


def simulate_each_model_from_timestamp(timestamp: str,
                                     n_simulations: int = 1000,
                                     sampling_strategy: str = "equal",
                                     random_seed: Optional[int] = None,
                                     environment_config: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """
    Simulate each model separately from specific timestamp results.
    
    Args:
        timestamp: Timestamp string (e.g., "20241226_143052")
        n_simulations: Number of simulations to run per model
        sampling_strategy: "equal" or "hierarchical"
        random_seed: Random seed for reproducibility
        environment_config: Environment configuration dict
        
    Returns:
        Dict mapping model_name -> DataFrame for each model
    """
    # Load results to get model names
    results, _ = load_specific_parallel_fit_results(timestamp)
    if results is None:
        raise ValueError(f"Could not load results for timestamp: {timestamp}")
    
    model_simulations = {}
    
    for model_name in results.keys():
        print(f"Simulating model: {model_name}")
        
        # Create a filtered dataset with only this model's results
        model_specific_results = {model_name: results[model_name]}
        
        config = SimulationConfig(
            n_simulations=n_simulations,
            sampling_strategy=sampling_strategy,
            random_seed=random_seed,
            environment_config=environment_config
        )
        
        simulator = PostFitSimulator(model_specific_results, config=config)
        model_df = simulator.simulate()
        model_simulations[model_name] = model_df
    
    return model_simulations


def analyze_model_simulations(model_simulations: Dict[str, pd.DataFrame], 
                            subject_data: Optional[pd.DataFrame] = None):
    """
    Analyze simulation results for each model separately.
    
    Args:
        model_simulations: Dict mapping model_name -> simulation DataFrame
        subject_data: Optional subject data for comparison
    """
    from data_analyzer import DataAnalyzer
    
    # Plot human data first if provided
    if subject_data is not None:
        print("=== HUMAN LEARNING CURVES ===")
        human_analyzer = DataAnalyzer(subject_data, is_subject_data=True)
        human_analyzer.plot_learning_curves(title="Human Learning Curves")
        print("\n" + "="*50 + "\n")
    
    # Plot each model's learning curves
    for model_name, sim_df in model_simulations.items():
        print(f"=== MODEL: {model_name.upper()} ===")
        
        # Basic info
        n_sims = sim_df['simulation_id'].nunique()
        n_participants = sim_df['original_participant'].nunique()
        fit_toolbox = sim_df['fit_type'].iloc[0] if 'fit_type' in sim_df.columns else 'unknown'
        
        print(f"Simulations: {n_sims}, Participants: {n_participants}, Fit: {fit_toolbox}")
        
        # Learning curves for this model
        analyzer = DataAnalyzer(sim_df, is_subject_data=False)
        analyzer.plot_learning_curves(title=f"Learning Curves: {model_name}")
        
        # WSLS analysis
        analyzer.plot_wsls(title=f"WSLS Analysis: {model_name}")
        
        print("\n" + "="*50 + "\n")