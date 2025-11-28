# %% fitting.py
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from agents.rw_agent import RWAgent, RWConfig
from agents.robot_dataset import RobotDataset
from agents.subject_data import SubjectData
from datetime import datetime
import time
import pickle
import os
import copy
from collections import defaultdict
import glob
from pathlib import Path
import dill
try:
    from pybads import BADS
    PYBADS_AVAILABLE = True
except ImportError:
    PYBADS_AVAILABLE = False
    print("PyBADS not available. Install with: pip install pybads")

try:
    from pyvbmc import VBMC
    import pyvbmc.priors as priors
    PYVBMC_AVAILABLE = True
except ImportError:
    PYVBMC_AVAILABLE = False
    print("PyVBMC not available. Install with: pip install pyvbmc")
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("TQDM not available. Install with: pip install tqdm")

try:
    import submitit
    SUBMITIT_AVAILABLE = True
except ImportError:
    SUBMITIT_AVAILABLE = False
    print("Submitit not available. Install with: pip install submitit")


# %


class AgentFitter:
    """Class for fitting agent parameters to behavioral data"""

    def __init__(self, agent_class=RWAgent, config_class=RWConfig,
                 fit_toolbox='pybads', vbmc_init_with_bads=True, slurm_fitting=False):
        """
        Initialize AgentFitter

        Args:
            agent_class: Agent class to fit
            config_class: Configuration class for agent
            fit_toolbox: 'pybads' or 'pyvbmc'
            vbmc_init_with_bads: Whether to initialize VBMC with BADS (recommended)
        """

        self.fit_toolbox = fit_toolbox.lower()
        self.vbmc_init_with_bads = vbmc_init_with_bads

        if self.fit_toolbox == 'pybads' and not PYBADS_AVAILABLE:
            raise ImportError(
                "PyBADS is required for fitting. Install with: pip install pybads")

        if self.fit_toolbox == 'pyvbmc' and not PYVBMC_AVAILABLE:
            raise ImportError(
                "PyVBMC is required for fitting. Install with: pip install pyvbmc")

        if self.fit_toolbox == 'pyvbmc' and self.vbmc_init_with_bads and not PYBADS_AVAILABLE:
            raise ImportError(
                "PyBADS is required for VBMC initialization. Install with: pip install pybads")

        if slurm_fitting and not (SUBMITIT_AVAILABLE):
            raise ImportError(
                "Submitit and tqdm is required for SLURM fitting. Install with: pip install submitit")

        print(f"Using {self.fit_toolbox.upper()} for optimization")
        if self.fit_toolbox == 'pyvbmc' and self.vbmc_init_with_bads:
            print("Will initialize VBMC with PyBADS")

        self.agent_class = agent_class
        self.config_class = config_class

    def create_subject_objective_function(self, data: SubjectData,
                                          config: RWConfig) -> Callable:
        """
        Create objective function for optimization
        Returns negative log-likelihood (to minimize) for PyBADS
        Returns log-likelihood (to maximize) for PyVBMC
        """
        def objective(params: np.ndarray) -> float:
            # Map parameters back to config
            param_dict = {}
            for i, param_name in enumerate(config.fit_parameters):
                param_dict[param_name] = params[i]

            # Create agent with these parameters
            temp_config = RWConfig(**{**config.__dict__, **param_dict})
            agent = self.agent_class(temp_config)

            # Compute log-likelihood
            try:
                log_likelihood = agent.compute_log_likelihood(data)
                if self.fit_toolbox == 'pybads':
                    return -log_likelihood  # Minimize negative log-likelihood
                else:
                    return log_likelihood   # Maximize log-likelihood for VBMC
            except Exception as e:
                # Return appropriate value if computation fails
                if self.fit_toolbox == 'pybads':
                    return 1e6
                else:
                    return -1e6

        return objective

    def _setup_priors(self, config: RWConfig) -> List:
        """Setup priors for PyVBMC"""
        param_names = config.fit_parameters
        prior_array = []

        for param_name in param_names:
            prior_shape = getattr(config, 'prior_shapes', {}).get(
                param_name, 'UniformBox')
            bounds = config.parameter_bounds[param_name]
            plausible_bounds = config.plausible_bounds[param_name]

            if prior_shape == 'UniformBox':
                prior_array.append(priors.UniformBox(bounds[0], bounds[1]))
            elif prior_shape == 'Trapezoidal':
                prior_array.append(priors.Trapezoidal(bounds[0], plausible_bounds[0],
                                                      plausible_bounds[1], bounds[1]))
            elif prior_shape == 'SmoothBox':
                prior_array.append(priors.SmoothBox(
                    plausible_bounds[0], plausible_bounds[1], 0.8))
            else:
                # Default to UniformBox
                prior_array.append(priors.UniformBox(bounds[0], bounds[1]))

        return prior_array

    def _compute_population_stats(self, all_results: List[Dict],
                                  config: RWConfig) -> Dict:
        """Compute population-level parameter estimates"""

        param_names = config.fit_parameters
        population_params = {}

        # Extract parameter values across subjects
        param_values = {name: [] for name in param_names}
        log_likelihoods = []
        aics = []
        bics = []

        for result in all_results:
            for name in param_names:
                param_values[name].append(result['fitted_params'][name])
            log_likelihoods.append(result['log_likelihood'])
            aics.append(result['aic'])
            bics.append(result['bic'])

        # Compute statistics for each parameter
        for name in param_names:
            values = np.array(param_values[name])
            population_params[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'individual_values': values
            }

        # Add model evidence for VBMC
        model_evidence_stats = {}
        if self.fit_toolbox == 'pyvbmc':
            log_evidences = [r.get('log_model_evidence', np.nan)
                             for r in all_results]
            log_evidences = [e for e in log_evidences if not np.isnan(e)]
            if log_evidences:
                model_evidence_stats = {
                    'mean_per_subject': np.mean(log_evidences),
                    'total': np.sum(log_evidences),
                    'individual': log_evidences
                }

        return {
            'parameters': population_params,
            'log_likelihood': {
                'mean': np.mean(log_likelihoods),
                'std': np.std(log_likelihoods),
                'total': np.sum(log_likelihoods)
            },
            'aic': {
                'mean': np.mean(aics),
                'total': np.sum(aics)
            },
            'bic': {
                'mean': np.mean(bics),
                'total': np.sum(bics)
            },
            'model_evidence': model_evidence_stats,
            'convergence_rate': np.mean([r['convergence_rate'] for r in all_results])
        }

    def fit_multiple_subjects(self, data: RobotDataset, config: RWConfig,
                              n_starts: int = 5,
                              bads_options: Dict = None,
                              vbmc_options: Dict = None,
                              vbmc_bads_init_options: Dict = None) -> Dict:
        """
        Fit multiple subjects using either PyBADS or PyVBMC

        Args:
            data: RobotDataset containing subject data
            config: Agent configuration
            n_starts: Number of random starts (for PyBADS) or restarts (for PyVBMC)
            bads_options: Options for PyBADS
            vbmc_options: Options for PyVBMC
            vbmc_bads_init_options: Options for PyBADS initialization when using PyVBMC
        """
        # Results storage
        all_subject_results = []

        for subject_idx, (subject, subject_data) in enumerate(data):
            print(
                f"Fitting subject {subject} with {self.fit_toolbox.upper()}...")

            subject_result = self._fit_single_subject(
                subject_data, config, n_starts,
                bads_options, vbmc_options, vbmc_bads_init_options
            )
            subject_result['subject'] = subject
            subject_result['subject_idx'] = subject_idx
            all_subject_results.append(subject_result)

        population_stats = self._compute_population_stats(
            all_subject_results, config=config)

        return {
            "individual_results": all_subject_results,
            "population_statistics": population_stats,
            "n_subjects": len(all_subject_results),
            "fit_toolbox": self.fit_toolbox
        }

    def _fit_single_subject(self, subject_data: SubjectData, config: RWConfig,
                            n_starts: int,
                            bads_options: Dict = None,
                            vbmc_options: Dict = None,
                            vbmc_bads_init_options: Dict = None) -> Dict:
        """Fit a single subject using PyBADS or PyVBMC"""

        param_names = config.fit_parameters
        initial_params = [getattr(config, name) for name in param_names]
        lower_bounds = [config.parameter_bounds[name][0]
                        for name in param_names]
        upper_bounds = [config.parameter_bounds[name][1]
                        for name in param_names]
        plausible_l_bound = [config.plausible_bounds[name][0]
                             for name in param_names]
        plausible_u_bound = [config.plausible_bounds[name][1]
                             for name in param_names]

        # Set default options
        if bads_options is None:
            bads_options = {
                'max_fun_evals': 300,        # Reduce from default ~1000 for faster fitting
                'tol_mesh': 1e-5,           # Slightly looser mesh tolerance
                'tol_fun': 1e-4,            # Faster convergence criterion
                'uncertainty_handling': 1   # Better handling of noisy objectives
            }
        if vbmc_options is None:
            vbmc_options = {}
        if vbmc_bads_init_options is None:
            vbmc_bads_init_options = {
                'max_fun_evals': 100,        # Fewer evals for BADS initialization
                'tol_fun': 1e-3             # Looser tolerance for initialization
            }

        objective = self.create_subject_objective_function(
            subject_data, config)

        if self.fit_toolbox == 'pybads':
            return self._fit_with_bads(
                objective,
                param_names,
                initial_params,
                lower_bounds,
                upper_bounds,
                plausible_l_bound,
                plausible_u_bound,
                n_starts,
                bads_options,
                subject_data
            )
        else:  # pyvbmc
            return self._fit_with_vbmc(
                objective,
                param_names,
                initial_params,
                lower_bounds,
                upper_bounds,
                plausible_l_bound,
                plausible_u_bound,
                n_starts,
                config,
                vbmc_options,
                vbmc_bads_init_options,
                subject_data
            )

    def _fit_with_bads(self, objective, param_names, initial_params,
                       lower_bounds, upper_bounds, plausible_l_bound, plausible_u_bound,
                       n_starts, bads_options, subject_data):
        """Fit using PyBADS"""

        best_result = None
        best_nll = float('inf')
        all_results = []

        for start_idx in range(n_starts):
            print(f"Starting run {start_idx}")

            if start_idx == 0:
                x0 = np.array(initial_params)
            else:
                x0 = plausible_l_bound + np.random.uniform(size=len(param_names)) * (
                    np.array(plausible_u_bound) - np.array(plausible_l_bound))

            try:
                bads = BADS(objective, x0, lower_bounds, upper_bounds,
                            plausible_l_bound, plausible_u_bound, options=bads_options)
                result = bads.optimize()

                all_results.append({
                    'start_idx': start_idx,
                    'x': result['x'],
                    'fval': result['fval'],
                    'success': True,
                    'message': 'Optimization completed'
                })

                if result['fval'] < best_nll:
                    best_nll = result['fval']
                    best_result = result

            except Exception as e:
                print(f"Exception occurred with PyBADS: {e}")
                all_results.append({
                    'start_idx': start_idx,
                    'x': x0,
                    'fval': float('inf'),
                    'success': False,
                    'message': str(e)
                })

        if best_result is None:
            raise RuntimeError("All optimization attempts failed for subject")

        fitted_params = {}
        for i, param_name in enumerate(param_names):
            fitted_params[param_name] = best_result['x'][i]

        n_data_points = 320

        return {
            'fitted_params': fitted_params,
            'negative_log_likelihood': best_nll,
            'log_likelihood': -best_nll,
            'aic': 2 * len(param_names) + 2 * best_nll,
            'bic': len(param_names) * np.log(n_data_points) + 2 * best_nll,
            'best_result': best_result,
            'all_results': all_results,
            'n_data_points': n_data_points,
            'n_parameters': len(param_names),
            'convergence_rate': sum(1 for r in all_results if r['success']) / len(all_results)
        }

    def _fit_with_vbmc(self, objective, param_names, initial_params,
                       lower_bounds, upper_bounds, plausible_l_bound, plausible_u_bound,
                       n_starts, config, vbmc_options, vbmc_bads_init_options, subject_data):
        """Fit using PyVBMC (with optional PyBADS initialization)"""

        # Setup priors for VBMC
        prior_array = self._setup_priors(config)

        best_result = None
        best_elbo = float('-inf')
        all_results = []
        vposteriors = []

        for start_idx in range(n_starts):
            if start_idx == 0:
                x0 = np.array(initial_params)
            else:
                x0 = plausible_l_bound + np.random.uniform(size=len(param_names)) * (
                    np.array(plausible_u_bound) - np.array(plausible_l_bound))

            try:
                # Initialize with BADS if requested
                if self.vbmc_init_with_bads:
                    # Use negative objective for BADS (since BADS minimizes)
                    def bads_objective(params): return -objective(params)
                    bads = BADS(bads_objective, x0, lower_bounds, upper_bounds,
                                plausible_l_bound, plausible_u_bound, options=vbmc_bads_init_options)
                    bads_result = bads.optimize()
                    x0_vbmc = bads_result['x']
                else:
                    x0_vbmc = x0

                # Run VBMC
                vbmc = VBMC(objective, x0_vbmc, lower_bounds, upper_bounds,
                            plausible_l_bound, plausible_u_bound,
                            options=vbmc_options, prior=prior_array)
                vp, result = vbmc.optimize()

                vposteriors.append(vp)

                # Get parameter estimates (posterior means)
                print(f"Debug: hasattr(vp, 'moments') = {hasattr(vp, 'moments')}")
                if hasattr(vp, 'moments'):
                    moments = vp.moments()
                
                    fitted_x = moments.flatten()

                else:
                    # Fallback: sample from posterior and take mean
                    samples = vp.sample(1000)
                    fitted_x = np.mean(samples, axis=0)
                                
                elbo = result['elbo']
                elbo_sd = result.get('elbo_sd', np.nan)
                log_evidence = result.get('lnZ', np.nan)
                
                all_results.append({
                    'start_idx': start_idx,
                    'x': fitted_x,
                    'elbo': elbo,
                    'elbo_sd': elbo_sd,
                    'log_evidence': log_evidence,
                    'success': True,
                    'message': 'Optimization completed',
                    'vp': vp
                })
                
                if elbo > best_elbo:
                    best_elbo = elbo
                    best_result = {
                        'x': fitted_x,
                        'elbo': elbo,
                        'elbo_sd': elbo_sd,
                        'log_evidence': log_evidence,
                        'vp': vp
                    }
                    print(f"Debug: New best result, x = {fitted_x}")


            except Exception as e:
                all_results.append({
                    'start_idx': start_idx,
                    'x': x0,
                    'elbo': float('-inf'),
                    'elbo_sd': np.nan,
                    'log_evidence': np.nan,
                    'success': False,
                    'message': str(e),
                    'vp': None
                })

        if best_result is None:
            raise RuntimeError(
                "All VBMC optimization attempts failed for subject")
    
        fitted_params = {}
        for i, param_name in enumerate(param_names):
            fitted_params[param_name] = best_result['x'][i]

        n_data_points = 320

        # Convert ELBO to negative log-likelihood for compatibility
        nll = -best_result['elbo']

        return {
            'fitted_params': fitted_params,
            'negative_log_likelihood': nll,
            'log_likelihood': best_result['elbo'],
            'elbo': best_result['elbo'],
            'elbo_sd': best_result['elbo_sd'],
            'log_model_evidence': best_result['log_evidence'],
            'aic': 2 * len(param_names) + 2 * nll,
            'bic': len(param_names) * np.log(n_data_points) + 2 * nll,
            'best_result': best_result,
            'all_results': all_results,
            'vposteriors': vposteriors,
            'n_data_points': n_data_points,
            'n_parameters': len(param_names),
            'convergence_rate': sum(1 for r in all_results if r['success']) / len(all_results)
        }

def clean_bads_results(results_dict):
    """
    Simple cleaning - just remove the PyBADS result objects that contain function references
    """
    # Make a copy so we don't modify the original
    cleaned_dict = copy.deepcopy(results_dict)

    # Clean individual subject results
    for result in cleaned_dict['individual_results']:
        # Remove the full PyBADS result objects, keep only essential info
        if 'best_result' in result:
            best_result = result['best_result']
            # Keep only the essential data from PyBADS result
            result['best_result_summary'] = {
                'x': best_result.get('x', None),
                'fval': best_result.get('fval', None),
                'iterations': best_result.get('iterations', None),
                'funccount': best_result.get('funccount', None),
                'message': best_result.get('message', None)
            }
            # Remove the original (potentially non-picklable) result
            del result['best_result']

        if 'all_results' in result:
            # Keep only essential info from all results
            cleaned_all_results = []
            for res in result['all_results']:
                if isinstance(res, dict):
                    cleaned_res = {
                        'start_idx': res.get('start_idx'),
                        'x': res.get('x'),
                        'fval': res.get('fval'),
                        'success': res.get('success'),
                        'message': res.get('message')
                    }
                    cleaned_all_results.append(cleaned_res)
            result['all_results'] = cleaned_all_results

    return cleaned_dict

def safe_dill_results(results_dict, fit_folder, config_name):
    """
    Save results using dill (more robust for complex objects like VBMC posteriors)
    """
    results_file = os.path.join(fit_folder, f"{config_name}_results.pkl")

    try:
        # Use dill for more robust serialization
        with open(results_file, 'wb') as f:
            dill.dump(results_dict, f)
        print(f"Results saved to: {results_file}")

    except (AttributeError, TypeError) as e:
        print(f"Dill failed with original results: {e}")
        print("Cleaning objects and trying again...")

        # Clean and try again
        cleaned_results = clean_bads_results(results_dict)
        with open(results_file, 'wb') as f:
            dill.dump(cleaned_results, f)
        print(f"Cleaned results saved to: {results_file}")

        # Also save the original error for debugging
        error_file = os.path.join(fit_folder, f"{config_name}_dill_error.txt")
        with open(error_file, 'w') as f:
            f.write(f"Original dill error: {str(e)}\n")
            f.write("Had to clean result objects to make serializable.\n")

    return results_file

# Note: Removed redundant VBMC-specific cleaning functions since dill handles complex objects better

# Convenience function for quick fitting
def fit_all_models_parallel(models_to_fit, data, toolbox="pybads", 
                           fit_with_slurm=True, n_starts=10, 
                           bads_options=None, vbmc_options=None, vbmc_bads_init_options=None,
                           slurm_config=None, developmental_subjects=None, vbmc_init_with_bads=True):
    """
    Fit multiple models in parallel by submitting all jobs at once
    and collecting results as they complete.
    """
    data = data.filter_subjects(
        developmental_subjects) if developmental_subjects else data
    
    if not fit_with_slurm:
        print("Falling back to sequential fitting.")
        for i, model in enumerate(models_to_fit):
            print(f"\nFitting model {i+1}/{len(models_to_fit)}: {model.name}")
    
            # Fall back to sequential fitting 
            res = fit_model_sequentially(
                data=data,
                agent_config=model,
                fit_toolbox=toolbox,
                fit_with_slurm=False,
                development_subjects=developmental_subjects,
                n_starts=n_starts,
                bads_options=bads_options,
                vbmc_options=vbmc_options,
                vbmc_bads_init_options=vbmc_bads_init_options,
                vbmc_init_with_bads=vbmc_init_with_bads
            )
    
    # Set up SLURM configuration
    if slurm_config is None:
        if toolbox == "pybads":
            slurm_config = {
                'mem': 2000,              # Reduce to 2GB (sufficient for single subject)
                'time': 45,               # Reduce to 45min with optimized BADS
                'partition': "batch",
                'cpus_per_task': 2,
                'signal_delay_s': 120     # Reduce cleanup time
            }
        else:
            slurm_config = {
                'mem': 4000,              # Keep at 4GB
                'time': 180,              # Increase to 3 hours for VBMC
                'partition': "batch",
                'cpus_per_task': 2,
                'signal_delay_s': 300     # More time for cleanup
            }

    # Create master results directory
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_fit_folder = os.path.join("model_fits", 'logs', 'parallel_fitting', now)
    os.makedirs(master_fit_folder, exist_ok=True)
    
    # Set up executor
    executor = submitit.SlurmExecutor(folder=master_fit_folder, max_num_timeout=5)
    executor.update_parameters(**slurm_config, wckey="parallel_model_fitting")
    
    # Submit all jobs for all models
    all_jobs = []
    job_metadata = {}  # job_id -> (model_name, subject_id, subject_idx)
    
    print(f"Submitting jobs for {len(models_to_fit)} models with {len(data)} subjects each...")
    
    with tqdm(total=len(models_to_fit) * len(data), desc="Submitting jobs") as pbar:
        for model in models_to_fit:
            # Create fitter for this model
            from fitting import AgentFitter
            fitter = AgentFitter(fit_toolbox=toolbox, vbmc_init_with_bads=vbmc_init_with_bads, slurm_fitting=True)
            
            for subject_idx, (subject_id, subject_data) in enumerate(data):
                # Submit job for this model-subject combination
                job = executor.submit(
                    fitter._fit_single_subject, 
                    subject_data, 
                    model,
                    n_starts,
                    bads_options,
                    vbmc_options,  # vbmc_options
                    vbmc_bads_init_options   # vbmc_bads_init_options
                )
                
                all_jobs.append(job)
                job_metadata[job.job_id] = (model.name, subject_id, subject_idx)
                
                time.sleep(0.1)  # Small delay to avoid overwhelming scheduler
                pbar.update(1)
    
    print(f"Submitted {len(all_jobs)} total jobs to SLURM")
    time.sleep(5)
    
    # Track results by model
    model_results = defaultdict(list)
    completed_jobs = set()
    total_jobs = len(all_jobs)
    
    # Wait for jobs to complete and collect results
    with tqdm(total=total_jobs, desc="Collecting results") as pbar:
        while len(completed_jobs) < total_jobs:
            time.sleep(2.0)  # Check every 2 seconds
            
            for job in all_jobs:
                if job.job_id not in completed_jobs and job.done():
                    model_name, subject_id, subject_idx = job_metadata[job.job_id]
                    
                    try:
                        result = job.result()
                        result['subject'] = subject_id
                        result['subject_idx'] = subject_idx
                        result['model_name'] = model_name
                        
                        model_results[model_name].append((subject_idx, result))
                        pbar.set_postfix_str(f'Latest: {model_name[:15]}... - {subject_id}')
                        
                    except Exception as e:
                        print(f"Job failed for {model_name} - {subject_id}: {e}")
                        # Create failed result
                        failed_result = {
                            'subject': subject_id,
                            'subject_idx': subject_idx,
                            'model_name': model_name,
                            'fitted_params': {},
                            'negative_log_likelihood': float('inf'),
                            'log_likelihood': float('-inf'),
                            'aic': float('inf'),
                            'bic': float('inf'),
                            'convergence_rate': 0.0,
                            'failed': True,
                            'error_message': str(e)
                        }
                        model_results[model_name].append((subject_idx, failed_result))
                        pbar.set_postfix_str(f'Failed: {model_name[:15]}... - {subject_id}')
                    
                    completed_jobs.add(job.job_id)
                    pbar.update(1)
    
    # Organize results and compute population statistics
    final_results = {}
    
    print("\nComputing population statistics for each model...")
    for model in models_to_fit:
        model_name = model.name
        
        if model_name not in model_results:
            print(f"Warning: No results found for model {model_name}")
            continue
            
        # Sort results by subject index to maintain order
        subject_results = [result for _, result in sorted(model_results[model_name])]
        
        # Filter successful results for population statistics
        successful_results = [r for r in subject_results if not r.get('failed', False)]
        
        if successful_results:
            # Create a temporary fitter to compute population stats
            fitter = AgentFitter(fit_toolbox=toolbox, vbmc_init_with_bads=True)
            population_stats = fitter._compute_population_stats(successful_results, model)
        else:
            population_stats = {}
            print(f"Warning: All subjects failed for model {model_name}")
        
        final_results[model_name] = {
            "individual_results": subject_results,
            "population_statistics": population_stats,
            "n_subjects": len(subject_results),
            "n_successful": len(successful_results),
            "fit_toolbox": toolbox,
            "config": model,
            "fit_folder": master_fit_folder
        }
        
        print(f"✓ {model_name}: {len(successful_results)}/{len(subject_results)} subjects successful")
    
    # Save all results to a master file
    master_results_file = os.path.join(master_fit_folder, "all_models_results.pkl")

    for model_results in final_results.values():
        for result in model_results['individual_results']: 
            if 'best_result' in result and 'fun' in result['best_result']:
                del result['best_result']['fun']

    with open(master_results_file, 'wb') as f:
        dill.dump(final_results, f)
    print(f"\nAll results saved to: {master_results_file}")
    print(f"Fitted {len(final_results)} models successfully!")
    
    return final_results

def fit_model_sequentially(data: RobotDataset,
                      n_states: int = 4, n_actions: int = 2,
                      agent_config: Optional[RWConfig] = None,
                      fit_toolbox: str = 'pybads', vbmc_init_with_bads: bool = True,
                      development_subjects: Optional[List[str]] = None,
                      fit_with_slurm: bool = False,
                      slurm_config: Optional[Dict] = None,
                      **kwargs) -> Dict:
    """
    Convenience function for fitting an agent to behavioral data sequentially (sequential, as in submit models -> wait for result, submit next model, etc.)
    Args:
        data: RobotDataset containing subject data
        n_states: Number of possible states
        n_actions: Number of possible actions
        agent_config: Agent configuration (uses defaults if None)
        fit_toolbox: 'pybads' or 'pyvbmc'
        fit_with_slurm: Whether to use SLURM for parallel processing
        slurm_config: SLURM configuration dict (optional)
        **kwargs: Additional arguments passed to fit_multiple_subjects
    Returns:
        Fitting results dictionary
    """
    config = agent_config if agent_config is not None else RWConfig()
    fitter = AgentFitter(fit_toolbox=fit_toolbox,
                         vbmc_init_with_bads=vbmc_init_with_bads)
    data = data.filter_subjects(
        development_subjects) if development_subjects else data

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting fitting at {now} with {len(data)} subjects")

    if not fit_with_slurm:
        # Use local fitting
        return fitter.fit_multiple_subjects(data, config, **kwargs)

    # Use SLURM fitting
    fit_folder = os.path.join("model_fits", 'logs', config.name, now)
    os.makedirs(fit_folder, exist_ok=True)

    # Set default SLURM config
    if slurm_config is None:
        slurm_config = {
            'mem': 2000,              # Reduce to 2GB (sufficient for single subject)
            'time': 45,               # Reduce to 45min with optimized BADS
            'partition': "batch",
            'cpus_per_task': 2,
            'signal_delay_s': 120     # Reduce cleanup time
        }

    executor = submitit.SlurmExecutor(folder=fit_folder, max_num_timeout=5)
    executor.update_parameters(**slurm_config, wckey=f"{config.name}")

    jobs = []
    subject_data_list = []

    # Submit individual subject fitting jobs
    with tqdm(total=len(data), desc=f"Submitting jobs for {config.name}") as pbar:
        for subject_idx, (subject_id, subject_data) in enumerate(data):
            # Store subject data for later
            subject_data_list.append((subject_id, subject_data))

            # Submit single subject fitting job
            job = executor.submit(fitter._fit_single_subject, subject_data, config,
                                kwargs.get('n_starts', 5),
                                kwargs.get('bads_options', None),
                                kwargs.get('vbmc_options', None),
                                kwargs.get('vbmc_bads_init_options', None))
            jobs.append(job)
            time.sleep(0.25)
            pbar.update(1)

    print(f"Submitted {len(jobs)} jobs to SLURM")
    time.sleep(5)

    # Wait for all jobs to complete and collect results
    all_subject_results = []
    jobs_ids = {job.job_id: idx for idx, job in enumerate(jobs)}
    returned_ids = []

    with tqdm(total=len(jobs), desc="Awaiting job completion...") as pbar:
        while len(returned_ids) < len(jobs):
            time.sleep(1.0)  # Check every second
            for job in jobs:
                if job.job_id not in returned_ids and job.done():
                    job_idx = jobs_ids[job.job_id]
                    subject_id, _ = subject_data_list[job_idx]

                    try:
                        result = job.result()
                        result['subject'] = subject_id
                        result['subject_idx'] = job_idx
                        all_subject_results.append((job_idx, result))
                        pbar.set_postfix_str(f'completed: {subject_id}')
                    except Exception as e:
                        print(f"Job {job.job_id} failed for subject {subject_id}: {e}")
                        # Create a failed result
                        failed_result = {
                            'subject': subject_id,
                            'subject_idx': job_idx,
                            'fitted_params': {},
                            'negative_log_likelihood': np.nan,
                            'log_likelihood': np.nan,
                            'aic': np.nan,
                            'bic': np.nan,
                            'convergence_rate': 0.0,
                            'failed': True,
                            'error_message': str(e)
                        }
                        all_subject_results.append((job_idx, failed_result))
                        pbar.set_postfix_str(f'failed: {subject_id}')

                    returned_ids.append(job.job_id)
                    pbar.update(1)

    # Sort results by original subject order
    all_subject_results.sort(key=lambda x: x[0])
    ordered_results = [result for _, result in all_subject_results]

    # Compute population statistics
    successful_results = [r for r in ordered_results if not r.get('failed', False)]
    if successful_results:
        population_stats = fitter._compute_population_stats(successful_results, config)
    else:
        population_stats = {}
        print("Warning: All subjects failed to fit!")

    # Save results
    results_dict = {
        "individual_results": ordered_results,
        "population_statistics": population_stats,
        "n_subjects": len(ordered_results),
        "n_successful": len(successful_results),
        "fit_toolbox": fit_toolbox,
        "config": config,
        "fit_folder": fit_folder
    }

    # Save to dill file
    safe_dill_results(results_dict, fit_folder, config.name)

    return results_dict

def find_latest_parallel_fit_results(base_dir=None):
    """
    Find the most recent parallel fitting results directory and load the results.
    
    Args:
        base_dir: Base directory to search in. If None, uses environment detection logic.
        
    Returns:
        tuple: (results_dict, results_path) or (None, None) if not found
    """
    # Use the same environment detection logic as RobotDataset
    if base_dir is None:
        # Import the detect_environment function from robot_dataset
        from agents.robot_dataset import detect_environment
        _, base_dir = detect_environment()
    
    base_dir = Path(base_dir)
    
    # Look for parallel fitting results
    parallel_fit_pattern = base_dir / "behavioral_study" / "scripts" / "gonogo-simfit" / "model_fits" / "logs" / "parallel_fitting" / "*"
    
    # Find all parallel fitting directories
    parallel_dirs = glob.glob(str(parallel_fit_pattern))
    parallel_dirs = [d for d in parallel_dirs if os.path.isdir(d)]
    
    if not parallel_dirs:
        print(f"No parallel fitting results found in: {parallel_fit_pattern}")
        return None, None
    
    # Sort by timestamp (directory name) to get the latest
    parallel_dirs.sort(reverse=True)
    latest_dir = parallel_dirs[0]
    
    # Look for the master results file
    results_file = os.path.join(latest_dir, "all_models_results.pkl")
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None, None
    
    try:
        # Try dill first (for newer files with VBMC)
        with open(results_file, 'rb') as f:
            results = dill.load(f)
        
        print(f"Loaded parallel fit results from: {results_file} (dill)")
        print(f"Found {len(results)} models fitted")
        
        return results, results_file
        
    except Exception as e:
        # Fallback to pickle for older files
        try:
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            
            print(f"Loaded parallel fit results from: {results_file} (pickle fallback)")
            print(f"Found {len(results)} models fitted")
            
            return results, results_file
            
        except Exception as e2:
            print(f"Error loading results from {results_file} (dill): {e}")
            print(f"Error loading results from {results_file} (pickle): {e2}")
            return None, None

def list_all_parallel_fit_results(base_dir=None):
    """
    List all available parallel fitting results with timestamps.
    
    Args:
        base_dir: Base directory to search in. If None, uses environment detection logic.
        
    Returns:
        list: List of (timestamp, directory_path) tuples
    """
    # Use the same environment detection logic as RobotDataset
    if base_dir is None:
        from agents.robot_dataset import detect_environment
        _, base_dir = detect_environment()
    
    base_dir = Path(base_dir)
    
    # Look for parallel fitting results
    parallel_fit_pattern = base_dir / "behavioral_study" / "scripts" / "gonogo-simfit" / "model_fits" / "logs" / "parallel_fitting" / "*"
    
    # Find all parallel fitting directories
    parallel_dirs = glob.glob(str(parallel_fit_pattern))
    parallel_dirs = [d for d in parallel_dirs if os.path.isdir(d)]
    
    if not parallel_dirs:
        print(f"No parallel fitting results found in: {parallel_fit_pattern}")
        return []
    
    # Extract timestamps and sort
    results = []
    for dir_path in parallel_dirs:
        timestamp = os.path.basename(dir_path)
        results_file = os.path.join(dir_path, "all_models_results.pkl")
        has_results = os.path.exists(results_file)
        results.append((timestamp, dir_path, has_results))
    
    # Sort by timestamp (most recent first)
    results.sort(reverse=True)
    
    print(f"Found {len(results)} parallel fitting directories:")
    for i, (timestamp, path, has_results) in enumerate(results):
        status = "✓ Has results" if has_results else "✗ No results file"
        print(f"  {i+1}. {timestamp} - {status}")
    
    return results

def load_specific_parallel_fit_results(timestamp, base_dir=None):
    """
    Load parallel fitting results from a specific timestamp.
    
    Args:
        timestamp: Timestamp string (e.g., "20241226_143052")
        base_dir: Base directory to search in. If None, uses environment detection logic.
        
    Returns:
        tuple: (results_dict, results_path) or (None, None) if not found
    """
    if base_dir is None:
        from agents.robot_dataset import detect_environment
        _, base_dir = detect_environment()
    
    base_dir = Path(base_dir)
    
    # Construct path to specific results
    results_dir = base_dir / "behavioral_study" / "scripts" / "gonogo-simfit" / "model_fits" / "logs" / "parallel_fitting" / timestamp
    results_file = results_dir / "all_models_results.pkl"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None, None
    
    try:
        # Try dill first (for newer files with VBMC)
        with open(results_file, 'rb') as f:
            results = dill.load(f)
        
        print(f"Loaded parallel fit results from: {results_file} (dill)")
        print(f"Found {len(results)} models fitted")
        
        return results, str(results_file)
        
    except Exception as e:
        # Fallback to pickle for older files
        try:
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            
            print(f"Loaded parallel fit results from: {results_file} (pickle fallback)")
            print(f"Found {len(results)} models fitted")
            
            return results, str(results_file)
            
        except Exception as e2:
            print(f"Error loading results from {results_file} (dill): {e}")
            print(f"Error loading results from {results_file} (pickle): {e2}")
            return None, None

# %%
