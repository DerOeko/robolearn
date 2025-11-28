# Go/NoGo Reinforcement Learning Model Fitting and Simulation

This repository provides a comprehensive framework for fitting, analyzing, and simulating Rescorla-Wagner reinforcement learning models on Go/NoGo task data. It supports both PyBADS (optimization-based) and PyVBMC (Bayesian) fitting methods, and includes tools for model comparison, parameter recovery, and post-fit simulation.

## Repository Overview

### Core Components

#### Environment (`envs/GoNoGoEnv.py`)
- **GoNoGoEnv**: Custom Gymnasium environment implementing the Go/NoGo task
- **Features**:
  - Configurable controllability schedules (11 predefined sequences)
  - 4 stimulus types: go2win, go2avoid, nogo2win, nogo2avoid
  - Three block types: high control (80%), low control (20%), yoked (performance-dependent)
  - Calibration blocks for determining yoked reward rates
  - Randomized trial sequences with controllable reward contingencies

#### Agents (`agents/rw_agent.py`)
- **RWAgent**: Flexible Rescorla-Wagner agent implementation
- **RWConfig**: Configuration dataclass for model parameters
- **Key Features**:
  - Modular design with swappable update functions
  - Support for multiple parameters: alpha (learning rate), beta (inverse temperature), bias, Pavlovian weights, etc.
  - Built-in parameter bounds for fitting
  - (WIP) Working memory integration capabilities
  - Configurable initial Q-values and state representations

#### Data Analysis (`data_analyzer.py`)
- **DataAnalyzer**: Unified analysis tool for both human and simulation data
- **Capabilities**:
  - Learning curves across trials and block types
  - Win-Stay Lose-Shift (WSLS) analysis
  - Accuracy/performance metrics
  - Reward contingency analysis
  - Side-by-side human vs. model comparisons
  - Consistent color schemes and formatting

#### Post-Fit Simulation (`post_fit_simulation.py`)
- **PostFitSimulator**: Run simulations using fitted parameters
- **Features**:
  - Load fitted results from both PyBADS and PyVBMC
  - Equal or hierarchical (likelihood-weighted) participant sampling
  - Automatic controllability schedule cycling
  - Compatible with existing analysis pipelines
  - Model-specific or combined simulations

#### Fit Analysis (`fit_analyzer.py`)
- **FitAnalyzer**: Comprehensive analysis of fitting results
- **Capabilities**:
  - Parameter distribution visualization (violin plots)
  - Model comparison via AIC/BIC
  - Convergence diagnostics
  - Bayes factor analysis (PyVBMC)
  - Probabilistic model comparison with uncertainty
  - Parameter correlation analysis
  - Export functionality for all analyses

## Workflow: From Model Config to Analysis

### 1. Define Model Configuration

Create model configurations using the `RWConfig` dataclass:

```python
from agents.rw_agent import RWConfig

# Basic Rescorla-Wagner model
basic_rw = RWConfig(
    name="rw_basic",
    fit_parameters=["alpha", "beta"],
    alpha=0.1,
    beta=1.0,
    parameter_bounds={
        "alpha": (0.01, 0.999),
        "beta": (0.1, 10.0)
    }
)

# More complex model with bias and Pavlovian components
complex_rw = RWConfig(
    name="rw_complex",
    fit_parameters=["alpha", "beta", "go_bias", "pavlovian_weight"],
    alpha=0.1,
    beta=1.0,
    go_bias=0.0,
    pavlovian_weight=0.0,
    parameter_bounds={
        "alpha": (0.01, 0.999),
        "beta": (0.1, 10.0),
        "go_bias": (-10.0, 10.0),
        "pavlovian_weight": (0.0, 1.0)
    }
)
```

### 2. Fit Models to Data

The main fitting workflow is implemented in `fitting_notebook.ipynb`. Here's the complete process:

```python
from fitting import fit_all_models_parallel
from agents.agents_configs import all_models, get_models_for_fitting, toggle_model_inclusion
from agents.robot_dataset import RobotDataset

# 1. Load your data
data = RobotDataset()  # Can also load and combine fMRI data into our data!

# 2. Select models for fitting
# Turn off specific models if needed
toggle_model_inclusion(['rw_noise_bias_pav_dynamic_collins_decay_both'], include=False)

# Get models to fit (you can filter by name patterns)
models_to_fit = get_models_for_fitting(include_names=['rw'], exclude_names=['rlwm'])

# 3a. Fit with PyBADS (optimization-based)
pybads_results = fit_all_models_parallel(
    models_to_fit=models_to_fit,
    data=data,
    toolbox="pybads",
    fit_with_slurm=True,  # Use SLURM cluster if available
    n_starts=10,
    bads_options={
        'max_fun_evals': 1000,
        'tol_mesh': 1e-8,
        'tol_fun': 1e-6,
        'tol_stall_iters': 20,
    },
    developmental_subjects=['sub-001', 'sub-020']  # Optional: fit subset first
)

# 3b. Fit with PyVBMC (Bayesian, with PyBADS initialization)
pyvbmc_results = fit_all_models_parallel(
    models_to_fit=models_to_fit,
    data=data,
    toolbox="pyvbmc",
    fit_with_slurm=True,
    n_starts=5,
    vbmc_options={
        'max_fun_evals': 500,
        'tol_con_loss': 0.01,
        'fun_eval_start': 10,
        'k_warmup': 5,
        'max_iter': 100
    },
    vbmc_bads_init_options={
        'max_fun_evals': 100,
        'tol_fun': 1e-3,
        'tol_mesh': 1e-5
    },
    vbmc_init_with_bads=True  # Initialize PyVBMC with PyBADS
)
```

**Key Features:**
- **Parallel Processing**: Automatically distributes fitting across available cores or SLURM cluster
- **Multiple Optimizers**: PyBADS for robust optimization, PyVBMC for Bayesian inference
- **Initialization**: PyVBMC can be initialized with PyBADS for better convergence
- **Developmental Fitting**: Test on subset of subjects before full fitting
- **Automatic Saving**: Results saved with timestamps for later retrieval

### 3. Analyze Fitting Results

Load and analyze fitted parameters using the approach from `post_fit_notebook.ipynb`:

```python
from fit_analyzer import FitAnalyzer
from fitting import list_all_parallel_fit_results, load_specific_parallel_fit_results

# 1. List all available fitted results
list_all_parallel_fit_results()

# 2. Choose a specific timestamp and load results
selected_timestamp = "20250629_194226"  # Replace with your timestamp
results_dict, results_path = load_specific_parallel_fit_results(selected_timestamp)

# 3. Create analyzer with loaded results
analyzer = FitAnalyzer(results_dict=results_dict)

# 4. Get comprehensive summary
analyzer.summary()

# 5. Model comparison table
aic_table = analyzer.get_aic_table()
print(aic_table)

# 6. Visualizations
analyzer.plot_parameter_violins()
analyzer.plot_aic_comparison()
analyzer.plot_parameter_correlations()
analyzer.plot_convergence_diagnostics()

# 7. Bayesian model comparison (PyVBMC only)
analyzer.print_bayes_factor_summary()
analyzer.print_probabilistic_model_comparison()

# 8. Export all analyses
analyzer.export_all_analyses("output_directory")
```

### 4. Run Post-Fit Simulations

Generate model predictions using fitted parameters:

```python
from post_fit_simulation import (
    list_available_fit_results,
    simulate_each_model_from_timestamp,
    analyze_model_simulations
)

# See available results
list_available_fit_results()

# Simulate each model separately
model_simulations = simulate_each_model_from_timestamp(
    timestamp="20250629_194226",
    n_simulations=1000,
    sampling_strategy="hierarchical",  # or "equal"
    random_seed=42
)

# Analyze simulation results
analyze_model_simulations(
    model_simulations=model_simulations,
    subject_data=human_data  # Optional: for comparison
)
```

### 5. Compare with Human Data

Analyze model performance against human behavior:

```python
from data_analyzer import DataAnalyzer

# Load human data (implementation depends on your data format)
human_analyzer = DataAnalyzer(human_data, is_subject_data=True)

# Human behavior analysis
human_analyzer.plot_learning_curves()
human_analyzer.plot_wsls()
human_analyzer.plot_accuracy()

# Model vs. human comparison
for model_name, sim_data in model_simulations.items():
    print(f"=== {model_name} ===")
    model_analyzer = DataAnalyzer(sim_data, is_subject_data=False)
    
    # Direct comparison
    DataAnalyzer.compare_datasets(
        subject_data=human_data,
        simulation_data=sim_data
    )
```

## Key Features

### Environment Design
- **Controllability Manipulation**: High vs. low control blocks test learning under different contingency strengths
- **Yoked Controls**: Performance-matched conditions control for reward rate effects
- **Calibration Blocks**: Establish baseline performance for yoked conditions
- **Stimulus Variety**: Four distinct stimulus-action-outcome combinations

### Model Flexibility
- **Modular Architecture**: Easy to add new parameters or update functions
- **Parameter Bounds**: Automatic constraint handling during fitting
- **Multiple Optimizers**: Support for both PyBADS and PyVBMC
- **Extensible**: Add new agent types or modify existing ones

### Analysis Pipeline
- **Unified Interface**: Same tools work for human and simulation data
- **Statistical Rigor**: Proper error bars, confidence intervals, and uncertainty quantification
- **Model Comparison**: Multiple criteria (AIC, BIC, Bayes factors)
- **Visualization**: Publication-ready plots with consistent styling

### Simulation System
- **Faithful Reproduction**: Uses exact fitted parameters and environments
- **Sampling Strategies**: Equal or likelihood-weighted participant selection
- **Controllability**: Automatic cycling through different experimental conditions
- **Scalability**: Efficient handling of large simulation studies

## Adding New Models

To add a new model variant:

1. **Define Configuration**:
```python
new_model_config = RWConfig(
    name="my_new_model",
    fit_parameters=["alpha", "beta", "new_param"],
    new_param=0.5,  # default value
    parameter_bounds={
        "alpha": (0.01, 0.999),
        "beta": (0.1, 10.0),
        "new_param": (0.0, 1.0)
    }
)
```

2. **Implement Update Function** (if needed):
```python
def my_custom_update(agent, observation, action, reward, next_observation, terminated, log_history):
    # Custom learning rule implementation
    # Access agent.new_param as needed
    pass

new_model_config.update_fn = my_custom_update
```

3. **Add to Model Space**: Include in your model comparison dictionary and run through the fitting pipeline.

## Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib, Seaborn
- Gymnasium (for environment)
- PyBADS and/or PyVBMC (for fitting)
- Dill (for result serialization)

## Usage Example: Complete Workflow

See `post_fit_notebook.ipynb` for a complete example that demonstrates:
1. Loading and visualizing human data
2. Exploring fitted model results
3. Running model-specific simulations
4. Comprehensive analysis and comparison
5. Advanced configuration options

This notebook serves as both documentation and a practical starting point for your own analyses.

## File Structure

```
gonogo-simfit/
├── agents/                    # Agent implementations
│   ├── rw_agent.py           # Main Rescorla-Wagner agent
│   └── ...
├── envs/                     # Environment implementations
│   └── GoNoGoEnv.py         # Go/NoGo task environment
├── data_analyzer.py         # Data analysis tools
├── fit_analyzer.py          # Fitting result analysis
├── post_fit_simulation.py   # Post-fit simulation system
├── post_fit_notebook.ipynb  # Complete workflow example
└── README.md                # This file
```

For questions or issues, please refer to the inline documentation in each module or examine the example notebook.

## Contact

For questions, reach out to me (Samuel Nellessen):
- Email: samuelgerrit.nellessen@gmail.com
- GitHub: https://github.com/DerOeko