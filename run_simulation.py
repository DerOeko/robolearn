#%% run_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from envs.GoNoGoEnv import GoNoGoEnv, GoNoGoConfig
from agents.rw_agent import RWAgent, RWConfig
from simulation_analysis import SimulationRunner, compare_agents
from data_analyzer import DataAnalyzer
from agents.agents_configs import return_configs
from agents.agents_configs import rlwm_models
from agents.agents_configs import all_models, get_models_for_fitting, toggle_model_inclusion
print("=== Enhanced Agent Simulation and Analysis ===")

# Setup Environment
print("\n--- Setting up Environment ---")
env_config = GoNoGoConfig(
    n_blocks=8,
    add_calibration=True,
    n_states=4,
    c_schedule_idx=4
)
env = GoNoGoEnv(config=env_config)

try:
    from gymnasium.utils.env_checker import check_env
    check_env(env)
    print("✓ Environment is API Compliant Gymnasium Environment.")
except ImportError:
    print("Note: gymnasium.utils.env_checker not found.")
except Exception as e:
    print(f"✗ Environment API Compliance Check Failed: {e}")

print(f"Environment: {env.spec.name if env.spec else 'GoNoGoEnv'}")
print(f"  - Observation Space: {env.observation_space}")
print(f"  - Action Space: {env.action_space}")

# Test different agent configurations
print("\n--- Testing Different Agent Configurations ---")
np.random.seed(122)  # for reproducible "randomness"
configs = get_models_for_fitting(include_names=['rw'], exclude_names=['rlwm', 'collins'])


# Compare configurations using structured simulation with multiple runs
print("\n--- Comparing Agent Configurations ---")
comparison_results = compare_agents(
    configs, env,
    n_simulations=100 # Run X simulations per configuration
)
save = False
for name, result in comparison_results.items():
    print(f"\n{name} Results:")

    raw_data = result['combined_data']
    analyzer = DataAnalyzer(raw_data = raw_data)
    #if save:
    #    learning_curves = analyzer.plot_learning_curves(title=f"Learning Curves: {name}", save_path=f"{name}_learning_curves.png")
    #else:
    #    learning_curves = analyzer.plot_learning_curves(title=f"Learning Curves: {name}")

    wsls_plots = analyzer.plot_wsls(title=f"WSLS Analysis: {name}", save_path=f"{name}_wsls_analysis.png")


# %%