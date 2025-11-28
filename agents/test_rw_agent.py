# %% test_rw_agent.py
import numpy as np
from rw_agent import RescorlaWagnerAgent
from agent_configs import RescorlaWagnerAgentConfig
# %%


def print_q_values(agent, state_idx=None):
    if state_idx is not None:
        print(f"  Q-values for state {state_idx}: {agent.q_values[state_idx]}")
    else:
        print(f"  All Q-values:\n{agent.q_values}")


def test_action_selection_and_probabilities():
    print("\n--- Testing Action Selection and Probabilities ---")
    n_s, n_a = 1, 2  # Single state, 2 actions

    # Test 1: Skewed Q-values, moderate beta
    cfg1 = RescorlaWagnerAgentConfig(beta=1.0, initial_q_value=0.0)
    agent1 = RescorlaWagnerAgent(n_states=n_s, n_actions=n_a, config=cfg1)
    agent1.q_values[0, 0] = 1.0  # Action 0 is better
    agent1.q_values[0, 1] = 0.0
    # Assuming you changed get_action_probabilities to _get_action_probabilities if it's internal
    probs1 = agent1._get_action_probabilities(0)
    assert np.isclose(np.sum(probs1), 1.0)
    assert probs1[0] > probs1[1]
    print(
        f"✓ Probabilities with beta={cfg1.beta}, Qs=[1,0]: {probs1} (Action 0 preferred)")

    # Test 2: Skewed Q-values, high beta (more deterministic)
    cfg2 = RescorlaWagnerAgentConfig(beta=10.0, initial_q_value=0.0)
    agent2 = RescorlaWagnerAgent(n_states=n_s, n_actions=n_a, config=cfg2)
    agent2.q_values[0, 0] = 1.0
    agent2.q_values[0, 1] = 0.0
    probs2 = agent2._get_action_probabilities(0)
    assert np.isclose(np.sum(probs2), 1.0)
    assert probs2[0] > 0.99  # Expect near greedy
    print(
        f"✓ Probabilities with beta={cfg2.beta}, Qs=[1,0]: {probs2} (Action 0 strongly preferred)")

    # Test 3: Equal Q-values (should be uniform probabilities)
    cfg3 = RescorlaWagnerAgentConfig(beta=1.0, initial_q_value=0.5)
    agent3 = RescorlaWagnerAgent(n_states=n_s, n_actions=n_a, config=cfg3)
    probs3 = agent3._get_action_probabilities(0)
    assert np.isclose(np.sum(probs3), 1.0)
    assert np.isclose(probs3[0], 0.5) and np.isclose(probs3[1], 0.5)
    print(
        f"✓ Probabilities with beta={cfg3.beta}, Qs=[0.5,0.5]: {probs3} (Uniform)")

    # Test 4: Action choice (stochastic)
    action_chosen = agent1.choose_action(0)
    assert action_chosen in [0, 1]
    print(
        f"✓ choose_action selected: {action_chosen} (based on probs {probs1})")

    # Test 5: More than 2 actions (general softmax)
    n_s_multi, n_a_multi = 1, 3
    cfg4 = RescorlaWagnerAgentConfig(beta=1.0)
    agent4 = RescorlaWagnerAgent(
        n_states=n_s_multi, n_actions=n_a_multi, config=cfg4)
    agent4.q_values[0, :] = np.array([2.0, 1.0, 0.0])
    probs4 = agent4._get_action_probabilities(0)
    assert np.isclose(np.sum(probs4), 1.0)
    assert probs4[0] > probs4[1] > probs4[2]
    print(
        f"✓ Probabilities (3 actions) with beta={cfg4.beta}, Qs=[2,1,0]: {probs4}")

    # AAAAAADDDDitional tests below this line

    # Test 6: _greedy method
    print("  --- Testing _greedy method ---")
    cfg_greedy = RescorlaWagnerAgentConfig()
    agent_greedy = RescorlaWagnerAgent(
        n_states=n_s, n_actions=n_a, config=cfg_greedy)
    agent_greedy.q_values[0, 0] = 2.5
    agent_greedy.q_values[0, 1] = 1.5
    max_q_value = agent_greedy._greedy(0)
    assert np.isclose(max_q_value, 2.5)
    print(f"✓ _greedy returned max Q-value: {max_q_value} (expected 2.5)")

    agent_greedy.q_values[0, 0] = -0.5
    agent_greedy.q_values[0, 1] = 0.5
    max_q_value_2 = agent_greedy._greedy(0)
    assert np.isclose(max_q_value_2, 0.5)
    print(f"✓ _greedy returned max Q-value: {max_q_value_2} (expected 0.5)")

    # Test 7: _epsilon_greedy method
    print("  --- Testing _epsilon_greedy method ---")
    # Case 7.1: Epsilon = 1.0 (always explore)
    cfg_eps1 = RescorlaWagnerAgentConfig(epsilon=1.0, seed=42)
    agent_eps1 = RescorlaWagnerAgent(
        n_states=n_s, n_actions=n_a, config=cfg_eps1)
    agent_eps1.q_values[0, 0] = 100  # Greedy action is 0
    agent_eps1.q_values[0, 1] = 0
    actions_eps1 = [agent_eps1._epsilon_greedy(
        0) for _ in range(20)]  # Generate a few actions
    assert all(a in [0, 1]
               for a in actions_eps1), "Epsilon-greedy (eps=1.0) chose invalid action."
    # Check if both actions are selected at some point (probabilistic, but likely for 20 samples)
    if len(set(actions_eps1)) > 1:
        print(
            f"✓ _epsilon_greedy (epsilon=1.0) selected multiple actions (e.g., some of {actions_eps1[:5]}), exploring.")
    else:
        print(
            f"✓ _epsilon_greedy (epsilon=1.0) selected one action repeatedly (e.g., {actions_eps1[:5]}), which is possible but less likely to show exploration in few samples.")

    # Case 7.2: Epsilon = 0.0 (always exploit - testing current agent behavior)
    cfg_eps0 = RescorlaWagnerAgentConfig(
        epsilon=0.0, seed=42)  # Ensure epsilon is not None
    agent_eps0 = RescorlaWagnerAgent(
        n_states=n_s, n_actions=n_a, config=cfg_eps0)
    agent_eps0.q_values[0, 0] = 100
    agent_eps0.q_values[0, 1] = 0
    # Based on current agent code, if not (rng < epsilon), it returns None implicitly.
    # If rng.random() can be exactly 0.0, and epsilon is 0.0, (0.0 < 0.0) is False.
    action_eps0 = agent_eps0._epsilon_greedy(0)
    assert action_eps0 is None, f"Expected None for epsilon=0.0 due to missing else, got {action_eps0}"
    print(
        f"✓ _epsilon_greedy (epsilon=0.0) returned {action_eps0} (as expected from current implementation with missing else clause).")

    # Case 7.3: Epsilon is None (should raise assertion in agent)
    cfg_eps_none = RescorlaWagnerAgentConfig(epsilon=None, seed=42)
    agent_eps_none = RescorlaWagnerAgent(
        n_states=n_s, n_actions=n_a, config=cfg_eps_none)
    raised_assertion = False
    try:
        agent_eps_none._epsilon_greedy(0)
    except AssertionError as e:
        if "epsilon must be provided" in str(e):
            raised_assertion = True
    assert raised_assertion, "_epsilon_greedy did not raise AssertionError when epsilon is None."
    print(f"✓ _epsilon_greedy correctly raised AssertionError when epsilon is None.")


def test_agent_initialization_and_reset():
    print("\n--- Testing Agent Initialization and Reset ---")
    n_s, n_a = 3, 2

    # Test 1: Default config
    agent_default = RescorlaWagnerAgent(n_states=n_s, n_actions=n_a)
    default_cfg = RescorlaWagnerAgentConfig()  # For comparison
    assert agent_default.n_states == n_s
    assert agent_default.n_actions == n_a
    assert agent_default.alpha == default_cfg.alpha
    assert agent_default.beta == default_cfg.beta
    assert np.all(agent_default.q_values == default_cfg.initial_q_value)
    print("✓ Default initialization successful.")
    print_q_values(agent_default, 0)

    # Test 2: Custom config
    custom_params = {"alpha": 0.5, "beta": 2.0,
                     "initial_q_value": -1.0, "name": "CustomRW"}
    agent_custom_cfg = RescorlaWagnerAgentConfig(**custom_params)
    agent_custom = RescorlaWagnerAgent(
        n_states=n_s, n_actions=n_a, config=agent_custom_cfg)

    assert agent_custom.alpha == custom_params["alpha"]
    assert agent_custom.beta == custom_params["beta"]
    assert np.all(agent_custom.q_values == custom_params["initial_q_value"])
    assert agent_custom.name == custom_params["name"]
    print(f"✓ Custom configuration successful ({agent_custom._get_name()}).")
    print_q_values(agent_custom, 0)

    # Test 3: Reset
    agent_custom.q_values[0, 0] = 100  # Modify Q-value
    agent_custom.reset()
    assert np.all(agent_custom.q_values == custom_params["initial_q_value"])
    print("✓ Reset to initial Q-values successful.")
    print_q_values(agent_custom, 0)


def test_learning_update():
    print("\n--- Testing Learning Update ---")
    n_s, n_a = 2, 2

    # Scenario A: Classic RW-like update
    print("  Scenario A: Classic RW-like update (target = reward)")
    cfg_rw = RescorlaWagnerAgentConfig(
        alpha=0.5, beta=1.0, initial_q_value=0.0)
    agent_rw = RescorlaWagnerAgent(n_states=n_s, n_actions=n_a, config=cfg_rw)

    obs, action, reward, next_obs, term = 0, 0, 1.0, 1, True  # Terminated
    q_old = agent_rw.q_values[obs, action]
    agent_rw.update(obs, action, reward, next_obs, term)
    q_new_expected = q_old + cfg_rw.alpha * (reward - q_old)
    assert np.isclose(agent_rw.q_values[obs, action], q_new_expected)
    print(
        f"✓ RW update (terminated): Q({obs},{action}) from {q_old:.2f} to {agent_rw.q_values[obs, action]:.2f} (expected {q_new_expected:.2f})")


def test_reproducibility_with_seed():
    print("\n--- Testing Reproducibility with Seed ---")
    n_s, n_a = 1, 2
    q_vals_skewed = np.array([[1.0, 0.0]])

    cfg_seed = RescorlaWagnerAgentConfig(beta=0.5, seed=123)
    agent_s1 = RescorlaWagnerAgent(
        n_states=n_s, n_actions=n_a, config=cfg_seed)
    agent_s1.q_values = q_vals_skewed.copy()
    actions1 = [agent_s1.choose_action(0) for _ in range(10)]

    agent_s2 = RescorlaWagnerAgent(
        n_states=n_s, n_actions=n_a, config=cfg_seed)  # Same seed
    agent_s2.q_values = q_vals_skewed.copy()
    actions2 = [agent_s2.choose_action(0) for _ in range(10)]

    assert actions1 == actions2, f"Actions mismatch with same seed: {actions1} vs {actions2}"
    print(f"✓ Agents with same seed produced same action sequence: {actions1}")

    cfg_seed_diff = RescorlaWagnerAgentConfig(
        beta=0.5, seed=456)  # Different seed
    agent_s3 = RescorlaWagnerAgent(
        n_states=n_s, n_actions=n_a, config=cfg_seed_diff)
    agent_s3.q_values = q_vals_skewed.copy()
    actions3 = [agent_s3.choose_action(0) for _ in range(10)]

    if actions1 != actions3:
        print(
            f"✓ Agent with different seed produced different action sequence (e.g., {actions3}) as expected.")
    else:
        print(f"✓ Agent with different seed produced same action sequence (possible for short sequences or low stochasticity).")


if __name__ == "__main__":
    test_agent_initialization_and_reset()
    test_action_selection_and_probabilities()
    test_learning_update()
    test_reproducibility_with_seed()
    print("\nAll agent tests finished.")
# %%
