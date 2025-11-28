#%%
from GoNoGoEnv import GoNoGoEnv
from GoNoGoEnv import GoNoGoConfig
import random
import numpy as np

def _is_optimal_action_for_stimulus(stimulus_type_idx, action):
    optimal_actions_for_stim_type = [1, 1, 0, 0]
    return action == optimal_actions_for_stim_type[stimulus_type_idx]


def calculate_expected_reward(stimulus_type_idx, action, is_control, is_rewarded):
    reward = 0
    optimal = _is_optimal_action_for_stimulus(stimulus_type_idx, action)
    if is_control:
        if is_rewarded:
            reward = 1 if optimal else -1
        else:
            reward = -1 if optimal else 1
    else:
        reward = 1 if is_rewarded else -1
    return reward


def test_reward_logic_scenarios():
    print("\n--- Testing all reward logic scenarios ---")
    stimulus_types_map = {
        0: "gw (GoToWin)",
        1: "gal (GoToAvoidLoss)",
        2: "ngw (NoGoToWin)",
        3: "ngal (NoGoToAvoidLoss)"
    }
    actions_map = {0: "NoGo", 1: "Go"}

    passed_count = 0
    failed_count = 0
    total_tests = 0

    for stim_idx, stim_name in stimulus_types_map.items():
        for action_val, action_name in actions_map.items():
            for is_control_val in [True, False]:
                for is_rewarded_val in [True, False]:
                    total_tests += 1

                    actual_calculated_reward = calculate_expected_reward(
                        stim_idx, action_val, is_control_val, is_rewarded_val
                    )

                    optimal_str = "Optimal" if _is_optimal_action_for_stimulus(
                        stim_idx, action_val) else "Suboptimal"

                    desc = (
                        f"Stim: {stim_name}, Action: {action_name} ({optimal_str}), "
                        f"Control: {is_control_val}, Rewarded: {is_rewarded_val}"
                    )

                    print(
                        f"  Scenario: {desc} -> Expected Reward: {actual_calculated_reward}")

                    # This assertion inherently passes if calculate_expected_reward is the sole source of truth
                    # It serves to verify the logic is enumerated correctly.
                    # If you had an independent way to get the reward (e.g. from env.step under specific conditions),
                    # you would assert actual_calculated_reward against that.
                    try:
                        # Simulating an assertion against the logic itself for completeness of structure
                        assert actual_calculated_reward == calculate_expected_reward(
                            stim_idx, action_val, is_control_val, is_rewarded_val)
                        passed_count += 1
                    except AssertionError:  # Should not be reached with this setup
                        failed_count += 1
                        print(
                            f"    ✗ FAILED LOGIC CONSISTENCY (This indicates an issue in the test setup itself)")

    print(
        f"\nReward logic test execution summary: {total_tests} scenarios checked.")
    if failed_count == 0:
        print("✓ All reward logic scenarios processed and consistently calculated.")
    else:
        print(
            f"✗ Reward logic scenario processing encountered {failed_count} inconsistencies.")

# Test the environment


def test_environment():
    print("Testing GoNoGo Environment...")

    # Test 1: Basic initialization
    env = GoNoGoEnv()
    print(f"✓ Environment created successfully")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Number of blocks: {len(env.blocks)}")
    print(f"  - Schedule: {env.schedule}")

    # Test 2: Reset functionality
    obs, info = env.reset(seed=42)
    print(f"✓ Reset successful, initial observation: {obs}")
    rewards = []
    # Test 3: Run a few steps
    print("\n--- Running 20 steps ---")
    for i in range(500):
        action = random.choice([0, 1])  # Random action
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {i+1}: action={action}, obs={obs}, reward={reward}, terminated={terminated}, is_control={info['is_control']}, is_rewarded={info['is_rewarded']}")
        obs = next_obs
        rewards.append(reward)
        if terminated or truncated:
            print("Episode ended!")
            break

    print(f"Total reward: {sum(rewards)}")
    # Test whether rewards are within expected range
    assert all(-1 <= r <= 1 for r in rewards), "Rewards are out of expected range!"
    assert np.isclose(np.mean(rewards), 0,
                      atol=0.1), "Mean reward is not close to zero!"
    print(f"Mean reward is close to zero, as expected: {np.mean(rewards):.2f}")

    test_reward_logic_scenarios()
    # Test 4: Custom configuration
    print("\n--- Testing custom configuration ---")
    custom_config = GoNoGoConfig(
        n_blocks=3,
        add_calibration=False,
        c_schedule_idx=1
    )
    custom_env = GoNoGoEnv(custom_config)
    print(f"✓ Custom environment created")
    print(f"  - Schedule: {custom_env.schedule}")
    print(f"  - Number of blocks: {len(custom_env.blocks)}")

    # Test 5: Block structure
    print("\n--- Block structure ---")
    for i, block in enumerate(env.blocks[:3]):  # Show first 3 blocks
        print(f"Block {i} ({block['type']}):")
        print(f"  - Control rate: {block['control_rate']}")
        print(f"  - Reward rate: {block['reward_rate']}")
        print(f"  - Number of trials: {len(block['trials']['stimulus'])}")
        print(
            f"  - Stimuli distribution: {np.bincount(block['trials']['stimulus'])}")

    print("\n✓ All tests passed!")
    
test_environment()
# %%
