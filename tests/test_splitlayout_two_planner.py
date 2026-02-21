import traceback
import numpy as np
import pytest

# Import the scenario under test
from ai_economist.foundation.scenarios.regional_two_planner import SplitLayoutTwoPlanner


def _make_env(
    n_agents=6,
    world_size=(25, 25),
    episode_length=200,
    multi_action_mode_agents=False,
    multi_action_mode_planner=False,
    flatten_observations=True,
    flatten_masks=True,
    planner_reward_type="coin_eq_times_productivity",
    mixing_weight_gini_vs_coin=0.0,
    planner_gets_spatial_info=False,
    collate_agent_step_and_reset_data=False,
):
    """
    Helper to build the 2-planner split-world env with minimal components required
    by SplitLayout (Build with pareto skills + Gather).
    """
    components = [
        {"Build": {"skill_dist": "pareto"}},  # SplitLayout expects pareto skills
        {"Gather": {}},
    ]
    env = SplitLayoutTwoPlanner(
        components=components,
        n_agents=int(n_agents),
        world_size=list(world_size),
        episode_length=int(episode_length),
        multi_action_mode_agents=bool(multi_action_mode_agents),
        multi_action_mode_planner=bool(multi_action_mode_planner),
        flatten_observations=bool(flatten_observations),
        flatten_masks=bool(flatten_masks),
        planner_reward_type=planner_reward_type,
        mixing_weight_gini_vs_coin=mixing_weight_gini_vs_coin,
        planner_gets_spatial_info=planner_gets_spatial_info,
        collate_agent_step_and_reset_data=collate_agent_step_and_reset_data,
    )
    return env


def test_world_has_two_planners_and_legacy_pointer():
    env = _make_env()
    obs = env.reset()
    planner_ids = [str(p.idx) for p in env.world.planners]
    assert set(planner_ids) == {"p_top", "p_bottom"}, f"Got planners: {planner_ids}"
    # legacy pointer should point to the top planner
    assert str(env.world.planner.idx) == "p_top"


def test_agent_assignment_matches_waterline_and_is_fixed_for_episode():
    env = _make_env()
    obs = env.reset()
    # The scenario builds fixed assignments at reset
    assert hasattr(env, "agent_to_planner") and hasattr(env, "planner_to_agents")
    assert "p_top" in env.planner_to_agents and "p_bottom" in env.planner_to_agents

    # Check each agent's initial row vs water line matches assignment
    for a in env.world.agents:
        assigned_pid = env.agent_to_planner[str(a.idx)]
        if a.loc[0] <= env._water_line:
            assert assigned_pid == "p_top"
        else:
            assert assigned_pid == "p_bottom"

    # Sanity: assignments partition the set of agent ids
    top_set = set(env.planner_to_agents["p_top"])
    bottom_set = set(env.planner_to_agents["p_bottom"])
    all_ids = set(str(a.idx) for a in env.world.agents)
    assert top_set.isdisjoint(bottom_set)
    assert top_set.union(bottom_set) == all_ids


def test_planner_observations_present_with_regional_summaries_and_masks():
    env = _make_env()
    obs = env.reset()
    for pid in ["p_top", "p_bottom"]:
        assert pid in obs, f"Missing planner obs for {pid}"
        # Regional summary keys in our scenario:
        for k in ["world-n_region", "world-avg_coin_region", "world-equality_region", "world-productivity_region", "action_mask"]:
            assert k in obs[pid], f"Missing key '{k}' in obs[{pid}]"


def test_planner_rewards_are_regional_productivity_when_lambda_eq_1():
    """
    Make sure regional reward signal is localized.
    We set mixing_weight_gini_vs_coin=1.0 → productivity-only (no equality)
    Then add coin to a TOP agent before stepping; TOP planner reward should increase
    more than BOTTOM.
    """
    env = _make_env(mixing_weight_gini_vs_coin=1.0)  # productivity-only
    obs = env.reset()

    # pick a top agent
    assert len(env.planner_to_agents["p_top"]) > 0, "No agents assigned to top region!"
    top_agent_id = env.planner_to_agents["p_top"][0]
    top_agent = env.get_agent(top_agent_id)
    # increase coin for top agent prior to the step
    top_agent.state["inventory"]["Coin"] += 10.0

    # all agents take NO-OP (planner has no actions yet)
    actions = {aid: 0 for aid in obs.keys()}
    obs2, rew, done, info = env.step(actions)

    assert "p_top" in rew and "p_bottom" in rew
    assert rew["p_top"] > rew["p_bottom"], f"Expected top reward > bottom; got {rew}"


def test_metrics_include_regional_keys():
    env = _make_env()
    env.reset()
    m = env.metrics  # scenario_metrics + components (we only check scenario keys)
    for k in ["global/productivity", "global/equality",
              "p_top/n", "p_top/productivity", "p_top/equality",
              "p_bottom/n", "p_bottom/productivity", "p_bottom/equality"]:
        assert k in m, f"Missing metrics key: {k}"


def test_collation_guard_raises_with_multiple_planners():
    env = _make_env(collate_agent_step_and_reset_data=True)
    with pytest.raises(NotImplementedError):
        env.reset()


def test_planner_gets_spatial_info_legacy_pointer_when_enabled():
    """
    When planner_gets_spatial_info=True, the base (SplitLayout) will place 'map' and
    'idx_map' under the legacy planner pointer id (which is p_top).
    """
    env = _make_env(planner_gets_spatial_info=True)
    obs = env.reset()
    assert "p_top" in obs, "Legacy planner pointer should be p_top"
    assert "world-map" in obs["p_top"] and "world-idx_map" in obs["p_top"], \
            "Expected spatial maps under obs['p_top'] when planner_gets_spatial_info=True"