"""
Test multi-planner support
"""

import traceback
import numpy as np


from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
from ai_economist.foundation.base.world import World
from ai_economist.foundation.agents.regional_planner import TopPlanner, BottomPlanner


@scenario_registry.add
class SinglePlannerScenario(BaseEnvironment):
    name = "test/single_planner"
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]
    required_entities = []

    def reset_starting_layout(self):
        self.world.maps.clear()

    def reset_agent_states(self):
        self.world.clear_agent_locs()
        for agent in self.world.agents:
            agent.state["inventory"] = {"Coin": 0}
            agent.state["escrow"] = {"Coin": 0}
            agent.state["endogenous"] = {"Labor": 0}
            # place randomly
            r = np.random.randint(0, self.world_size[0])
            c = np.random.randint(0, self.world_size[1])
            self.world.set_agent_loc(agent, r, c)

        # initialize planner
        p = self.world.planner
        p.state["inventory"] = {"Coin": 0}
        p.state["escrow"] = {"Coin": 0}

    def scenario_step(self):
        return

    def generate_observations(self):
        obs = {}
        for a in self.world.agents:
            obs[str(a.idx)] = {"loc": np.array(a.loc)}
        obs[self.world.planner.idx] = {"planner-obs": 1.0}
        return obs

    def compute_reward(self):
        r = {str(a.idx): 0.0 for a in self.world.agents}
        r[self.world.planner.idx] = 0.0
        return r


@scenario_registry.add
class MultiPlannerScenario(BaseEnvironment):
    name = "test/multi_planner"
    agent_subclasses = ["BasicMobileAgent", "TopPlanner", "BottomPlanner"]
    required_entities = []
    planner_subclasses = ["TopPlanner", "BottomPlanner"]

    def reset_starting_layout(self):
        self.world.maps.clear()

    def reset_agent_states(self):
        self.world.clear_agent_locs()
        for agent in self.world.agents:
            agent.state["inventory"] = {"Coin": 0}
            agent.state["escrow"] = {"Coin": 0}
            agent.state["endogenous"] = {"Labor": 0}
            r = np.random.randint(0, self.world_size[0])
            c = np.random.randint(0, self.world_size[1])
            self.world.set_agent_loc(agent, r, c)

        for p in self.world.planners:
            p.state["inventory"] = {"Coin": 0}
            p.state["escrow"] = {"Coin": 0}

    def scenario_step(self):
        return

    def generate_observations(self):
        obs = {}
        for a in self.world.agents:
            obs[str(a.idx)] = {"loc": np.array(a.loc)}
        for p in self.world.planners:
            obs[str(p.idx)] = {"planner-obs": 1.0}
        return obs

    def compute_reward(self):
        r = {str(a.idx): 0.0 for a in self.world.agents}
        for p in self.world.planners:
            r[str(p.idx)] = 0.0
        return r


def run_test(func):
    try:
        print(f"\n=== Running {func.__name__} ===")
        func()
        print(f"[PASS] {func.__name__}")
    except Exception as e:
        print(f"[FAIL] {func.__name__}: {e}")
        traceback.print_exc()


def test_world_multi_planner():
    w = World(
        world_size=[8,8],
        n_agents=3,
        world_resources=["Coin"],
        world_landmarks=[],
        multi_action_mode_agents=False,
        multi_action_mode_planner=False,
        planner_subclasses=["TopPlanner", "BottomPlanner"],
    )

    ids = [p.idx for p in w.planners]
    print("Planners:", ids)
    assert ids == ["p_top", "p_bottom"]
    assert w.planner.idx == "p_top"  # legacy pointer


def test_single_planner_env():
    env = SinglePlannerScenario(
        components=[],
        n_agents=3,
        world_size=[6, 6],
        flatten_observations=True,
        flatten_masks=True,
    )

    obs = env.reset()
    planner_id = str(env.world.planner.idx)
    print("Planner ID:", planner_id)
    assert planner_id in obs
    assert "action_mask" in obs[planner_id]


def test_multi_planner_env():
    env = MultiPlannerScenario(
        components=[],
        n_agents=4,
        world_size=[7, 7],
        flatten_observations=True,
        flatten_masks=True,
        multi_action_mode_planner=False, #could also just change the global in base_env to false: might look at later
    )

    obs = env.reset()
    planner_ids = [str(p.idx) for p in env.world.planners]
    print("Planner IDs:", planner_ids)

    for pid in planner_ids:
        assert pid in obs
        assert "action_mask" in obs[pid]

    for _ in range(2):
        actions = {aid: 0 for aid in obs.keys()}
        obs, rew, done, info = env.step(actions)
        for pid in planner_ids:
            assert pid in rew


def test_multi_planner_collation_guard():
    try:
        env = MultiPlannerScenario(
            components=[],
            n_agents=3,
            world_size=[6,6],
            flatten_observations=True,
            flatten_masks=True,
            collate_agent_step_and_reset_data=True,
        )
        env.reset()
        raise AssertionError("Expected collate guard error")
    except NotImplementedError:
        print("Collation guard OK")



if __name__ == "__main__":
    run_test(test_world_multi_planner)
    run_test(test_single_planner_env)
    run_test(test_multi_planner_env)
    run_test(test_multi_planner_collation_guard)