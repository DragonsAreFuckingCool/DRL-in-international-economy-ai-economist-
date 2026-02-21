# Copyright (c) 2026, your org.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from copy import deepcopy

from ai_economist.foundation.agents.regional_planner import TopPlanner, BottomPlanner 

from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
# IMPORTANT: import the multi-planner capable SplitLayout (yours)
# If SplitLayout is in ai_economist.foundation.scenarios.layout_from_file, import it:
from ai_economist.foundation.scenarios.simple_wood_and_stone.layout_from_file import SplitLayout

from ai_economist.foundation.scenarios.utils import rewards, social_metrics


@scenario_registry.add
class SplitLayoutTwoPlanner(SplitLayout):
    """
    Split world by a water row with **two planners**:
      - TopPlanner (idx="p_top") regulating agents in the **north** (row <= water_line)
      - BottomPlanner (idx="p_bottom") regulating agents in the **south** (row > water_line)

    There is **no interaction** between planners. Each planner observes and is rewarded
    based only on **its assigned agents**.

    Notes:
      - This subclass assumes your BaseEnvironment+World support multi-planner:
            planner_subclasses=["TopPlanner","BottomPlanner"]
      - Agent->Planner assignment is **fixed at reset** (by initial side of water).
      - Planner reward uses your `rewards.py` & `social_metrics.py` utilities.
    """

    # Registry name for your config usage
    name = "split_layout/two_planner_simple"
    # Agents used in this scenario
    agent_subclasses = ["BasicMobileAgent", "TopPlanner", "BottomPlanner"]
    required_entities = ["Wood", "Stone", "Water"]

    # Tell BaseEnvironment/World we want two planners
    planner_subclasses = ["TopPlanner", "BottomPlanner"]

    def __init__(
        self,
        *base_env_args,
        # === Planner reward configuration ===
        planner_reward_type="coin_eq_times_productivity",
        mixing_weight_gini_vs_coin=0.0,  # 0 => eq*prod, 1 => prod-only
        # === Planner observation configuration ===
        planner_gets_spatial_info=False,
        **base_env_kwargs,
    ):
        """
        `planner_reward_type` in {"coin_eq_times_productivity",
                                  "inv_income_weighted_coin_endowments",
                                  "inv_income_weighted_utility"}
        `mixing_weight_gini_vs_coin` only applies to coin_eq_times_productivity.

        `planner_gets_spatial_info` (False): keep planners' obs lightweight. Set True
        if you also want to add per-planner spatial maps (optional).
        """
        super().__init__(
            *base_env_args,
            planner_gets_spatial_info=planner_gets_spatial_info,
            **base_env_kwargs
        )
        self.planner_reward_type = str(planner_reward_type).lower()
        self.mixing_weight_gini_vs_coin = float(mixing_weight_gini_vs_coin)
        assert 0.0 <= self.mixing_weight_gini_vs_coin <= 1.0

        # Regional mappings are set at reset
        self.agent_to_planner = {}
        self.planner_to_agents = {"p_top": [], "p_bottom": []}

        # For marginal rewards (same pattern as AI-Economist)
        self.init_optimization_metric = None
        self.prev_optimization_metric = None
        self.curr_optimization_metric = None

    # ------------------------------
    # Helper: Fixed assignment logic
    # ------------------------------
    def _assign_agents_to_planners(self):
        """Assign each agent to 'p_top' or 'p_bottom' based on initial row vs water line."""
        self.agent_to_planner = {}
        self.planner_to_agents = {"p_top": [], "p_bottom": []}

        # You already have self._water_line from SplitLayout
        for a in self.world.agents:
            pid = "p_top" if a.loc[0] <= self._water_line else "p_bottom"
            self.agent_to_planner[str(a.idx)] = pid
            self.planner_to_agents[pid].append(str(a.idx))


    def _get_region_arrays(self, agent_ids):
        coins, utils = [], []
        for aid in agent_ids:
            a = self.get_agent(aid)
            c = a.total_endowment("Coin")
            coins.append(c)
            utils.append(
                rewards.isoelastic_coin_minus_labor(
                    coin_endowment=c,
                    total_labor=a.state["endogenous"]["Labor"],
                    isoelastic_eta=self.isoelastic_eta,
                    labor_coefficient=self.energy_weight * self.energy_cost,
                )
            )
        return np.array(coins, dtype=np.float32), np.array(utils, dtype=np.float32)

    # ------------------------------
    # Reset / Step plumbing
    # ------------------------------
    def reset_starting_layout(self):
        # Reuse SplitLayout -> LayoutFromFile reset logic
        super().reset_starting_layout()

    def reset_agent_states(self):
        # Reuse SplitLayout logic for clearing and placing agents + coin/labor reset
        super().reset_agent_states()

    def additional_reset_steps(self):
        """
        After components reset, finalize:
          - Place/skill logic (SplitLayout already sets build_payment + positions)
          - Build fixed agent->planner mapping by side of water
          - Initialize marginal reward trackers for agents & planners
        """
        super().additional_reset_steps()

        # Build fixed assignments (based on initial row)
        self._assign_agents_to_planners()

        # Initialize marginal reward trackers for all agents + both planners
        # Compute current "objective values" and store them
        curr = self._compute_objectives_snapshot()

        # Extend to include both planners
        for p in self.world.planners:
            pid = str(p.idx)
            coin_endowments, utilities = self._get_region_arrays(self.planner_to_agents[pid])

            if self.planner_reward_type == "coin_eq_times_productivity":
                n = len(coin_endowments)
                eq_weight = 1.0 - self.mixing_weight_gini_vs_coin  # 0.0 for productivity-only
                if n == 0:
                    curr[pid] = 0.0
                elif eq_weight <= 0.0 or n <= 1:
                    # Productivity-only OR too few agents to define Gini: equality := 1, reward = prod/n
                    curr[pid] = float(np.sum(coin_endowments)) / n
                else:
                    curr[pid] = rewards.coin_eq_times_productivity(
                        coin_endowments=coin_endowments,
                        equality_weight=eq_weight,
                    )

            elif self.planner_reward_type == "inv_income_weighted_coin_endowments":
                curr[pid] = rewards.inv_income_weighted_coin_endowments(
                    coin_endowments=coin_endowments
                )
            elif self.planner_reward_type == "inv_income_weighted_utility":
                curr[pid] = rewards.inv_income_weighted_utility(
                    coin_endowments=coin_endowments,
                    utilities=utilities
                )
            else:
                raise NotImplementedError(f"Unknown planner_reward_type: {self.planner_reward_type}")

        # Initialize logs for marginal reward computation
        self.curr_optimization_metric = deepcopy(curr)
        self.init_optimization_metric = deepcopy(curr)
        self.prev_optimization_metric = deepcopy(curr)

    # Helper: snapshot agent utilities (without planner)
    def _compute_objectives_snapshot(self):
        """
        Return dict {agent.idx: utility} at current state.
        Uses isoelastic coin minus labor; matches AI-Economist style.
        """
        d = {}
        for a in self.world.agents:
            d[str(a.idx)] = rewards.isoelastic_coin_minus_labor(
                coin_endowment=a.total_endowment("Coin"),
                total_labor=a.state["endogenous"]["Labor"],
                isoelastic_eta=self.isoelastic_eta,
                labor_coefficient=self.energy_weight * self.energy_cost,
            )
        return d

    # ------------------------------
    # Observations
    # ------------------------------
    def generate_observations(self):
        
        obs = super().generate_observations()

        def _keep(x):
            # Make 2D so the flattener keeps the key
            return np.array([[x]], dtype=np.float32)

        # --- Regional planner summaries (ensure keys survive flattening) ---
        for p in self.world.planners:
            pid = str(p.idx)
            assigned = self.planner_to_agents.get(pid, [])
            if len(assigned) == 0:
                n, avg_coin, eq, prod = 0, 0.0, 1.0, 0.0
            else:
                coin_endowments, _ = self._get_region_arrays(assigned)
                n = len(assigned)
                avg_coin = float(np.mean(coin_endowments)) if n > 0 else 0.0
                if n <= 1:
                    eq = 1.0
                else:
                    eq = float(social_metrics.get_equality(coin_endowments))
                prod = float(social_metrics.get_productivity(coin_endowments)) if n > 0 else 0.0

            if pid not in obs:
                obs[pid] = {}

            # Make values 2D to preserve keys after flattening
            obs[pid]["n_region"] = _keep(n)
            obs[pid]["avg_coin_region"] = _keep(avg_coin)
            obs[pid]["equality_region"] = _keep(eq)
            obs[pid]["productivity_region"] = _keep(prod)

        # Ensure spatial maps are visible under p_top if enabled
        legacy_pid = str(self.world.planner.idx)  # should be "p_top"
        if self._planner_gets_spatial_info and legacy_pid in obs:
            if "p_top" not in obs:
                obs["p_top"] = {}
            # NOTE: BaseEnvironment prefixes scenario obs with "world-".
            # So the spatial maps arrive as "world-map" and "world-idx_map".
            if "world-map" in obs[legacy_pid]:
                obs["p_top"]["world-map"] = obs[legacy_pid]["world-map"]
            if "world-idx_map" in obs[legacy_pid]:
                obs["p_top"]["world-idx_map"] = obs[legacy_pid]["world-idx_map"]

        return obs
    
    # ------------------------------
    # Rewards
    # ------------------------------
    def compute_reward(self):
        """
        Compute **marginal** rewards for:
          - Agents: change in utility (isoelastic coin - labor)
          - Planners: change in **regional SWF** over assigned agents only
        """
        util_last = deepcopy(self.curr_optimization_metric)

        # Recompute agent utilities (current)
        curr = self._compute_objectives_snapshot()

        # Recompute per-planner objective from assigned agents only
        for p in self.world.planners:
            pid = str(p.idx)
            assigned = self.planner_to_agents[pid]
            coin_endowments, utilities = self._get_region_arrays(assigned)

            if self.planner_reward_type == "coin_eq_times_productivity":
                n = len(coin_endowments)
                eq_weight = 1.0 - self.mixing_weight_gini_vs_coin
                if n == 0:
                    curr[pid] = 0.0
                elif eq_weight <= 0.0 or n <= 1:
                    curr[pid] = float(np.sum(coin_endowments)) / n
                else:
                    curr[pid] = rewards.coin_eq_times_productivity(
                        coin_endowments=coin_endowments,
                        equality_weight=eq_weight,
                    )

            elif self.planner_reward_type == "inv_income_weighted_coin_endowments":
                curr[pid] = rewards.inv_income_weighted_coin_endowments(
                    coin_endowments=coin_endowments
                )
            elif self.planner_reward_type == "inv_income_weighted_utility":
                curr[pid] = rewards.inv_income_weighted_utility(
                    coin_endowments=coin_endowments,
                    utilities=utilities
                )
            else:
                raise NotImplementedError

        # Marginal reward
        rew = {k: float(curr[k] - util_last[k]) for k in curr.keys()}

        self.prev_optimization_metric.update(util_last)
        self.curr_optimization_metric = deepcopy(curr)



        # ===== TEMP DEBUG START =====
        assigned_top = self.planner_to_agents["p_top"]
        assigned_bottom = self.planner_to_agents["p_bottom"]
        top_coins, _ = self._get_region_arrays(assigned_top)
        bottom_coins, _ = self._get_region_arrays(assigned_bottom)
        print("[DBG] top ids:", assigned_top, "sum_coin:", float(np.sum(top_coins)))
        print("[DBG] bottom ids:", assigned_bottom, "sum_coin:", float(np.sum(bottom_coins)))
        print("[DBG] rew:", rew)
        # ===== TEMP DEBUG END =====

        return rew

    # ------------------------------
    # Metrics
    # ------------------------------
    
    def scenario_metrics(self):
        metrics = {}

        # Global
        coin = np.array([a.total_endowment("Coin") for a in self.world.agents], dtype=np.float32)
        metrics["global/productivity"] = social_metrics.get_productivity(coin)
        # Guard when the number of agents is too small for Gini (0 or 1)
        metrics["global/equality"] = (
            1.0 if len(coin) <= 1 else social_metrics.get_equality(coin)
        )

        # Regions
        for pid in ("p_top", "p_bottom"):
            assigned = self.planner_to_agents.get(pid, [])
            if len(assigned) == 0:
                metrics[f"{pid}/n"] = 0
                metrics[f"{pid}/productivity"] = 0.0
                metrics[f"{pid}/equality"] = 1.0
            else:
                c, _ = self._get_region_arrays(assigned)
                metrics[f"{pid}/n"] = len(assigned)
                metrics[f"{pid}/productivity"] = social_metrics.get_productivity(c)
                # Guard for n <= 1 to avoid divide-by-zero in Gini
                metrics[f"{pid}/equality"] = (
                    1.0 if len(assigned) <= 1 else social_metrics.get_equality(c)
                )

        return metrics