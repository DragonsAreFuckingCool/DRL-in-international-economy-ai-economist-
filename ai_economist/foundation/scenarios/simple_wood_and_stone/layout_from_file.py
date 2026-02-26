# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy import signal

from ai_economist.foundation.scenarios import BaseEnvironment, scenario_registry
from ai_economist.foundation.scenarios.utils import rewards, social_metrics


@scenario_registry.add
class LayoutFromFile(BaseEnvironment):
    """
    World containing stone and wood with stochastic regeneration. Refers to a fixed
    layout file (see ./map_txt/ for examples) to determine the spatial arrangement of
    stone, wood, and water tiles.

    Args:
        planner_gets_spatial_obs (bool): Whether the planner agent receives spatial
            observations from the world.
        full_observability (bool): Whether the mobile agents' spatial observation
            includes the full world view or is instead an egocentric view.
        mobile_agent_observation_range (int): If not using full_observability,
            the spatial range (on each side of the agent) that is visible in the
            spatial observations.
        env_layout_file (str): Name of the layout file in ./map_txt/ to use.
            Note: The world dimensions of that layout must match the world dimensions
            argument used to construct the environment.
        resource_regen_prob (float): Probability that an empty source tile will
            regenerate a new resource unit.
        fixed_four_skill_and_loc (bool): Whether to use a fixed set of build skills and
            starting locations, with agents grouped into starting locations based on
            which skill quartile they are in. False, by default.
            True, for experiments in https://arxiv.org/abs/2004.13332.
            Note: Requires that the environment uses the "Build" component with
            skill_dist="pareto".
        starting_agent_coin (int, float): Amount of coin agents have at t=0. Defaults
            to zero coin.
        isoelastic_eta (float): Parameter controlling the shape of agent utility
            wrt coin endowment.
        energy_cost (float): Coefficient for converting labor to negative utility.
        energy_warmup_constant (float): Decay constant that controls the rate at which
            the effective energy cost is annealed from 0 to energy_cost. Set to 0
            (default) to disable annealing, meaning that the effective energy cost is
            always energy_cost. The units of the decay constant depend on the choice of
            energy_warmup_method.
        energy_warmup_method (str): How to schedule energy annealing (warmup). If
            "decay" (default), use the number of completed episodes. If "auto",
            use the number of timesteps where the average agent reward was positive.
        planner_reward_type (str): The type of reward used for the planner. Options
            are "coin_eq_times_productivity" (default),
            "inv_income_weighted_coin_endowment", and "inv_income_weighted_utility".
        mixing_weight_gini_vs_coin (float): Degree to which equality is ignored w/
            "coin_eq_times_productivity". Default is 0, which weights equality and
            productivity equally. If set to 1, only productivity is rewarded.
    """

    name = "layout_from_file/simple_wood_and_stone"
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]
    required_entities = ["Wood", "Stone", "Water"]

    def __init__(
        self,
        *base_env_args,
        planner_gets_spatial_info=True,
        full_observability=False,
        mobile_agent_observation_range=5,
        env_layout_file="quadrant_25x25_20each_30clump.txt",
        resource_regen_prob=0.01,
        fixed_four_skill_and_loc=False,
        starting_agent_coin=0,
        isoelastic_eta=0.23,
        energy_cost=0.21,
        energy_warmup_constant=0,
        energy_warmup_method="decay",
        planner_reward_type="coin_eq_times_productivity",
        mixing_weight_gini_vs_coin=0.0,
        agent_start_locs=None,
        **base_env_kwargs,
    ):
        super().__init__(*base_env_args, **base_env_kwargs)

        # Whether agents receive spatial information in their observation tensor
        self._planner_gets_spatial_info = bool(planner_gets_spatial_info)

        # Whether the (non-planner) agents can see the whole world map
        self._full_observability = bool(full_observability)

        self._mobile_agent_observation_range = int(mobile_agent_observation_range)

        self.agent_start_locs = agent_start_locs

        # Load in the layout
        path_to_layout_file = Path(f"{Path(__file__).parent}/map_txt/{env_layout_file}")

        with open(path_to_layout_file, "r") as f:
            self.env_layout_string = f.read()
            self.env_layout = self.env_layout_string.split(";")

        # Convert the layout to landmark maps
        landmark_lookup = {"W": "Wood", "S": "Stone", "@": "Water"}
        self._source_maps = {
            r: np.zeros(self.world_size) for r in landmark_lookup.values()
        }
        for r, symbol_row in enumerate(self.env_layout):
            for c, symbol in enumerate(symbol_row):
                landmark = landmark_lookup.get(symbol, None)
                if landmark:
                    self._source_maps[landmark][r, c] = 1

        # For controlling how resource regeneration behavior
        self.layout_specs = dict(
            Wood={
                "regen_weight": float(resource_regen_prob),
                "regen_halfwidth": 0,
                "max_health": 1,
            },
            Stone={
                "regen_weight": float(resource_regen_prob),
                "regen_halfwidth": 0,
                "max_health": 1,
            },
        )
        assert 0 <= self.layout_specs["Wood"]["regen_weight"] <= 1
        assert 0 <= self.layout_specs["Stone"]["regen_weight"] <= 1

        # How much coin do agents begin with at upon reset
        self.starting_agent_coin = float(starting_agent_coin)
        assert self.starting_agent_coin >= 0.0

        # Controls the diminishing marginal utility of coin.
        # isoelastic_eta=0 means no diminishing utility.
        self.isoelastic_eta = float(isoelastic_eta)
        assert 0.0 <= self.isoelastic_eta <= 1.0

        # The amount that labor is weighted in utility computation
        # (once annealing is finished)
        self.energy_cost = float(energy_cost)
        assert self.energy_cost >= 0

        # Which method to use for calculating the progress of energy annealing
        # If method = 'decay': #completed episodes
        # If method = 'auto' : #timesteps where avg. agent reward > 0
        self.energy_warmup_method = energy_warmup_method.lower()
        assert self.energy_warmup_method in ["decay", "auto"]
        # Decay constant for annealing to full energy cost
        # (if energy_warmup_constant == 0, there is no annealing)
        self.energy_warmup_constant = float(energy_warmup_constant)
        assert self.energy_warmup_constant >= 0
        self._auto_warmup_integrator = 0

        # Which social welfare function to use
        self.planner_reward_type = str(planner_reward_type).lower()

        # How much to weight equality if using SWF=eq*prod:
        # 0 -> SWF=eq * prod
        # 1 -> SWF=prod
        self.mixing_weight_gini_vs_coin = float(mixing_weight_gini_vs_coin)
        assert 0 <= self.mixing_weight_gini_vs_coin <= 1.0

        # Use this to calculate marginal changes and deliver that as reward
        self.init_optimization_metric = {agent.idx: 0 for agent in self.all_agents}
        self.prev_optimization_metric = {agent.idx: 0 for agent in self.all_agents}
        self.curr_optimization_metric = {agent.idx: 0 for agent in self.all_agents}

        """
        Fixed Four Skill and Loc
        ------------------------
        """
        self.agent_starting_pos = {agent.idx: [] for agent in self.world.agents}

        self.fixed_four_skill_and_loc = bool(fixed_four_skill_and_loc)
        if self.fixed_four_skill_and_loc:
            bm = self.get_component("Build")
            assert bm.skill_dist == "pareto"
            pmsm = bm.payment_max_skill_multiplier

            # Temporarily switch to a fixed seed for controlling randomness
            seed_state = np.random.get_state()
            np.random.seed(seed=1)

            # Generate a batch (100000) of num_agents (sorted/clipped) Pareto samples.
            pareto_samples = np.random.pareto(4, size=(100000, self.n_agents))
            clipped_skills = np.minimum(pmsm, (pmsm - 1) * pareto_samples + 1)
            sorted_clipped_skills = np.sort(clipped_skills, axis=1)
            # The skill level of the i-th skill-ranked agent is the average of the
            # i-th ranked samples throughout the batch.
            average_ranked_skills = sorted_clipped_skills.mean(axis=0)
            self._avg_ranked_skill = average_ranked_skills * bm.payment

            np.random.set_state(seed_state)

            # Fill in the starting location associated with each skill rank
            starting_ranked_locs = [
                # Worst group of agents goes in top right
                (0, self.world_size[1] - 1),
                # Second-worst group of agents goes in bottom left
                (self.world_size[0] - 1, 0),
                # Second-best group of agents goes in top left
                (0, 0),
                # Best group of agents goes in bottom right
                (self.world_size[1] - 1, self.world_size[1] - 1),
            ]
            self._ranked_locs = []

            # Based on skill, assign each agent to one of the location groups
            skill_groups = np.floor(
                np.arange(self.n_agents) * (4 / self.n_agents),
            ).astype(np.int32)
            n_in_group = np.zeros(4, dtype=np.int32)
            for g in skill_groups:
                # The position within the group is given by the number of agents
                # counted in the group thus far.
                g_pos = n_in_group[g]

                # Top right
                if g == 0:
                    r = starting_ranked_locs[g][0] + (g_pos // 4)
                    c = starting_ranked_locs[g][1] - (g_pos % 4)
                    self._ranked_locs.append((r, c))

                # Bottom left
                elif g == 1:
                    r = starting_ranked_locs[g][0] - (g_pos // 4)
                    c = starting_ranked_locs[g][1] + (g_pos % 4)
                    self._ranked_locs.append((r, c))

                # Top left
                elif g == 2:
                    r = starting_ranked_locs[g][0] + (g_pos // 4)
                    c = starting_ranked_locs[g][1] + (g_pos % 4)
                    self._ranked_locs.append((r, c))

                # Bottom right
                elif g == 3:
                    r = starting_ranked_locs[g][0] - (g_pos // 4)
                    c = starting_ranked_locs[g][1] - (g_pos % 4)
                    self._ranked_locs.append((r, c))

                else:
                    raise ValueError

                # Count the agent we just placed.
                n_in_group[g] = n_in_group[g] + 1

    @property
    def energy_weight(self):
        """
        Energy annealing progress. Multiply with self.energy_cost to get the
        effective energy coefficient.
        """
        if self.energy_warmup_constant <= 0.0:
            return 1.0

        if self.energy_warmup_method == "decay":
            return float(1.0 - np.exp(-self._completions / self.energy_warmup_constant))

        if self.energy_warmup_method == "auto":
            return float(
                1.0
                - np.exp(-self._auto_warmup_integrator / self.energy_warmup_constant)
            )

        raise NotImplementedError

    def get_current_optimization_metrics(self):
        """
        Compute optimization metrics based on the current state. Used to compute reward.

        Returns:
            curr_optimization_metric (dict): A dictionary of {agent.idx: metric}
                with an entry for each agent (including the planner) in the env.
        """
        curr_optimization_metric = {}
        # (for agents)
        for agent in self.world.agents:
            curr_optimization_metric[agent.idx] = rewards.isoelastic_coin_minus_labor(
                coin_endowment=agent.total_endowment("Coin"),
                total_labor=agent.state["endogenous"]["Labor"],
                isoelastic_eta=self.isoelastic_eta,
                labor_coefficient=self.energy_weight * self.energy_cost,
            )
        # (for the planner)
        if self.planner_reward_type == "coin_eq_times_productivity":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.coin_eq_times_productivity(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                ),
                equality_weight=1 - self.mixing_weight_gini_vs_coin,
            )
        elif self.planner_reward_type == "inv_income_weighted_coin_endowments":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.inv_income_weighted_coin_endowments(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                )
            )
        elif self.planner_reward_type == "inv_income_weighted_utility":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.inv_income_weighted_utility(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                ),
                utilities=np.array(
                    [curr_optimization_metric[agent.idx] for agent in self.world.agents]
                ),
            )
        else:
            print("No valid planner reward selected!")
            raise NotImplementedError
        return curr_optimization_metric

    # The following methods must be implemented for each scenario
    # -----------------------------------------------------------

    def reset_starting_layout(self):
        """
        Part 1/2 of scenario reset. This method handles resetting the state of the
        environment managed by the scenario (i.e. resource & landmark layout).

        Here, reset to the layout in the fixed layout file
        """
        self.world.maps.clear()
        for landmark, landmark_map in self._source_maps.items():
            self.world.maps.set(landmark, landmark_map)
            if landmark in ["Stone", "Wood"]:
                self.world.maps.set(landmark + "SourceBlock", landmark_map)

    def reset_agent_states(self):
        """
        Part 2/2 of scenario reset. This method handles resetting the state of the
        agents themselves (i.e. inventory, locations, etc.).

        Here, empty inventories and place mobile agents in random, accessible
        locations to start. Note: If using fixed_four_skill_and_loc, the starting
        locations will be overridden in self.additional_reset_steps.
        """
        self.world.clear_agent_locs()
        for agent in self.world.agents:
            agent.state["inventory"] = {k: 0 for k in agent.inventory.keys()}
            agent.state["escrow"] = {k: 0 for k in agent.inventory.keys()}
            agent.state["endogenous"] = {k: 0 for k in agent.endogenous.keys()}
            # Add starting coin
            agent.state["inventory"]["Coin"] = float(self.starting_agent_coin)

        self.world.planner.state["inventory"] = {
            k: 0 for k in self.world.planner.inventory.keys()
        }
        self.world.planner.state["escrow"] = {
            k: 0 for k in self.world.planner.escrow.keys()
        }

        for agent in self.world.agents:
            r = np.random.randint(0, self.world_size[0])
            c = np.random.randint(0, self.world_size[1])
            n_tries = 0
            while not self.world.can_agent_occupy(r, c, agent):
                r = np.random.randint(0, self.world_size[0])
                c = np.random.randint(0, self.world_size[1])
                n_tries += 1
                if n_tries > 200:
                    raise TimeoutError
            r, c = self.world.set_agent_loc(agent, r, c)

    def scenario_step(self):
        """
        Update the state of the world according to whatever rules this scenario
        implements.

        This gets called in the 'step' method (of base_env) after going through each
        component step and before generating observations, rewards, etc.

        In this class of scenarios, the scenario step handles stochastic resource
        regeneration.
        """

        resources = ["Wood", "Stone"]

        for resource in resources:
            d = 1 + (2 * self.layout_specs[resource]["regen_halfwidth"])
            kernel = (
                self.layout_specs[resource]["regen_weight"] * np.ones((d, d)) / (d ** 2)
            )

            resource_map = self.world.maps.get(resource)
            resource_source_blocks = self.world.maps.get(resource + "SourceBlock")
            spawnable = (
                self.world.maps.empty + resource_map + resource_source_blocks
            ) > 0
            spawnable *= resource_source_blocks > 0

            health = np.maximum(resource_map, resource_source_blocks)
            respawn = np.random.rand(*health.shape) < signal.convolve2d(
                health, kernel, "same"
            )
            respawn *= spawnable

            self.world.maps.set(
                resource,
                np.minimum(
                    resource_map + respawn, self.layout_specs[resource]["max_health"]
                ),
            )

    def generate_observations(self):
        """
        Generate observations associated with this scenario.

        A scenario does not need to produce observations and can provide observations
        for only some agent types; however, for a given agent type, it should either
        always or never yield an observation. If it does yield an observation,
        that observation should always have the same structure/sizes!

        Returns:
            obs (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent (which can including
                the planner) for which this scenario provides an observation. For each
                entry, the key specifies the index of the agent and the value contains
                its associated observation dictionary.

        Here, non-planner agents receive spatial observations (depending on the env
        config) as well as the contents of their inventory and endogenous quantities.
        The planner also receives spatial observations (again, depending on the env
        config) as well as the inventory of each of the mobile agents.
        """
        obs = {}
        curr_map = self.world.maps.state

        owner_map = self.world.maps.owner_state
        loc_map = self.world.loc_map
        agent_idx_maps = np.concatenate([owner_map, loc_map[None, :, :]], axis=0)
        agent_idx_maps += 2
        agent_idx_maps[agent_idx_maps == 1] = 0

        agent_locs = {
            str(agent.idx): {
                "loc-row": agent.loc[0] / self.world_size[0],
                "loc-col": agent.loc[1] / self.world_size[1],
            }
            for agent in self.world.agents
        }
        agent_invs = {
            str(agent.idx): {
                "inventory-" + k: v * self.inv_scale for k, v in agent.inventory.items()
            }
            for agent in self.world.agents
        }

        obs[self.world.planner.idx] = {
            "inventory-" + k: v * self.inv_scale
            for k, v in self.world.planner.inventory.items()
        }
        if self._planner_gets_spatial_info:
            obs[self.world.planner.idx].update(
                dict(map=curr_map, idx_map=agent_idx_maps)
            )

        # Mobile agents see the full map. Convey location info via one-hot map channels.
        if self._full_observability:
            for agent in self.world.agents:
                my_map = np.array(agent_idx_maps)
                my_map[my_map == int(agent.idx) + 2] = 1
                sidx = str(agent.idx)
                obs[sidx] = {"map": curr_map, "idx_map": my_map}
                obs[sidx].update(agent_invs[sidx])

        # Mobile agents only see within a window around their position
        else:
            w = (
                self._mobile_agent_observation_range
            )  # View halfwidth (only applicable without full observability)

            padded_map = np.pad(
                curr_map,
                [(0, 1), (w, w), (w, w)],
                mode="constant",
                constant_values=[(0, 1), (0, 0), (0, 0)],
            )

            padded_idx = np.pad(
                agent_idx_maps,
                [(0, 0), (w, w), (w, w)],
                mode="constant",
                constant_values=[(0, 0), (0, 0), (0, 0)],
            )

            for agent in self.world.agents:
                r, c = [c + w for c in agent.loc]
                visible_map = padded_map[
                    :, (r - w) : (r + w + 1), (c - w) : (c + w + 1)
                ]
                visible_idx = np.array(
                    padded_idx[:, (r - w) : (r + w + 1), (c - w) : (c + w + 1)]
                )

                visible_idx[visible_idx == int(agent.idx) + 2] = 1

                sidx = str(agent.idx)

                obs[sidx] = {"map": visible_map, "idx_map": visible_idx}
                obs[sidx].update(agent_locs[sidx])
                obs[sidx].update(agent_invs[sidx])

                # Agent-wise planner info (gets crunched into the planner obs in the
                # base scenario code)
                obs["p" + sidx] = agent_invs[sidx]
                if self._planner_gets_spatial_info:
                    obs["p" + sidx].update(agent_locs[sidx])

        return obs

    def compute_reward(self):
        """
        Apply the reward function(s) associated with this scenario to get the rewards
        from this step.

        Returns:
            rew (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent in the environment
                (including the planner). For each entry, the key specifies the index of
                the agent and the value contains the scalar reward earned this timestep.

        Rewards are computed as the marginal utility (agents) or marginal social
        welfare (planner) experienced on this timestep. Ignoring discounting,
        this means that agents' (planner's) objective is to maximize the utility
        (social welfare) associated with the terminal state of the episode.
        """

        # "curr_optimization_metric" hasn't been updated yet, so it gives us the
        # utility from the last step.
        utility_at_end_of_last_time_step = deepcopy(self.curr_optimization_metric)

        # compute current objectives and store the values
        self.curr_optimization_metric = self.get_current_optimization_metrics()

        # reward = curr - prev objectives
        rew = {
            k: float(v - utility_at_end_of_last_time_step[k])
            for k, v in self.curr_optimization_metric.items()
        }

        # store the previous objective values
        self.prev_optimization_metric.update(utility_at_end_of_last_time_step)

        # Automatic Energy Cost Annealing
        # -------------------------------
        avg_agent_rew = np.mean([rew[a.idx] for a in self.world.agents])
        # Count the number of timesteps where the avg agent reward was > 0
        if avg_agent_rew > 0:
            self._auto_warmup_integrator += 1

        return rew

    # Optional methods for customization
    # ----------------------------------

    def additional_reset_steps(self):
        """
        Extra scenario-specific steps that should be performed at the end of the reset
        cycle.

        For each reset cycle...
            First, reset_starting_layout() and reset_agent_states() will be called.

            Second, <component>.reset() will be called for each registered component.

            Lastly, this method will be called to allow for any final customization of
            the reset cycle.

        For this scenario, this method resets optimization metric trackers. If using
        fixed_four_skill_and_loc, this is where each agent gets assigned to one of
        the four fixed skill/loc combinations. The agent-->skill/loc assignment is
        permuted so that all four skill/loc combinations are used.
        """
        # --- CUSTOM STARTING LOCATIONS (minimal safe patch) ---
        # If user provided custom start positions, apply them here and override
        # any default/random/fixed-four placement.
        if hasattr(self, "agent_start_locs") and self.agent_start_locs is not None:
            # Validate provided list
            if len(self.agent_start_locs) != self.n_agents:
                raise ValueError(
                    f"agent_start_locs must have {self.n_agents} entries, "
                    f"got {len(self.agent_start_locs)}"
                )

            self.world.clear_agent_locs()
            H, W = self.world_size

            for agent, (r, c) in zip(self.world.agents, self.agent_start_locs):
                r = int(r)
                c = int(c)

                # Validate bounds
                if not (0 <= r < H and 0 <= c < W):
                    raise ValueError(
                        f"Custom start position {(r,c)} out of bounds for world size {self.world_size}"
                    )

                # Check if cell is occupiable
                if not self.world.can_agent_occupy(r, c, agent):
                    raise ValueError(
                        f"Cannot place agent at specified location {(r,c)}; "
                        "cell is blocked or not occupiable."
                    )

                self.world.set_agent_loc(agent, r, c)

            # Recompute objective trackers and return early
            curr_optimization_metric = self.get_current_optimization_metrics()
            self.curr_optimization_metric = deepcopy(curr_optimization_metric)
            self.init_optimization_metric = deepcopy(curr_optimization_metric)
            self.prev_optimization_metric = deepcopy(curr_optimization_metric)
            return

        if self.fixed_four_skill_and_loc:
            self.world.clear_agent_locs()
            for i, agent in enumerate(self.world.get_random_order_agents()):
                self.world.set_agent_loc(agent, *self._ranked_locs[i])
                agent.state["build_payment"] = self._avg_ranked_skill[i]

        # compute current objectives
        curr_optimization_metric = self.get_current_optimization_metrics()

        self.curr_optimization_metric = deepcopy(curr_optimization_metric)
        self.init_optimization_metric = deepcopy(curr_optimization_metric)
        self.prev_optimization_metric = deepcopy(curr_optimization_metric)

    def scenario_metrics(self):
        """
        Allows the scenario to generate metrics (collected along with component metrics
        in the 'metrics' property).

        To have the scenario add metrics, this function needs to return a dictionary of
        {metric_key: value} where 'value' is a scalar (no nesting or lists!)

        Here, summarize social metrics, endowments, utilities, and labor cost annealing.
        """
        metrics = dict()

        coin_endowments = np.array(
            [agent.total_endowment("Coin") for agent in self.world.agents]
        )
        metrics["social/productivity"] = social_metrics.get_productivity(
            coin_endowments
        )
        metrics["social/equality"] = social_metrics.get_equality(coin_endowments)

        utilities = np.array(
            [self.curr_optimization_metric[agent.idx] for agent in self.world.agents]
        )
        metrics[
            "social_welfare/coin_eq_times_productivity"
        ] = rewards.coin_eq_times_productivity(
            coin_endowments=coin_endowments, equality_weight=1.0
        )
        metrics[
            "social_welfare/inv_income_weighted_coin_endow"
        ] = rewards.inv_income_weighted_coin_endowments(coin_endowments=coin_endowments)
        metrics[
            "social_welfare/inv_income_weighted_utility"
        ] = rewards.inv_income_weighted_utility(
            coin_endowments=coin_endowments, utilities=utilities
        )

        for agent in self.all_agents:
            for resource, quantity in agent.inventory.items():
                metrics[
                    "endow/{}/{}".format(agent.idx, resource)
                ] = agent.total_endowment(resource)

            if agent.endogenous is not None:
                for resource, quantity in agent.endogenous.items():
                    metrics["endogenous/{}/{}".format(agent.idx, resource)] = quantity

            metrics["util/{}".format(agent.idx)] = self.curr_optimization_metric[
                agent.idx
            ]

        # Labor weight
        metrics["labor/weighted_cost"] = self.energy_cost * self.energy_weight
        metrics["labor/warmup_integrator"] = int(self._auto_warmup_integrator)

        return metrics


@scenario_registry.add
class SplitLayout(LayoutFromFile):
    """
    Extends layout_from_file/simple_wood_and_stone to impose a row of water midway
    through the map, uses a fixed set of pareto-distributed building skills (requires a
    Build component), and places agents in the top/bottom depending on skill rank.

    Args:
        water_row (int): Row of the map where the water barrier is placed. Defaults
        to half the world height.
        skill_rank_of_top_agents (int, float, tuple, list): Index/indices specifying
            which agent(s) to place in the top of the map. Indices refer to the skill
            ranking, with 0 referring to the highest-skilled agent. Defaults to only
            the highest-skilled agent in the top.
        planner_gets_spatial_obs (bool): Whether the planner agent receives spatial
            observations from the world.
        full_observability (bool): Whether the mobile agents' spatial observation
            includes the full world view or is instead an egocentric view.
        mobile_agent_observation_range (int): If not using full_observability,
            the spatial range (on each side of the agent) that is visible in the
            spatial observations.
        env_layout_file (str): Name of the layout file in ./map_txt/ to use.
            Note: The world dimensions of that layout must match the world dimensions
            argument used to construct the environment.
        resource_regen_prob (float): Probability that an empty source tile will
            regenerate a new resource unit.
        starting_agent_coin (int, float): Amount of coin agents have at t=0. Defaults
            to zero coin.
        isoelastic_eta (float): Parameter controlling the shape of agent utility
            wrt coin endowment.
        energy_cost (float): Coefficient for converting labor to negative utility.
        energy_warmup_constant (float): Decay constant that controls the rate at which
            the effective energy cost is annealed from 0 to energy_cost. Set to 0
            (default) to disable annealing, meaning that the effective energy cost is
            always energy_cost. The units of the decay constant depend on the choice of
            energy_warmup_method.
        energy_warmup_method (str): How to schedule energy annealing (warmup). If
            "decay" (default), use the number of completed episodes. If "auto",
            use the number of timesteps where the average agent reward was positive.
        planner_reward_type (str): The type of reward used for the planner. Options
            are "coin_eq_times_productivity" (default),
            "inv_income_weighted_coin_endowment", and "inv_income_weighted_utility".
        mixing_weight_gini_vs_coin (float): Degree to which equality is ignored w/
            "coin_eq_times_productivity". Default is 0, which weights equality and
            productivity equally. If set to 1, only productivity is rewarded.
    """

    name = "split_layout/simple_wood_and_stone"

    def __init__(
        self,
        *args,
        water_row=None,
        skill_rank_of_top_agents=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.fixed_four_skill_and_loc:
            raise ValueError(
                "The split layout scenario does not support "
                "fixed_four_skill_and_loc. Set this to False."
            )

        # Augment the fixed layout to include a row of water through the middle
        # --- REPLACE the entire augmentation block in SplitLayout.__init__ with this ---

        # Decide barrier row
        if water_row is None:
            self._water_line = self.world_size[0] // 2
        else:
            self._water_line = int(water_row)
            assert 0 < self._water_line < self.world_size[0] - 1

        # 1) Start with a CLEAN water map (removes any water from the layout file)
        clean_water = np.zeros(self.world_size, dtype=np.int32)
        # 2) Draw only the horizontal barrier line
        clean_water[self._water_line, :] = 1
        self._source_maps["Water"] = clean_water

        # 3) Ensure no Wood/Stone sits on the barrier row
        for landmark, landmark_map in self._source_maps.items():
            if landmark in ["Wood", "Stone"]:
                m = np.array(landmark_map, copy=True)
                m[self._water_line, :] = 0
                self._source_maps[landmark] = m

        # Controls logic for which agents (by skill rank) get placed on the top
        if skill_rank_of_top_agents is None:
            skill_rank_of_top_agents = [0]

        if isinstance(skill_rank_of_top_agents, (int, float)):
            self.skill_rank_of_top_agents = [int(skill_rank_of_top_agents)]
        elif isinstance(skill_rank_of_top_agents, (tuple, list)):
            self.skill_rank_of_top_agents = list(set(skill_rank_of_top_agents))
        else:
            raise TypeError(
                "skill_rank_of_top_agents must be a scalar "
                "index, or a list of scalar indices."
            )
        for rank in self.skill_rank_of_top_agents:
            assert 0 <= rank < self.n_agents
        assert 0 < len(self.skill_rank_of_top_agents) < self.n_agents

        # Set the skill associated with each skill rank
        bm = self.get_component("Build")
        assert bm.skill_dist == "pareto"
        pmsm = bm.payment_max_skill_multiplier
        # Generate a batch (100000) of num_agents (sorted/clipped) Pareto samples.
        pareto_samples = np.random.pareto(4, size=(100000, self.n_agents))
        clipped_skills = np.minimum(pmsm, (pmsm - 1) * pareto_samples + 1)
        sorted_clipped_skills = np.sort(clipped_skills, axis=1)
        # The skill level of the i-th skill-ranked agent is the average of the
        # i-th ranked samples throughout the batch.
        average_ranked_skills = sorted_clipped_skills.mean(axis=0)
        self._avg_ranked_skill = average_ranked_skills * bm.payment
        # Reverse the order so index 0 is the highest-skilled
        self._avg_ranked_skill = self._avg_ranked_skill[::-1]

    def additional_reset_steps(self):
        """
        Extra scenario-specific steps that should be performed at the end of the reset
        cycle.

        For each reset cycle...
            First, reset_starting_layout() and reset_agent_states() will be called.

            Second, <component>.reset() will be called for each registered component.

            Lastly, this method will be called to allow for any final customization of
            the reset cycle.

        For this scenario, this method resets optimization metric trackers. This is
        where each agent gets assigned to one of the skills and the starting
        locations are reset according to self.skill_rank_of_top_agents.
        """
        self.world.clear_agent_locs()
        for i, agent in enumerate(self.world.get_random_order_agents()):
            agent.state["build_payment"] = self._avg_ranked_skill[i]
            if i in self.skill_rank_of_top_agents:
                r_min, r_max = 0, self._water_line
            else:
                r_min, r_max = self._water_line + 1, self.world_size[0]

            r = np.random.randint(r_min, r_max)
            c = np.random.randint(0, self.world_size[1])
            n_tries = 0
            while not self.world.can_agent_occupy(r, c, agent):
                r = np.random.randint(r_min, r_max)
                c = np.random.randint(0, self.world_size[1])
                n_tries += 1
                if n_tries > 200:
                    raise TimeoutError
            self.world.set_agent_loc(agent, r, c)

        # compute current objectives
        curr_optimization_metric = self.get_current_optimization_metrics()

        self.curr_optimization_metric = deepcopy(curr_optimization_metric)
        self.init_optimization_metric = deepcopy(curr_optimization_metric)
        self.prev_optimization_metric = deepcopy(curr_optimization_metric)



from copy import deepcopy
import numpy as np
from ai_economist.foundation.scenarios import scenario_registry
# LayoutFromFile must be defined above in this module

@scenario_registry.add
class CustomSplitOverlayFromFile(LayoutFromFile):
    """
    Overlay-only split on top of file map:
      - Preserve all water from the file.
      - Overlay a horizontal water line at `water_row`.
      - Optional vertical line (off by default).
      - Optional gaps in the horizontal line so the map is not closed off.
      - Honors agent_start_locs (if provided) and multi-top skill ranks.
    """
    name = "custom/split_overlay_from_file"

    def __init__(
        self,
        *args,
        water_row=None,
        add_vertical_line=False,          # default OFF
        vertical_col=None,
        skill_rank_of_top_agents=None,
        preserve_file_water=True,         # keep file lakes/ponds
        horizontal_gap_cols=None,         # e.g. [12, 25, 38]
        gap_width=3,                      # width of each gap in cols
        random_gap_count=0,               # if >0, add N random gaps
        random_gap_seed=None,             # for reproducibility
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        H, W = self.world_size
        if water_row is None:
            water_row = H // 2
        water_row = int(water_row)
        assert 0 <= water_row < H, "water_row must be within map bounds"

        # ---- Start from file water (preserve) or from empty water map ----
        if "Water" not in self._source_maps:
            self._source_maps["Water"] = np.zeros(self.world_size, dtype=np.int32)

        if preserve_file_water:
            water_map = np.array(self._source_maps["Water"], copy=True)
        else:
            water_map = np.zeros(self.world_size, dtype=np.int32)

        # ---- Overlay horizontal barrier (full row first) ----
        water_map[water_row, :] = 1

        # ---- Carve gaps in the horizontal barrier ----
        if horizontal_gap_cols is None:
            horizontal_gap_cols = []
        if random_gap_count and random_gap_count > 0:
            rng = np.random.RandomState(random_gap_seed)
            # avoid edges so gaps are visible and valid
            candidate_cols = list(range(2, W - 2))
            rng.shuffle(candidate_cols)
            horizontal_gap_cols.extend(candidate_cols[:int(random_gap_count)])

        gap_w = max(1, int(gap_width))
        half_left = gap_w // 2
        half_right = gap_w - half_left
        for c in horizontal_gap_cols:
            c0 = max(0, int(c) - half_left)
            c1 = min(W, int(c) + half_right)
            water_map[water_row, c0:c1] = 0  # open water -> no barrier here

        # ---- Optional vertical line (you said you don't want it now) ----
        if add_vertical_line:
            if vertical_col is None:
                vertical_col = W // 2
            vertical_col = int(vertical_col)
            assert 0 <= vertical_col < W, "vertical_col must be within bounds"
            water_map[:, vertical_col] = 1
            # (If you also want vertical gaps: do similar carving on `water_map[:, vertical_col]`)

        # ---- Push water map back ----
        self._source_maps["Water"] = water_map

        # Avoid stone/wood on top of overlayed lines (but keep file lakes everywhere else)
        for landmark in ["Wood", "Stone"]:
            if landmark in self._source_maps:
                m = np.array(self._source_maps[landmark], copy=True)
                m[water_row, :] = 0
                if add_vertical_line:
                    m[:, vertical_col] = 0
                self._source_maps[landmark] = m

        # ---------- Top-half by skill rank (unchanged from your custom class) ----------
        if skill_rank_of_top_agents is None:
            skill_rank_of_top_agents = [0]
        if isinstance(skill_rank_of_top_agents, (int, float)):
            self.skill_rank_of_top_agents = [int(skill_rank_of_top_agents)]
        elif isinstance(skill_rank_of_top_agents, (tuple, list)):
            self.skill_rank_of_top_agents = list(dict.fromkeys(skill_rank_of_top_agents))
        else:
            raise TypeError("skill_rank_of_top_agents must be scalar or list/tuple.")
        for rank in self.skill_rank_of_top_agents:
            assert 0 <= int(rank) < self.n_agents, \
                f"Skill rank {rank} must be in [0, {self.n_agents-1}]"

        # Pre-compute skills (same logic as before)
        bm = self.get_component("Build")
        assert bm is not None, "Build component required for skill-based placement."
        pmsm = getattr(bm, "payment_max_skill_multiplier", 1)
        if getattr(bm, "skill_dist", None) == "pareto":
            pareto_samples = np.random.pareto(4, size=(100_000, self.n_agents))
            clipped = np.minimum(pmsm, (pmsm - 1) * pareto_samples + 1)
            avg_sorted = np.sort(clipped, axis=1).mean(axis=0)
            self._avg_ranked_skill = avg_sorted[::-1] * bm.payment
        else:
            self._avg_ranked_skill = np.array([bm.payment] * self.n_agents, dtype=float)

        # Save overlay positions we used for placement in additional_reset_steps
        self._overlay_water_row = water_row
        self._overlay_vertical = add_vertical_line
        self._overlay_vertical_col = int(vertical_col) if add_vertical_line else None

    def additional_reset_steps(self):
        """
        Placement order:
            1) If LayoutFromFile supports agent_start_locs and it's provided -> use it (and return).
            2) Else if fixed_four_skill_and_loc -> defer to parent logic.
            3) Else -> place using skill_rank_of_top_agents (top/bottom halves).
        """
        # 1) Defer to LayoutFromFile custom starts if present (your earlier patch)
        if hasattr(self, "agent_start_locs") and self.agent_start_locs is not None:
            return LayoutFromFile.additional_reset_steps(self)

        # 2) Respect fixed-four logic from LayoutFromFile if enabled
        if getattr(self, "fixed_four_skill_and_loc", False):
            return LayoutFromFile.additional_reset_steps(self)

        # 3) Place by skill rank: TOP vs BOTTOM halves
        self.world.clear_agent_locs()

        # Random order breaks ties; then assign ranked skills and positions
        for i, agent in enumerate(self.world.get_random_order_agents()):
            # Assign the pre-computed skill for rank i
            agent.state["build_payment"] = float(self._avg_ranked_skill[i])

            # Decide which half this rank belongs to
            if i in self.skill_rank_of_top_agents:
                r_min, r_max = 0, self._overlay_water_row
            else:
                r_min, r_max = self._overlay_water_row + 1, self.world_size[0]

            # Sample a valid spot in that half
            r = np.random.randint(r_min, r_max)
            c = np.random.randint(0, self.world_size[1])

            # Avoid water/blocked tiles
            n_tries = 0
            while not self.world.can_agent_occupy(r, c, agent):
                r = np.random.randint(r_min, r_max)
                c = np.random.randint(0, self.world_size[1])
                n_tries += 1
                if n_tries > 200:
                    raise TimeoutError("Could not place agent in the desired half.")
            self.world.set_agent_loc(agent, r, c)

        # Update optimization metrics
        curr_optimization_metric = self.get_current_optimization_metrics()
        self.curr_optimization_metric = deepcopy(curr_optimization_metric)
        self.init_optimization_metric = deepcopy(curr_optimization_metric)
        self.prev_optimization_metric = deepcopy(curr_optimization_metric)
