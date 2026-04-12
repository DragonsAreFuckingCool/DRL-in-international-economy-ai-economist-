from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)

import random
@component_registry.add
class CrossWaterTravel(BaseComponent):

    name = "CrossWaterTravel"
    required_entities = ["Coin", "Labor"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *args,
        travel_cost_coin=5.0,
        travel_cost_labor=0.0,
        cooldown=10,
        allow_only_agent=None,
        enabled=True,
        debug=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.travel_cost_coin = float(travel_cost_coin)
        self.travel_cost_labor = float(travel_cost_labor)
        self.cooldown = int(cooldown)
        self.allow_only_agent = allow_only_agent
        self.enabled = bool(enabled)
        self.debug = bool(debug)

        self.successful_travelers = set()
        self.travel_log = []

        self.agent_home_by_id = {
            0: {"top": (0, 0),   "bottom": (26, 0)},
            1: {"top": (24, 0),  "bottom": (50, 0)},
            2: {"top": (0, 24),  "bottom": (26, 24)},
            3: {"top": (24, 24), "bottom": (50, 24)},
            4: {"top": (0, 0),   "bottom": (26, 0)},
            5: {"top": (24, 0),  "bottom": (50, 0)},
            6: {"top": (0, 24),  "bottom": (26, 24)},
            7: {"top": (24, 24), "bottom": (50, 24)},
        }

        # start-location <-> travel-target mapping
        # self.travel_pairs = {
        #     (0, 0): (26, 0),
        #     (24, 0): (50, 0),
        #     (0, 24): (26, 24),
        #     (24, 24): (50, 24),

        #     (26, 0): (0, 0),
        #     (50, 0): (24, 0),
        #     (26, 24): (0, 24),
        #     (50, 24): (24, 24),
        # }

        # cache each agent's original start location
        self.agent_start_locs = {}

        if self.debug:
            print(
                f"[CrossWaterTravel INIT] enabled={self.enabled}, "
                f"travel_cost_coin={self.travel_cost_coin}, "
                f"travel_cost_labor={self.travel_cost_labor}, cooldown={self.cooldown}, "
                f"allow_only_agent={self.allow_only_agent}"
            )

    # TRAVEL LOCATION HELPERS         
    def _get_travel_target_for_agent(self, agent):
        aid = int(agent.idx)
        homes = self.agent_home_by_id.get(aid, None)
        if homes is None:
            return None

        current_region = agent.state["region"]

        if current_region == "top":
            return homes["bottom"]
        elif current_region == "bottom":
            return homes["top"]

        return None

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return 1
        return None

    def get_additional_state_fields(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return {
                "travel_cooldown": 0,
                "region": "top", 
            }
        return {}
    
    def _region_from_row(self, row):
        waterline = self.world.world_size[0] // 2
        return "top" if int(row) < waterline else "bottom"

    def component_step(self):

        self.successful_travelers = set()

        if not self.enabled:
            return

        world = self.world
        waterline = world.world_size[0] // 2

        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            if action == 0:
                continue

            if self.allow_only_agent is not None and agent.idx != self.allow_only_agent:
                continue

            if agent.state["travel_cooldown"] > 0:
                continue

            coin_before = float(agent.inventory["Coin"])
            if coin_before < self.travel_cost_coin:
                continue

            old_r, old_c = agent.loc

            target_pair = self._get_travel_target_for_agent(agent)
            if self.debug:
                print(
                    f"[TARGET MAP] agent={int(agent.idx)} "
                    f"cached_start={self.agent_start_locs.get(int(agent.idx), None)} "
                    f"target_pair={target_pair}"
                )

            if target_pair is None:
                if self.debug:
                    print(f"[TRAVEL FAIL] Agent {agent.idx} | no mapped travel target")
                continue

            desired_r, desired_c = target_pair

            target = self._find_valid_target(agent, desired_r, desired_c, max_radius=6)
            if target is None:
                if self.debug:
                    print(
                        f"[TRAVEL FAIL] Agent {agent.idx} | no valid target near {(desired_r, desired_c)}"
                    )
                continue

            target_r, target_c = target

            if self.debug:
                print(
                    f"[TRAVEL DEBUG] desired={(desired_r, desired_c)} chosen={(target_r, target_c)} "
                    f"accessible={world.maps.accessibility[agent.idx, target_r, target_c]} "
                    f"unoccupied={world.maps.unoccupied[target_r, target_c]}"
                )

            new_r, new_c = world.set_agent_loc(agent, target_r, target_c)

            if (new_r, new_c) == (target_r, target_c):
                origin_region = agent.state["region"]
                agent.inventory["Coin"] -= self.travel_cost_coin
                agent.state["endogenous"]["Labor"] += self.travel_cost_labor
                agent.state["travel_cooldown"] = self.cooldown
                agent.state["region"] = self._region_from_row(new_r)

                # Remove any market listings/orders from old region when agent travels
                if hasattr(self.world, "scenario") and hasattr(self.world.scenario, "get_component"):
                    cda = self.world.scenario.get_component("ContinuousDoubleAuction")
                    if cda is not None:
                        cda.cancel_all_orders_for_agent(agent.idx)

                # Send travel payment to scenario (regional pool)
                if hasattr(self.world, "scenario"):
                    self.world.scenario.add_travel_revenue(origin_region, self.travel_cost_coin)

                assert agent.inventory["Coin"] >= 0, (
                    f"Negative coin after travel for agent {agent.idx}: "
                    f"{agent.inventory['Coin']}"
                )

                self.successful_travelers.add(agent.idx)
                self.travel_log.append(
                    {
                        "t": int(world.timestep),
                        "agent": int(agent.idx),
                        "from": (int(old_r), int(old_c)),
                        "to": (int(new_r), int(new_c)),
                        "coin_after": float(agent.inventory["Coin"]),
                    }
                )

                if self.debug:
                    print(
                        f"[TRAVEL OK] Agent {agent.idx} | from {(old_r, old_c)} -> {(new_r, new_c)} | "
                        f"coin={agent.inventory['Coin']:.2f} | cooldown={agent.state['travel_cooldown']}"
                    )
            else:
                if self.debug:
                    print(
                        f"[TRAVEL FAIL] Agent {agent.idx} | from {(old_r, old_c)} tried {(target_r, target_c)} "
                        f"but ended at {(new_r, new_c)}"
                    )

        for agent in world.agents:
            if agent.state["travel_cooldown"] > 0:
                agent.state["travel_cooldown"] -= 1

    def _find_valid_target(self, agent, center_r, center_c, max_radius=6):
        world = self.world

        for radius in range(max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    r = center_r + dr
                    c = center_c + dc

                    if r < 0 or r >= world.world_size[0]:
                        continue
                    if c < 0 or c >= world.world_size[1]:
                        continue

                    if world.maps.accessibility[agent.idx, r, c] and world.maps.unoccupied[r, c]:
                        return r, c

        return None

    def generate_masks(self, completions=0):
        masks = {}

        for agent in self.world.agents:
            allow = 1.0

            if not self.enabled:
                allow = 0.0
            elif agent.state["travel_cooldown"] > 0:
                allow = 0.0
            elif agent.inventory["Coin"] < self.travel_cost_coin:
                allow = 0.0
            elif self.allow_only_agent is not None and agent.idx != self.allow_only_agent:
                allow = 0.0

            masks[agent.idx] = [allow]

        return masks

    
    def generate_observations(self):
        obs = {}
        for agent in self.world.agents:
            eligible = 1.0
            if self.allow_only_agent is not None and agent.idx != self.allow_only_agent:
                eligible = 0.0
            obs[agent.idx] = {
                "travel_cooldown": agent.state["travel_cooldown"],
                "travel_enabled_for_me": eligible if self.enabled else 0.0,
                "my_region_top": 1.0 if agent.state["region"] == "top" else 0.0,
                "my_region_bottom": 1.0 if agent.state["region"] == "bottom" else 0.0,
            }
        return obs

    def additional_reset_steps(self):
        self.successful_travelers = set()
        self.travel_log = []
        self.agent_start_locs = {}

        for agent in self.world.agents:
            row = int(agent.loc[0])
            col = int(agent.loc[1])

            agent.state["region"] = self._region_from_row(row)
            agent.state["travel_cooldown"] = 0

            self.agent_start_locs[int(agent.idx)] = (row, col)

        if self.debug:
            print("[RESET] agent_start_locs =", self.agent_start_locs)

    def get_dense_log(self):
        return self.travel_log