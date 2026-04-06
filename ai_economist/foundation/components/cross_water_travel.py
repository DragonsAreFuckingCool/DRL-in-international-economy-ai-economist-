from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


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
        target_top=(7, 12),
        target_bottom=(43, 12),
        allow_only_agent=None,
        enabled=True,
        debug=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.travel_cost_coin = float(travel_cost_coin)
        self.travel_cost_labor = float(travel_cost_labor)
        self.cooldown = int(cooldown)
        self.target_top = target_top
        self.target_bottom = target_bottom
        self.allow_only_agent = allow_only_agent
        self.enabled = bool(enabled)
        self.debug = bool(debug)

        self.successful_travelers = set()
        self.travel_log = []

        if self.debug:
            print(
                f"[CrossWaterTravel INIT] enabled={self.enabled}, "
                f"target_top={self.target_top}, target_bottom={self.target_bottom}, "
                f"travel_cost_coin={self.travel_cost_coin}, "
                f"travel_cost_labor={self.travel_cost_labor}, cooldown={self.cooldown}, "
                f"allow_only_agent={self.allow_only_agent}"
            )

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

            if old_r < waterline:
                desired_r, desired_c = self.target_bottom
            else:
                desired_r, desired_c = self.target_top

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
        for agent in self.world.agents:
            row = int(agent.loc[0])
            agent.state["region"] = self._region_from_row(row)
            agent.state["travel_cooldown"] = 0

    def get_dense_log(self):
        return self.travel_log