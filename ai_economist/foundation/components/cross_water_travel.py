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

        print(
            f"[CrossWaterTravel INIT] enabled={self.enabled}, "
            f"target_top={self.target_top}, target_bottom={self.target_bottom}, "
            f"travel_cost_coin={self.travel_cost_coin}, "
            f"travel_cost_labor={self.travel_cost_labor}, cooldown={self.cooldown}, "
            f"allow_only_agent={self.allow_only_agent}"
        )

    # --------------------------------------------------
    # ACTION SPACE
    # --------------------------------------------------

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return 1  # travel action
        return None

    # --------------------------------------------------
    # STATE
    # --------------------------------------------------

    def get_additional_state_fields(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return {"travel_cooldown": 0}
        return {}

    # --------------------------------------------------
    # STEP LOGIC
    # --------------------------------------------------

    def component_step(self):

        if not self.enabled:
            return

        world = self.world
        waterline = world.world_size[0] // 2  # row split

        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            # TEMP SANITY CHECK:
            # Force agent 0 to take the travel action whenever cooldown is 0.
            # Remove after verifying that cross-world travel works correctly.
            if agent.idx == 0 and agent.state["travel_cooldown"] == 0:
                action = 1

            if action == 0:
                continue

            # Restrict to one agent for testing
            if self.allow_only_agent is not None and agent.idx != self.allow_only_agent:
                continue

            # Cooldown check
            if agent.state["travel_cooldown"] > 0:
                continue

            # Cost check
            coin_before = float(agent.inventory["Coin"])
            if coin_before < self.travel_cost_coin:
                print(
                    #f"[TRAVEL BLOCKED] Agent {agent.idx} | coin={coin_before:.2f} "
                    #f"< cost={self.travel_cost_coin:.2f}"
                )
                continue

            old_r, old_c = agent.loc

            # Desired destination region
            if old_r < waterline:
                desired_r, desired_c = self.target_bottom
            else:
                desired_r, desired_c = self.target_top

            target = self._find_valid_target(agent, desired_r, desired_c, max_radius=6)
            if target is None:
                print(
                    f"[TRAVEL FAIL] Agent {agent.idx} | no valid target near {(desired_r, desired_c)}"
                )
                continue

            target_r, target_c = target

            print(
                f"[TRAVEL DEBUG] desired={(desired_r, desired_c)} chosen={(target_r, target_c)} "
                f"accessible={world.maps.accessibility[agent.idx, target_r, target_c]} "
                f"unoccupied={world.maps.unoccupied[target_r, target_c]}"
            )

            new_r, new_c = world.set_agent_loc(agent, target_r, target_c)

            if (new_r, new_c) == (target_r, target_c):
                # Charge cost exactly once, only after successful move
                agent.inventory["Coin"] -= self.travel_cost_coin
                agent.state["endogenous"]["Labor"] += self.travel_cost_labor
                agent.state["travel_cooldown"] = self.cooldown

                assert agent.inventory["Coin"] >= 0, (
                    f"Negative coin after travel for agent {agent.idx}: "
                    f"{agent.inventory['Coin']}"
                )

                print(
                    f"[TRAVEL OK] Agent {agent.idx} | from {(old_r, old_c)} -> {(new_r, new_c)} | "
                    f"coin={agent.inventory['Coin']:.2f} | cooldown={agent.state['travel_cooldown']}"
                )
            else:
                print(
                    f"[TRAVEL FAIL] Agent {agent.idx} | from {(old_r, old_c)} tried {(target_r, target_c)} "
                    f"but ended at {(new_r, new_c)}"
                )

        # decrement cooldown
        for agent in world.agents:
            if agent.state["travel_cooldown"] > 0:
                agent.state["travel_cooldown"] -= 1

    # --------------------------------------------------
    # TRAVEL TARGET 
    # --------------------------------------------------    

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

    # --------------------------------------------------
    # MASKS
    # --------------------------------------------------

    def generate_masks(self, completions=0):
        if not self.enabled:
            allow = 0.0

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

    # --------------------------------------------------
    # OBSERVATIONS
    # --------------------------------------------------

    def generate_observations(self):

        return {
            str(agent.idx): {
                "travel_cooldown": agent.state["travel_cooldown"]
            }
            for agent in self.world.agents
        }