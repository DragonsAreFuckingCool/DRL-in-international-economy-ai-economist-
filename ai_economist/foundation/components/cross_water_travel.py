from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class CrossWaterTravel(BaseComponent):

    name = "CrossWaterTravel"
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *args,
        travel_cost_coin=5.0,
        travel_cost_labor=0.0,
        cooldown=10,
        target_top=(12, 12),
        target_bottom=(38, 12),
        allow_only_agent=None,  # e.g. 0 for testing
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.travel_cost_coin = float(travel_cost_coin)
        self.travel_cost_labor = float(travel_cost_labor)
        self.cooldown = int(cooldown)

        self.target_top = target_top
        self.target_bottom = target_bottom

        self.allow_only_agent = allow_only_agent

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

        world = self.world
        waterline = world.world_size[0] // 2  # row split

        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            # TEMP SANITY CHECK:
            # Force agent 0 to take the travel action on the first few steps,
            # regardless of what the policy outputs. Remove after verifying
            # that cross-world travel works correctly.
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
            if agent.inventory["Coin"] < self.travel_cost_coin:
                continue

            old_r, old_c = agent.loc

            # Determine destination
            if old_r < waterline:
                target_r, target_c = self.target_bottom
            else:
                target_r, target_c = self.target_top

            # Deduct costs
            agent.inventory["Coin"] -= self.travel_cost_coin
            agent.state["endogenous"]["Labor"] += self.travel_cost_labor

            # Attempt move
            new_r, new_c = world.set_agent_loc(agent, target_r, target_c)

            # Check whether teleport succeeded
            if (new_r, new_c) == (target_r, target_c):
                agent.state["travel_cooldown"] = self.cooldown
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
    # MASKS
    # --------------------------------------------------

    def generate_masks(self, completions=0):

        masks = {}

        for agent in self.world.agents:

            allow = 1.0

            if agent.state["travel_cooldown"] > 0:
                allow = 0.0

            if agent.inventory["Coin"] < self.travel_cost_coin:
                allow = 0.0

            if self.allow_only_agent is not None:
                if agent.idx != self.allow_only_agent:
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