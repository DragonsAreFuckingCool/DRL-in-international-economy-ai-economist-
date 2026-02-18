# Copyright (...)
# SPDX-License-Identifier: BSD-3-Clause

from ai_economist.foundation.base.base_agent import BaseAgent
from ai_economist.foundation.agents import agent_registry


# ---------------------------------------------------------------------
# Base class for multi-planner support
# ---------------------------------------------------------------------
@agent_registry.add
class RegionalPlannerBase(BaseAgent):
    """
    A multi-planner friendly base class.
    Designed to replace BasicPlanner in scenarios requiring >1 planners.

    Key differences from BasicPlanner:
      - Does NOT force idx="p"
      - Removes location from agent state (planners have no loc)
      - Allows multiple planner subclasses
      - Registers planner-only action subspaces (taxes, etc.)
    """

    name = "RegionalPlannerBase"

    def __init__(self, idx, multi_action_mode=False):
        super().__init__(idx=idx, multi_action_mode=multi_action_mode)

        # Planners do NOT have a location
        if "loc" in self.state:
            del self.state["loc"]

        # Initialize planner-specific state
        # (Will be extended by TaxComponent)
        self.state["planner-info"] = {}

    @property
    def loc(self):
        """Planners cannot have a 2D location."""
        raise AttributeError(
            f"{self.name} agents do not occupy a location in the world."
        )


# ---------------------------------------------------------------------
# TOP REGION PLANNER
# ---------------------------------------------------------------------
@agent_registry.add
class TopPlanner(RegionalPlannerBase):
    """
    Planner responsible for regulating the region NORTH of the waterline.
    """

    name = "TopPlanner"

    def __init__(self, multi_action_mode=False):
        super().__init__(idx="p_top", multi_action_mode=multi_action_mode)


# ---------------------------------------------------------------------
# BOTTOM REGION PLANNER
# ---------------------------------------------------------------------
@agent_registry.add
class BottomPlanner(RegionalPlannerBase):
    """
    Planner responsible for regulating the region SOUTH of the waterline.
    """

    name = "BottomPlanner"

    def __init__(self, multi_action_mode=False):
        super().__init__(idx="p_bottom", multi_action_mode=multi_action_mode)