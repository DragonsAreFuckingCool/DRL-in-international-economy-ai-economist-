import gc
import json
import os
import pickle
import shutil
import sys
import time
import types
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import ray
from ray.rllib.agents.ppo import PPOTrainer

# -----------------------------------------------------------------------------
# Local project imports
# -----------------------------------------------------------------------------
# PROJECT_ROOT = Path(r"C:\Users\adria\coding\katja\DRL-in-international-economy-ai-economist-")


# -----------------------------------------------------------------------------
# Project imports for HPC
# -----------------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1] # Go up to project root (simulation.py → tutorials → project root)
print(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from ai_economist import foundation 
from ai_economist.foundation.components.utils import annealed_tax_limit
from ai_economist.foundation.scenarios.utils.rewards import coin_eq_times_productivity
from rllib import tf_models as _tf_models  # Registers masking-aware custom RLlib models.
from rllib.env_wrapper import RLlibEnvWrapper

# -----------------------------------------------------------------------------
# Global runtime settings
# -----------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@dataclass(frozen=True)
class ExperimentSettings:
    framework: str = "tf" 

    phase1_iters: int = 50
    phase2_iters: int = 125
    phase3a_iters: int = 50
    phase3b_iters: int = 125

    num_workers: int = 0
    num_envs_per_worker: int = 1
    num_gpus: int = 1
    num_cpus_per_worker: int = 1
    rollout_fragment_length: int = 100
    train_batch_size: int = 800
    sgd_minibatch_size: int = 128
    num_sgd_iter: int = 2

    min_band: int = 4
    period: int = 100
    episode_length: int = 1000
    world_size: Tuple[int, int] = (51, 25)
    layout_file: str = "stacked_51x25_symetric_original.txt"

    save_results: bool = True
    results_dir: str = "results"

    restrict_trade_to_region: bool = False

    travel_enabled_phase1: bool = False
    travel_enabled_phase2: bool = False
    travel_enabled_phase3a: bool = False
    travel_enabled_phase3b: bool = False

    travel_cost_coin_phase1: float = 2
    travel_cost_labor_phase1: float = 2
    travel_cooldown_phase1: int = 20

    travel_cost_coin_phase2: float = 2
    travel_cost_labor_phase2: float = 2
    travel_cooldown_phase2: int = 20

    travel_cost_coin_phase3a: float = 10
    travel_cost_labor_phase3a: float = 10
    travel_cooldown_phase3a: int = 101

    travel_cost_coin_phase3b: float = 10
    travel_cost_labor_phase3b: float = 10
    travel_cooldown_phase3b: int = 101

    fixed_tax_planner_id: Any = None
    fixed_tax_bracket_rates: Tuple[float, ...] = (
        0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00
    )
    fixed_tax_bracket_rates_top: Optional[Tuple[float, ...]] = None
    fixed_tax_bracket_rates_bottom: Optional[Tuple[float, ...]] = None

    experiment_extra_tag: str = "original_baseline"


SETTINGS = ExperimentSettings()

COMMON_AGENT_START_LOCS: List[Tuple[int, int]] = [
    (0, 0), (24, 0), (0, 24), (24, 24),
    (26, 0), (50, 0), (26, 24), (50, 24),
]

COMMON_BUILD_MULTIPLIERS: List[float] = [
    1.1, 1.3, 1.6, 2.2,
    1.1, 1.3, 1.6, 2.2,
]

COMMON_GATHER_MULTIPLIERS: List[float] = [1.5] * 8


# -----------------------------------------------------------------------------
# File utilities
# -----------------------------------------------------------------------------
def make_experiment_name(
    *,
    travel_enabled: bool,
    restrict_trade_to_region: bool,
    layout_file: str,
    extra_tag: Optional[str] = None,
) -> str:
    parts = [
        f"travel-{'on' if travel_enabled else 'off'}",
        f"regionaltrade-{'on' if restrict_trade_to_region else 'off'}",
        f"layout-{Path(layout_file).stem}",
    ]
    if extra_tag:
        parts.append(str(extra_tag))
    return "__".join(parts)


def make_experiment_dir(experiment_name: str, save_results: bool, root: str) -> Optional[Path]:
    if not save_results:
        print("Skipping saving results (save_results=False)")
        return None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def maybe_save_json(obj: Any, run_dir: Optional[Path], filename: str) -> None:
    if run_dir is not None:
        save_json(obj, run_dir / filename)


def maybe_save_pickle(obj: Any, run_dir: Optional[Path], filename: str) -> None:
    if run_dir is not None:
        save_pickle(obj, run_dir / filename)


def copy_rllib_checkpoint_to_run_dir(
    checkpoint_path: str,
    run_dir: Optional[Path],
    phase_key: str,
) -> str:
    if run_dir is None:
        return str(checkpoint_path)

    source = Path(checkpoint_path)
    if not source.exists():
        print(f"[WARN] Checkpoint not copied for {phase_key}; missing: {source}")
        return str(checkpoint_path)

    checkpoint_root = run_dir / "ray_checkpoints" / phase_key

    if source.is_dir():
        destination = checkpoint_root / source.name
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return str(destination)

    destination_dir = checkpoint_root / source.parent.name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)

    metadata_source = Path(str(source) + ".tune_metadata")
    if metadata_source.exists():
        shutil.copy2(metadata_source, Path(str(destination) + ".tune_metadata"))
    else:
        print(f"[WARN] Checkpoint metadata not found for {phase_key}: {metadata_source}")

    return str(destination)


# -----------------------------------------------------------------------------
# Environment / component configuration
# -----------------------------------------------------------------------------
def policy_mapping_fn(agent_id: Any) -> str:
    aid = str(agent_id)
    if aid.isdigit():
        return "a"
    if aid == "p_top":
        return "p_top"
    if aid == "p_bottom":
        return "p_bottom"
    return "a"


def travel_component_config(
    *,
    enabled: bool,
    allow_only_agent: Optional[int] = None,
    travel_cost_coin: float = 10.0,
    travel_cost_labor: float = 10.0,
    cooldown: int = 100,
) -> Tuple[str, Dict[str, Any]]:
    return (
        "CrossWaterTravel",
        {
            "enabled": enabled,
            "travel_cost_coin": travel_cost_coin,
            "travel_cost_labor": travel_cost_labor,
            "cooldown": cooldown,
            "allow_only_agent": allow_only_agent,
        },
    )


def build_component() -> Tuple[str, Dict[str, Any]]:
    return (
        "Build",
        {
            "skill_dist": "pareto",
            "payment_max_skill_multiplier": 3,
            "build_labor": 2.1,
            "payment": 10,
        },
    )


def auction_component(restrict_trade_to_region: bool) -> Tuple[str, Dict[str, Any]]:
    return (
        "ContinuousDoubleAuction",
        {
            "max_bid_ask": 10,
            "order_labor": 0.05,
            "max_num_orders": 5,
            "order_duration": 50,
            "restrict_trade_to_region": restrict_trade_to_region,
            "cross_region_trade_tax_mode": "percent",
            "cross_region_trade_tax_flat": 0.0,
            "cross_region_trade_tax_rate": 0.10,
            "cross_region_trade_tax_sink": False,
        },
    )


def gather_component() -> Tuple[str, Dict[str, Any]]:
    return (
        "Gather",
        {
            "move_labor": 0.21,
            "collect_labor": 0.21,
            "skill_dist": "pareto",
            "custom_gather_multipliers": COMMON_GATHER_MULTIPLIERS,
        },
    )


def regional_tax_component(
    *,
    region: str,
    planner_id: str,
    disable_taxes: bool,
    period: int,
    fixed_bracket_rates: Optional[Sequence[float]] = None,
    tax_annealing_schedule: Optional[Sequence[float]] = None,
) -> Tuple[str, Dict[str, Any]]:
    cfg: Dict[str, Any] = {
        "region": region,
        "planner_id": planner_id,
        "period": period,
        "bracket_spacing": "us-federal",
        "usd_scaling": 1000,
        "disable_taxes": disable_taxes,
    }
    if not disable_taxes:
        cfg.update(
            {
                "tax_model": "model_wrapper",
            }
        )
        if fixed_bracket_rates is None:
            if tax_annealing_schedule is not None:
                cfg["tax_annealing_schedule"] = list(tax_annealing_schedule)
        else:
            cfg["fixed_planner_bracket_rates"] = list(fixed_bracket_rates)
    return ("RegionalPeriodicBracketTax", cfg)


def base_env_config(settings: ExperimentSettings) -> Dict[str, Any]:
    return {
        "scenario_name": "custom/splitworld_overlay_regional",
        "env_layout_file": settings.layout_file,
        "world_size": list(settings.world_size),
        "episode_length": settings.episode_length,
        "starting_agent_coin": 0,
        "fixed_four_skill_and_loc": False,
        "n_agents": 8,
        "planner_subclasses": ["TopPlanner", "BottomPlanner"],
        "multi_action_mode_planner": True,
        "multi_action_mode_agents": True,
        "flatten_observations": True,
        "flatten_masks": True,
        "dense_log_frequency": 1,
        "agent_start_locs": COMMON_AGENT_START_LOCS,
        "agent_start_build_payment_multipliers": COMMON_BUILD_MULTIPLIERS,
    }


def make_phase_env_config(
    settings: ExperimentSettings,
    *,
    disable_taxes: bool,
    travel_enabled: bool,
    travel_cost_coin: float,
    travel_cost_labor: float,
    travel_cooldown: int,
    restrict_trade_to_region: bool,
    tax_annealing_schedule: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    cfg = deepcopy(base_env_config(settings))
    fixed_tax_planner_ids = fixed_tax_planner_ids_from_settings(settings)
    cfg["components"] = [
        build_component(),
        auction_component(restrict_trade_to_region=restrict_trade_to_region),
        gather_component(),
        regional_tax_component(
            region="top",
            planner_id="p_top",
            disable_taxes=disable_taxes,
            period=settings.period,
            fixed_bracket_rates=(
                fixed_tax_bracket_rates_for_planner(settings, "p_top")
                if "p_top" in fixed_tax_planner_ids
                else None
            ),
            tax_annealing_schedule=tax_annealing_schedule,
        ),
        regional_tax_component(
            region="bottom",
            planner_id="p_bottom",
            disable_taxes=disable_taxes,
            period=settings.period,
            fixed_bracket_rates=(
                fixed_tax_bracket_rates_for_planner(settings, "p_bottom")
                if "p_bottom" in fixed_tax_planner_ids
                else None
            ),
            tax_annealing_schedule=tax_annealing_schedule,
        ),
        travel_component_config(
            enabled=travel_enabled,
            allow_only_agent=None,
            travel_cost_coin=travel_cost_coin,
            travel_cost_labor=travel_cost_labor,
            cooldown=travel_cooldown,
        ),
    ]
    return cfg


def fixed_tax_planner_ids_from_settings(settings: ExperimentSettings) -> List[str]:
    planner_ids = ["p_top", "p_bottom"]

    value = settings.fixed_tax_planner_id
    if value is None:
        return []

    if isinstance(value, str):
        fixed_ids = planner_ids if value.lower() == "both" else [value]
    elif isinstance(value, (list, tuple, set)):
        fixed_ids = list(value)
    else:
        raise ValueError(
            "fixed_tax_planner_id must be None, 'p_top', 'p_bottom', 'both', "
            "or a list/tuple/set of planner ids; "
            f"got {settings.fixed_tax_planner_id!r}"
        )

    fixed_ids = [str(pid) for pid in fixed_ids]
    unknown = [pid for pid in fixed_ids if pid not in planner_ids]
    if unknown:
        raise ValueError(
            "fixed_tax_planner_id can only contain 'p_top' and/or 'p_bottom'; "
            f"got {unknown!r}"
        )

    return [pid for pid in planner_ids if pid in set(fixed_ids)]


def fixed_tax_bracket_rates_for_planner(
    settings: ExperimentSettings,
    planner_id: str,
) -> Sequence[float]:
    if planner_id == "p_top" and settings.fixed_tax_bracket_rates_top is not None:
        return settings.fixed_tax_bracket_rates_top
    if planner_id == "p_bottom" and settings.fixed_tax_bracket_rates_bottom is not None:
        return settings.fixed_tax_bracket_rates_bottom
    return settings.fixed_tax_bracket_rates


def trainable_planner_policies(settings: ExperimentSettings) -> List[str]:
    planner_ids = ["p_top", "p_bottom"]
    fixed_ids = set(fixed_tax_planner_ids_from_settings(settings))
    return [pid for pid in planner_ids if pid not in fixed_ids]


def build_all_phase_env_configs(settings: ExperimentSettings) -> Dict[str, Dict[str, Any]]:
    return {
        "phase1": make_phase_env_config(
            settings,
            disable_taxes=True,
            travel_enabled=settings.travel_enabled_phase1,
            travel_cost_coin=settings.travel_cost_coin_phase1,
            travel_cost_labor=settings.travel_cost_labor_phase1,
            travel_cooldown=settings.travel_cooldown_phase1,
            restrict_trade_to_region=settings.restrict_trade_to_region,
            tax_annealing_schedule=None,
        ),
        "phase2": make_phase_env_config(
            settings,
            disable_taxes=False,
            travel_enabled=settings.travel_enabled_phase2,
            travel_cost_coin=settings.travel_cost_coin_phase2,
            travel_cost_labor=settings.travel_cost_labor_phase2,
            travel_cooldown=settings.travel_cooldown_phase2,
            restrict_trade_to_region=settings.restrict_trade_to_region,
            #tax_annealing_schedule=(0, 0.01),
            tax_annealing_schedule=(0, 0.0125)

        ),
        "phase3a": make_phase_env_config(
            settings,
            disable_taxes=False,
            travel_enabled=settings.travel_enabled_phase3a,
            travel_cost_coin=settings.travel_cost_coin_phase3a,
            travel_cost_labor=settings.travel_cost_labor_phase3a,
            travel_cooldown=settings.travel_cooldown_phase3a,
            restrict_trade_to_region=settings.restrict_trade_to_region,
            tax_annealing_schedule=None,
        ),
        "phase3b": make_phase_env_config(
            settings,
            disable_taxes=False,
            travel_enabled=settings.travel_enabled_phase3b,
            travel_cost_coin=settings.travel_cost_coin_phase3b,
            travel_cost_labor=settings.travel_cost_labor_phase3b,
            travel_cooldown=settings.travel_cooldown_phase3b,
            restrict_trade_to_region=settings.restrict_trade_to_region,
            tax_annealing_schedule=None,
        ),
    }


# -----------------------------------------------------------------------------
# RLlib policy/trainer configuration
# -----------------------------------------------------------------------------
def build_policies(
    env_obj: RLlibEnvWrapper,
    *,
    agent_policy_config: Optional[Dict[str, Any]] = None,
    p_top_config: Optional[Dict[str, Any]] = None,
    p_bottom_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[Any, Any, Any, Dict[str, Any]]]:
    obs_space_a = env_obj.observation_space
    act_space_a = env_obj.action_space
    obs_space_top = env_obj.observation_space_pl["p_top"]
    act_space_top = env_obj.action_space_pl["p_top"]
    obs_space_bottom = env_obj.observation_space_pl["p_bottom"]
    act_space_bottom = env_obj.action_space_pl["p_bottom"]

    agent_policy_config = agent_policy_config or {"lr": 3e-4}
    default_planner_config = {
        "lr": 1e-4,
        "entropy_coeff": 0.02,
        "model": {
            "custom_model": "keras_linear",
            "custom_options": {
                "fully_connected_value": True,
                "fc_dim": 128,
                "num_fc": 2,
            },
        },
    }
    p_top_config = {**default_planner_config, **(p_top_config or {})}
    p_bottom_config = {**default_planner_config, **(p_bottom_config or {})}

    return {
        "a": (None, obs_space_a, act_space_a, agent_policy_config),
        "p_top": (None, obs_space_top, act_space_top, p_top_config),
        "p_bottom": (None, obs_space_bottom, act_space_bottom, p_bottom_config),
    }


# def build_trainer_config(
#     *,
#     settings: ExperimentSettings,
#     env_config_dict: Dict[str, Any],
#     policies: Dict[str, Any],
#     policies_to_train: Sequence[str],
# ) -> Dict[str, Any]:
#     return {
#         "env": RLlibEnvWrapper,
#         "env_config": {
#             "env_config_dict": env_config_dict,
#             "num_envs_per_worker": 1,
#         },
#         "multiagent": {
#             "policies": policies,
#             "policies_to_train": list(policies_to_train),
#             "policy_mapping_fn": policy_mapping_fn,
#         },
#         "num_workers": 0,
#         "num_envs_per_worker": 1,
#         "framework": settings.framework,
#         "num_gpus": 1,
#         "rollout_fragment_length": 50,
#         "batch_mode": "truncate_episodes",
#         "train_batch_size": 800,
#         "sgd_minibatch_size": 128,
#         "num_sgd_iter": 2,
#         "log_level": "WARN",
#     }

def build_trainer_config(
    *,
    settings: ExperimentSettings,
    env_config_dict: Dict[str, Any],
    policies: Dict[str, Any],
    policies_to_train: Sequence[str],
) -> Dict[str, Any]:
    return {
        "env": RLlibEnvWrapper,
        "env_config": {
            "env_config_dict": env_config_dict,
            "num_envs_per_worker": settings.num_envs_per_worker,
        },
        "multiagent": {
            "policies": policies,
            "policies_to_train": list(policies_to_train),
            "policy_mapping_fn": policy_mapping_fn,
        },
        "num_workers": settings.num_workers,
        "num_envs_per_worker": settings.num_envs_per_worker,
        "num_cpus_per_worker": settings.num_cpus_per_worker,
        "framework": settings.framework,
        "num_gpus": settings.num_gpus,
        "rollout_fragment_length": settings.rollout_fragment_length,
        "batch_mode": "truncate_episodes",
        "train_batch_size": settings.train_batch_size,
        "sgd_minibatch_size": settings.sgd_minibatch_size,
        "num_sgd_iter": settings.num_sgd_iter,
        "log_level": "WARN",
        #"reuse_actors": True, # Could be false, but i think its good for runtime 
    }

# def build_trainer_config(
#     *,
#     settings: ExperimentSettings,
#     env_config_dict: Dict[str, Any],
#     policies: Dict[str, Any],
#     policies_to_train: Sequence[str],
# ) -> Dict[str, Any]:
#     return {
#         "env": RLlibEnvWrapper,
#         "env_config": {
#             "env_config_dict": env_config_dict,
#             "num_envs_per_worker": 2,
#         },
#         "multiagent": {
#             "policies": policies,
#             "policies_to_train": list(policies_to_train),
#             "policy_mapping_fn": policy_mapping_fn,
#         },

#         # parallel rollout
#         "num_workers": 12,
#         "num_envs_per_worker": 2,

#         # hardware
#         "framework": settings.framework,
#         "num_gpus": 1,
#         "num_cpus_per_worker": 1,
#         "num_gpus_per_worker": 0,

#         # PPO sampling/training
#         "rollout_fragment_length": 200,
#         "batch_mode": "truncate_episodes",
#         "train_batch_size": 4800,      # 12 workers * 2 envs * 200 steps
#         "sgd_minibatch_size": 512,
#         "num_sgd_iter": 8,

#         # stability / practical
#         "log_level": "WARN",
#     }


# -----------------------------------------------------------------------------
# Tax mask patching helpers
# -----------------------------------------------------------------------------
def patch_regional_tax_masks(env_obj: RLlibEnvWrapper, min_band: int = 4) -> None:
    """
    Patch all RegionalPeriodicBracketTax components inside one RLlibEnvWrapper
    so each bound planner only has its own bracket controls active, while keeping
    at least `min_band` legal rate indices on tax day.
    """
    for comp in env_obj.env.components:
        if "BracketTax" not in comp.name:
            continue

        def make_patched(component: Any):
            def _patched_generate_masks(self: Any, completions: int = 0) -> Dict[str, Dict[str, np.ndarray]]:
                planner = self._get_bound_planner(self.world)
                if planner is None or str(planner.idx) != str(self._planner_id):
                    return {}

                if (
                    completions != self._last_completions
                    and self.tax_annealing_schedule is not None
                ):
                    self._last_completions = int(completions)
                    self._annealed_rate_max = annealed_tax_limit(
                        completions,
                        self._annealing_warmup,
                        self._annealing_slope,
                        self.rate_max,
                    )

                all_keys = [f"TaxIndexBracket_{int(r):03d}" for r in self.bracket_cutoffs]
                my_keys = [f"TaxIndexBracket_{int(r):03d}" for r in self.regional_brackets]

                disc = np.array(self.disc_rates, dtype=float)
                n_rates = len(disc)

                if getattr(self, "tax_annealing_schedule", None) is not None:
                    cap = float(getattr(self, "_annealed_rate_max", disc[-1]))
                    allowed_idx = np.where(disc <= cap + 1e-8)[0]
                else:
                    allowed_idx = np.arange(n_rates)

                if allowed_idx.size < min_band:
                    allowed_idx = np.arange(min(min_band, n_rates))

                if getattr(self, "_fixed_planner_bracket_rates", None) is not None:
                    rate_mask = np.zeros(n_rates, dtype=np.float32)
                elif getattr(self, "tax_cycle_pos", None) == 1:
                    rate_mask = np.zeros(n_rates, dtype=np.float32)
                    rate_mask[allowed_idx] = 1.0
                else:
                    rate_mask = np.zeros(n_rates, dtype=np.float32)

                zero_mask = np.zeros_like(rate_mask, dtype=np.float32)
                masks = {
                    key: rate_mask if key in my_keys else zero_mask
                    for key in all_keys
                }
                return {str(planner.idx): masks}

            return types.MethodType(_patched_generate_masks, component)

        comp.generate_masks = make_patched(comp)


def patch_trainer_envs(trainer, min_band: int = 4) -> None:
    def _patch_env(env: RLlibEnvWrapper) -> bool:
        patch_regional_tax_masks(env, min_band=min_band)
        return True

    trainer.workers.foreach_worker(lambda worker: worker.foreach_env(_patch_env))


def is_tax_day(env_obj: RLlibEnvWrapper) -> bool:
    for comp in env_obj.env.components:
        if "BracketTax" in comp.name and getattr(comp, "tax_cycle_pos", None) == 1:
            return True
    return False


def print_legal_rate_counts_at_next_tax_day(env_obj: RLlibEnvWrapper, fallback_period: int) -> None:
    obs = env_obj.reset(force_dense_logging=False)
    dummy_actions: Dict[str, Any] = {}

    for aid in obs.keys():
        if str(aid).isdigit():
            dummy_actions[aid] = env_obj.action_space.sample()
        elif aid in ("p_top", "p_bottom"):
            dummy_actions[aid] = env_obj.action_space_pl[aid].sample()

    try:
        _ = next(comp.period for comp in env_obj.env.components if "BracketTax" in comp.name)
    except StopIteration:
        _ = fallback_period

    while not is_tax_day(env_obj):
        obs, rew, done, info = env_obj.step(dummy_actions)
        if done.get("__all__", False):
            obs = env_obj.reset()

    masks_raw = env_obj.env._generate_masks(flatten_masks=False)
    for pid in ("p_top", "p_bottom"):
        if pid not in masks_raw:
            print(f"{pid}: no mask dict found.")
            continue
        counts = [int(np.sum(np.array(mask) > 0)) for mask in masks_raw[pid].values()]
        print(f"== {pid} legal counts per subspace ==\n{counts}")


# -----------------------------------------------------------------------------
# Dense log / planner action helpers
# -----------------------------------------------------------------------------
def get_active_planner_halves(dense_log: Dict[str, Any], top_first: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    arr_top = dense_log["planner_actions"]["p_top"]
    arr_bottom = dense_log["planner_actions"]["p_bottom"]

    if arr_top.size == 0 or arr_bottom.size == 0:
        empty = np.zeros((0, 0), dtype=np.int64)
        return empty, empty

    half = arr_top.shape[1] // 2
    if top_first:
        active_top = arr_top[:, :half]
        active_bottom = arr_bottom[:, half:]
    else:
        active_top = arr_top[:, half:]
        active_bottom = arr_bottom[:, :half]

    return active_top, active_bottom


def print_active_planner_debug(dense_log: Dict[str, Any], top_first: bool = True, max_rows: int = 10) -> None:
    active_top, active_bottom = get_active_planner_halves(dense_log, top_first=top_first)

    print("\n=== ACTIVE BRACKETS ONLY ===")
    print("p_top active shape:", active_top.shape)
    if active_top.size:
        print("p_top active unique rows:", np.unique(active_top, axis=0).shape[0])
        print("p_top active sample:")
        print(active_top[:max_rows])

    print("\np_bottom active shape:", active_bottom.shape)
    if active_bottom.size:
        print("p_bottom active unique rows:", np.unique(active_bottom, axis=0).shape[0])
        print("p_bottom active sample:")
        print(active_bottom[:max_rows])


def get_disc_rates(env_obj: RLlibEnvWrapper) -> np.ndarray:
    for comp in env_obj.env.components:
        if "BracketTax" in comp.name and hasattr(comp, "disc_rates"):
            return np.array(comp.disc_rates, dtype=float)
    return np.arange(0.0, 1.0 + 1e-9, 0.05)


def print_active_planner_rates(
    dense_log: Dict[str, Any],
    env_obj: RLlibEnvWrapper,
    top_first: bool = True,
    max_rows: int = 10,
) -> None:
    disc_rates = get_disc_rates(env_obj)
    active_top, active_bottom = get_active_planner_halves(dense_log, top_first=top_first)

    if active_top.size:
        top_rates = disc_rates[np.clip(active_top, 0, len(disc_rates) - 1)]
        print("\np_top active rates:")
        print(top_rates[:max_rows])

    if active_bottom.size:
        bottom_rates = disc_rates[np.clip(active_bottom, 0, len(disc_rates) - 1)]
        print("\np_bottom active rates:")
        print(bottom_rates[:max_rows])


def summarize_dense_log(log: Dict[str, Any]) -> Dict[str, Any]:
    states = log["states"]
    last_state = states[-1]

    agent_ids = sorted(int(key) for key in last_state.keys() if str(key).isdigit())

    final_coin: List[float] = []
    final_labor: List[float] = []
    final_region = {"top": 0, "bottom": 0}

    split = SETTINGS.world_size[0] // 2   # for 51 rows this is 25

    for aid in agent_ids:
        state = last_state[str(aid)]
        coin = state["inventory"]["Coin"] + state["escrow"]["Coin"]
        labor = state["endogenous"]["Labor"]
        r, c = state["loc"]

        final_coin.append(float(coin))
        final_labor.append(float(labor))

        region = "top" if r < split else "bottom"
        final_region[region] += 1

    n_trades = 0
    if "Trade" in log:
        for trade_step in log["Trade"]:
            trades = trade_step.get("trades", []) if isinstance(trade_step, dict) else trade_step
            n_trades += len(trades)

    n_builds = 0
    total_build_income = 0.0
    if "Build" in log:
        for build_step in log["Build"]:
            builds = build_step.get("builds", []) if isinstance(build_step, dict) else build_step
            n_builds += len(builds)
            total_build_income += sum(build.get("income", 0.0) for build in builds)

    n_travel = 0
    if "CrossWaterTravel" in log:
        for event in log["CrossWaterTravel"]:
            if isinstance(event, list):
                n_travel += len(event)
            elif isinstance(event, dict):
                n_travel += 1

    return {
        "mean_final_coin": float(np.mean(final_coin)) if final_coin else np.nan,
        "std_final_coin": float(np.std(final_coin)) if final_coin else np.nan,
        "mean_final_labor": float(np.mean(final_labor)) if final_labor else np.nan,
        "n_trades": int(n_trades),
        "n_builds": int(n_builds),
        "total_build_income": float(total_build_income),
        "n_travel": int(n_travel),
        "n_top_final": int(final_region["top"]),
        "n_bottom_final": int(final_region["bottom"]),
    }


# -----------------------------------------------------------------------------
# Training / rollout
# -----------------------------------------------------------------------------
def generate_rollout_with_planner_actions(
    trainer,
    env_obj: RLlibEnvWrapper,
    *,
    num_dense_logs: int = 2,
    explore: bool = False,
    log_only_tax_days: bool = True,
) -> Dict[int, Dict[str, Any]]:
    def _compute_action(policy_id: str, obs: Any, state: List[np.ndarray]) -> Tuple[Any, List[np.ndarray]]:
        out = trainer.compute_action(
            observation=obs,
            state=state,
            policy_id=policy_id,
            full_fetch=False,
            explore=explore,
        )
        if isinstance(out, tuple):
            if len(out) >= 2:
                return out[0], out[1]
            return out[0], state
        return out, state

    dense_logs: Dict[int, Dict[str, Any]] = {}

    for episode_idx in range(num_dense_logs):
        obs = env_obj.reset(force_dense_logging=True)

        agent_states = {
            str(i): trainer.get_policy("a").get_initial_state()
            for i in range(env_obj.env.n_agents)
        }
        p_top_state = trainer.get_policy("p_top").get_initial_state()
        p_bottom_state = trainer.get_policy("p_bottom").get_initial_state()

        top_actions: List[np.ndarray] = []
        bottom_actions: List[np.ndarray] = []
        top_rewards: List[float] = []
        bottom_rewards: List[float] = []
        agent_utility_history: List[Dict[str, float]] = []

        for _ in range(env_obj.env.episode_length):
            actions: Dict[str, Any] = {}

            for i in range(env_obj.env.n_agents):
                aid = str(i)
                action, next_state = _compute_action("a", obs[aid], agent_states[aid])
                actions[aid] = action
                agent_states[aid] = next_state

            top_action, p_top_state = _compute_action("p_top", obs["p_top"], p_top_state)
            bottom_action, p_bottom_state = _compute_action("p_bottom", obs["p_bottom"], p_bottom_state)
            actions["p_top"] = top_action
            actions["p_bottom"] = bottom_action

            should_log = (not log_only_tax_days) or is_tax_day(env_obj)
            if should_log:
                top_actions.append(np.array(top_action, copy=True))
                bottom_actions.append(np.array(bottom_action, copy=True))

            obs, rew, done, info = env_obj.step(actions)

            util_t: Dict[str, float] = {}
            for agent in env_obj.env.world.agents:
                aid = str(agent.idx)
                util_t[aid] = float(env_obj.env.curr_optimization_metric.get(agent.idx, np.nan))
            agent_utility_history.append(util_t)

            top_rewards.append(rew.get("p_top", np.nan))
            bottom_rewards.append(rew.get("p_bottom", np.nan))

            if done.get("__all__", False):
                break

        top_arr = np.stack(top_actions, axis=0) if top_actions else np.zeros((0, 0), dtype=np.int64)
        bottom_arr = np.stack(bottom_actions, axis=0) if bottom_actions else np.zeros((0, 0), dtype=np.int64)

        dense_logs[episode_idx] = dict(env_obj.env.dense_log)

        if "states" in dense_logs[episode_idx]:
            n_states = len(dense_logs[episode_idx]["states"])
            n_utils = len(agent_utility_history)
            n_match = min(n_states, n_utils)

            for t in range(n_match):
                for aid, util in agent_utility_history[t].items():
                    if aid in dense_logs[episode_idx]["states"][t]:
                        dense_logs[episode_idx]["states"][t][aid]["utility"] = util

        dense_logs[episode_idx]["planner_actions"] = {
            "p_top": top_arr,
            "p_bottom": bottom_arr,
        }
        dense_logs[episode_idx]["planner_rewards"] = {
            "p_top": top_rewards,
            "p_bottom": bottom_rewards,
        }

    return dense_logs


def train_phase(
    trainer,
    *,
    phase_name: str,
    iterations: int,
    metrics_store: List[Dict[str, Any]],
    run_dir: Optional[Path],
    save_results: bool,
) -> str:
    start_time = time.time()

    for iteration in range(iterations):
        result = trainer.train()

        row = {
            "phase": phase_name,
            "iter": iteration,
            "timesteps_total": result.get("timesteps_total"),
            "episodes_total": result.get("episodes_total"),
            "episode_reward_mean": result.get("episode_reward_mean"),
            "episode_reward_min": result.get("episode_reward_min"),
            "episode_reward_max": result.get("episode_reward_max"),
        }

        policy_reward_mean = result.get("policy_reward_mean", {})
        row["policy_reward_mean/p_top"] = policy_reward_mean.get("p_top", np.nan)
        row["policy_reward_mean/p_bottom"] = policy_reward_mean.get("p_bottom", np.nan)
        row["policy_reward_mean/a"] = policy_reward_mean.get("a", np.nan)

        try:
            env = trainer.workers.local_worker().env
            coins = np.array(
                [float(agent.total_endowment("Coin")) for agent in env.env.world.agents],
                dtype=float,
            )
            social_welfare = coin_eq_times_productivity(
                coin_endowments=coins,
                equality_weight=1.0,
            )
            row["social_welfare_coin_eq_times_prod"] = float(social_welfare)
        except Exception as exc:
            row["social_welfare_coin_eq_times_prod"] = np.nan
            print(f"DEBUG social welfare error in {phase_name}, iter {iteration}: {exc!r}")

        metrics_store.append(row)

        if iteration % 25 == 0:
            print(f"[{phase_name}] Iter={iteration:04d}/{iterations} reward={result.get('episode_reward_mean')}")
            print(
                f"   p_top reward: {row['policy_reward_mean/p_top']}, "
                f"p_bottom reward: {row['policy_reward_mean/p_bottom']}"
            )
            print(f"   social welfare (coin_eq_times_prod): {row['social_welfare_coin_eq_times_prod']}")

        if iteration % 100 == 0 and iteration > 0:
            checkpoint_tmp = trainer.save()
            if save_results and run_dir is not None:
                save_json(
                    {"latest_intermediate_checkpoint": str(checkpoint_tmp)},
                    run_dir / f"{phase_name.lower().replace(' ', '_')}_intermediate_checkpoint.json",
                )

        if iteration % 10 == 0:
            gc.collect()

    checkpoint = trainer.save()
    elapsed_minutes = (time.time() - start_time) / 60.0
    print(f"[{phase_name}] Final checkpoint: {checkpoint}  ({elapsed_minutes:.1f} min)")
    return str(checkpoint)


# -----------------------------------------------------------------------------
# Phase orchestration
# -----------------------------------------------------------------------------
def create_env(env_config_dict: Dict[str, Any]) -> RLlibEnvWrapper:
    return RLlibEnvWrapper({"env_config_dict": env_config_dict}, verbose=False)


def create_trainer(settings, env_config_dict, env_obj, policies_to_train, agent_policy_config=None):
    policies = build_policies(env_obj, agent_policy_config=agent_policy_config)
    return PPOTrainer(
        config=build_trainer_config(
            settings=settings,
            env_config_dict=env_config_dict,
            policies=policies,
            policies_to_train=policies_to_train,
        )
    )


def set_policy_weights_and_sync(trainer, policy_weights: Dict[str, Any]) -> None:
    """Set local policy weights and push them to remote rollout workers."""
    for policy_id, weights in policy_weights.items():
        trainer.get_policy(policy_id).set_weights(weights)

    # With num_workers=0 this is effectively a no-op. With remote workers it is
    # essential: manual get_policy(...).set_weights(...) only touches the local worker.
    try:
        trainer.workers.sync_weights(policies=list(policy_weights.keys()))
    except TypeError:
        trainer.workers.sync_weights()


def run_experiment(settings: ExperimentSettings) -> Dict[str, Any]:
    experiment_name = make_experiment_name(
        travel_enabled=settings.travel_enabled_phase3b,
        restrict_trade_to_region=settings.restrict_trade_to_region,
        layout_file=settings.layout_file,
        extra_tag=settings.experiment_extra_tag,
    )

    phase_env_configs = build_all_phase_env_configs(settings)
    trainable_planners = trainable_planner_policies(settings)
    run_dir = make_experiment_dir(experiment_name, settings.save_results, settings.results_dir)

    if settings.save_results and run_dir is not None:
        save_json(
            {
                **asdict(settings),
                "experiment_name": experiment_name,
                "env_config_dict_phase1": phase_env_configs["phase1"],
                "env_config_dict_phase2": phase_env_configs["phase2"],
                "env_config_dict_phase3a": phase_env_configs["phase3a"],
                "env_config_dict_phase3b": phase_env_configs["phase3b"],
            },
            run_dir / "config.json",
        )

    all_metrics: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # PHASE 1: train agents only, taxes disabled
    # -------------------------------------------------------------------------
    env_phase1 = create_env(phase_env_configs["phase1"])
    trainer_phase1 = create_trainer(
        settings=settings,
        env_config_dict=phase_env_configs["phase1"],
        env_obj=env_phase1,
        policies_to_train=["a"],
    )

    ckpt_phase1 = train_phase(
        trainer_phase1,
        phase_name="PHASE 1",
        iterations=settings.phase1_iters,
        metrics_store=all_metrics,
        run_dir=run_dir,
        save_results=settings.save_results,
    )

    dense_logs_phase1 = generate_rollout_with_planner_actions(
        trainer_phase1,
        env_phase1,
        num_dense_logs=1,
        explore=False,
        log_only_tax_days=False,
    )
    maybe_save_pickle(dense_logs_phase1, run_dir, "dense_logs_phase1.pkl")

    agent_weights_phase1 = trainer_phase1.get_policy("a").get_weights()

    trainer_phase1.stop()
    del trainer_phase1
    gc.collect()

    # -------------------------------------------------------------------------
    # PHASE 2: freeze agents, train planners
    # -------------------------------------------------------------------------
    env_phase2 = create_env(phase_env_configs["phase2"])
    patch_regional_tax_masks(env_phase2, min_band=settings.min_band)

    trainer_phase2 = create_trainer(
        settings=settings,
        env_config_dict=phase_env_configs["phase2"],
        env_obj=env_phase2,
        policies_to_train=trainable_planners,
    )
    patch_trainer_envs(trainer_phase2, min_band=settings.min_band)
    set_policy_weights_and_sync(trainer_phase2, {"a": agent_weights_phase1})

    ckpt_phase2 = train_phase(
        trainer_phase2,
        phase_name="PHASE 2",
        iterations=settings.phase2_iters,
        metrics_store=all_metrics,
        run_dir=run_dir,
        save_results=settings.save_results,
    )

    planner_top_weights_phase2 = trainer_phase2.get_policy("p_top").get_weights()
    planner_bottom_weights_phase2 = trainer_phase2.get_policy("p_bottom").get_weights()

    dense_logs_phase2 = generate_rollout_with_planner_actions(
        trainer_phase2,
        env_phase2,
        num_dense_logs=1,
        explore=False,
        log_only_tax_days=False,
    )
    maybe_save_pickle(dense_logs_phase2, run_dir, "dense_logs_phase2.pkl")

    # -------------------------------------------------------------------------
    # PHASE 3A: train agents against fixed planners
    # -------------------------------------------------------------------------
    env_phase3a = create_env(phase_env_configs["phase3a"])
    patch_regional_tax_masks(env_phase3a, min_band=settings.min_band)

    trainer_phase3a = create_trainer(
        settings=settings,
        env_config_dict=phase_env_configs["phase3a"],
        env_obj=env_phase3a,
        policies_to_train=["a"],
        agent_policy_config={"lr": 3e-4, "entropy_coeff": 0.01},
    )
    patch_trainer_envs(trainer_phase3a, min_band=settings.min_band)

    set_policy_weights_and_sync(
        trainer_phase3a,
        {
            "a": agent_weights_phase1,
            "p_top": planner_top_weights_phase2,
            "p_bottom": planner_bottom_weights_phase2,
        },
    )

    ckpt_phase3a = train_phase(
        trainer_phase3a,
        phase_name="PHASE 3A",
        iterations=settings.phase3a_iters,
        metrics_store=all_metrics,
        run_dir=run_dir,
        save_results=settings.save_results,
    )

    agent_weights_phase3a = trainer_phase3a.get_policy("a").get_weights()

    dense_logs_phase3a = generate_rollout_with_planner_actions(
        trainer_phase3a,
        env_phase3a,
        num_dense_logs=1,
        explore=False,
        log_only_tax_days=False,
    )
    maybe_save_pickle(dense_logs_phase3a, run_dir, "dense_logs_phase3a.pkl")

    # -------------------------------------------------------------------------
    # PHASE 3B: joint training
    # -------------------------------------------------------------------------
    env_phase3b = create_env(phase_env_configs["phase3b"])
    patch_regional_tax_masks(env_phase3b, min_band=settings.min_band)

    trainer_phase3b = create_trainer(
        settings=settings,
        env_config_dict=phase_env_configs["phase3b"],
        env_obj=env_phase3b,
        policies_to_train=["a", *trainable_planners],
    )
    patch_trainer_envs(trainer_phase3b, min_band=settings.min_band)

    set_policy_weights_and_sync(
        trainer_phase3b,
        {
            "a": agent_weights_phase3a,
            "p_top": planner_top_weights_phase2,
            "p_bottom": planner_bottom_weights_phase2,
        },
    )

    ckpt_phase3b = train_phase(
        trainer_phase3b,
        phase_name="PHASE 3B",
        iterations=settings.phase3b_iters,
        metrics_store=all_metrics,
        run_dir=run_dir,
        save_results=settings.save_results,
    )

    # -------------------------------------------------------------------------
    # Final rollout / metrics / summary
    # -------------------------------------------------------------------------
    dense_logs_final = generate_rollout_with_planner_actions(
        trainer_phase3b,
        env_phase3b,
        num_dense_logs=20, #3
        explore=True,
        log_only_tax_days=True,
    )

    metrics_df = pd.DataFrame(all_metrics)
    maybe_save_pickle(dense_logs_final, run_dir, "dense_logs_final.pkl")

    portable_checkpoints = {
        "phase1": ckpt_phase1,
        "phase2": ckpt_phase2,
        "phase3a": ckpt_phase3a,
        "phase3b": copy_rllib_checkpoint_to_run_dir(ckpt_phase3b, run_dir, "phase3b"),
    }

    if settings.save_results and run_dir is not None:
        metrics_df.to_csv(run_dir / "training_metrics.csv", index=False)
        save_json(portable_checkpoints, run_dir / "checkpoints.json")

    summary = summarize_dense_log(dense_logs_final[0])
    summary.update(
        {
            "experiment_name": experiment_name,
            "phase1_iters": settings.phase1_iters,
            "phase2_iters": settings.phase2_iters,
            "phase3a_iters": settings.phase3a_iters,
            "phase3b_iters": settings.phase3b_iters,
        }
    )
    maybe_save_json(summary, run_dir, "summary.json")

    # Existing debug prints
    arr_top = dense_logs_final[0]["planner_actions"]["p_top"]
    arr_bottom = dense_logs_final[0]["planner_actions"]["p_bottom"]

    if arr_top.size:
        print("p_top (#decision_rows, #unique):", arr_top.shape[0], np.unique(arr_top, axis=0).shape[0])
        print("p_top sample rows:\n", arr_top[:5])

    if arr_bottom.size:
        print(
            "p_bottom (#decision_rows, #unique):",
            arr_bottom.shape[0],
            np.unique(arr_bottom, axis=0).shape[0],
        )
        print("p_bottom sample rows:\n", arr_bottom[:5])

    print_legal_rate_counts_at_next_tax_day(env_phase3b, fallback_period=settings.period)

    if settings.save_results and run_dir is not None:
        print(f"\nExperiment saved to: {run_dir}")
    else:
        print("\nExperiment not saved (save_results=False)")

    return {
        "experiment_name": experiment_name,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "summary": summary,
        "metrics_df": metrics_df,
        "dense_logs": dense_logs_final,
        "phase_checkpoints": portable_checkpoints,
        "trainers": {
            "phase2": trainer_phase2,
            "phase3a": trainer_phase3a,
            "phase3b": trainer_phase3b,
        },
        "envs": {
            "phase1": env_phase1,
            "phase2": env_phase2,
            "phase3a": env_phase3a,
            "phase3b": env_phase3b,
        },
    }


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> None:
    ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    results: Optional[Dict[str, Any]] = None
    try:
        results = run_experiment(SETTINGS)
    finally:
        # Uncomment this block if you want automatic cleanup every run.
        # if results is not None:
        #     for trainer in results["trainers"].values():
        #         trainer.stop()
        #     del results
        # gc.collect()
        # ray.shutdown()
        pass


if __name__ == "__main__":
    main()
