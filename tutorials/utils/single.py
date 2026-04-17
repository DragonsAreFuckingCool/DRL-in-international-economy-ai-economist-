import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def force_single_agent_move(env_obj, agent_id, target_region, max_tries=500):
    """
    Force a single agent to relocate once to a valid empty tile in the target region.
    Does not use the travel action or require CrossWaterTravel target attributes.
    """
    world = env_obj.env.world
    agent = world.agents[int(agent_id)]

    H, W = world.world_size
    split = H // 2

    if target_region == "top":
        r_min, r_max = 0, split
    elif target_region == "bottom":
        r_min, r_max = split + 1, H
    else:
        raise ValueError("target_region must be 'top' or 'bottom'")

    old_r, old_c = agent.loc

    # If already in target region, do nothing
    if (target_region == "top" and old_r < split) or (target_region == "bottom" and old_r > split):
        return {
            "agent": int(agent.idx),
            "from": (int(old_r), int(old_c)),
            "to": (int(old_r), int(old_c)),
            "new_region": target_region,
            "moved": False,
            "reason": "already_in_target_region",
        }

    # Try random valid locations in the target region
    for _ in range(max_tries):
        r = np.random.randint(r_min, r_max)
        c = np.random.randint(0, W)

        if world.maps.unoccupied[r, c] and world.can_agent_occupy(r, c, agent):
            new_r, new_c = world.set_agent_loc(agent, r, c)

            # update region state if present
            if "region" in agent.state:
                agent.state["region"] = "top" if int(new_r) < split else "bottom"

            # remove outstanding market listings/orders if supported
            try:
                cda = env_obj.env.get_component("ContinuousDoubleAuction")
                if cda is not None and hasattr(cda, "cancel_all_orders_for_agent"):
                    cda.cancel_all_orders_for_agent(agent.idx)
            except Exception:
                pass

            return {
                "agent": int(agent.idx),
                "from": (int(old_r), int(old_c)),
                "to": (int(new_r), int(new_c)),
                "new_region": agent.state.get("region", target_region),
                "moved": True,
            }

    raise RuntimeError(f"Could not find valid relocation target for agent {agent_id} to region {target_region}")

def generate_rollout_with_forced_move(
    trainer,
    env_obj,
    forced_agent_id=0,
    forced_timestep=200,
    forced_target_region="bottom",
    explore=False,
):
    def _compute_action(pid, obs, state):
        out = trainer.compute_action(
            observation=obs,
            state=state,
            policy_id=pid,
            full_fetch=False,
            explore=explore,
        )
        if isinstance(out, tuple):
            if len(out) >= 2:
                return out[0], out[1]
            return out[0], state
        return out, state

    obs = env_obj.reset(force_dense_logging=True)

    agent_states = {
        str(i): trainer.get_policy("a").get_initial_state()
        for i in range(env_obj.env.n_agents)
    }
    p_top_state = trainer.get_policy("p_top").get_initial_state()
    p_bottom_state = trainer.get_policy("p_bottom").get_initial_state()

    forced_move_log = None
    agent_utility_history = []

    for t in range(env_obj.env.episode_length):
        actions = {}

        for i in range(env_obj.env.n_agents):
            aid = str(i)
            a, ns = _compute_action("a", obs[aid], agent_states[aid])
            actions[aid] = a
            agent_states[aid] = ns

        a_top, p_top_state = _compute_action("p_top", obs["p_top"], p_top_state)
        a_bottom, p_bottom_state = _compute_action("p_bottom", obs["p_bottom"], p_bottom_state)
        actions["p_top"] = a_top
        actions["p_bottom"] = a_bottom

        obs, rew, done, info = env_obj.step(actions)

        if t == forced_timestep:
            forced_move_log = force_single_agent_move(
                env_obj=env_obj,
                agent_id=forced_agent_id,
                target_region=forced_target_region,
            )
            forced_move_log["timestep"] = t

        util_t = {}
        for agent in env_obj.env.world.agents:
            aid = str(agent.idx)
            util_t[aid] = float(env_obj.env.curr_optimization_metric.get(agent.idx, np.nan))
        agent_utility_history.append(util_t)

        if done.get("__all__", False):
            break

    dense_log = dict(env_obj.env.dense_log)

    if "states" in dense_log:
        n_states = len(dense_log["states"])
        n_utils = len(agent_utility_history)
        n_match = min(n_states, n_utils)

        for tt in range(n_match):
            for aid, util in agent_utility_history[tt].items():
                if aid in dense_log["states"][tt]:
                    dense_log["states"][tt][aid]["utility"] = util

    dense_log["forced_move"] = forced_move_log
    return dense_log


def get_forced_move_info(log):
    fm = log.get("forced_move", None)
    if fm is None:
        raise ValueError("No forced_move entry found in log.")
    return fm["agent"], fm.get("timestep", None), fm


def forced_move_agent_summary(log, window_before=100, window_after=100):
    fm = log.get("forced_move", None)
    if fm is None:
        raise ValueError("No forced_move entry found in log.")

    agent_id = int(fm["agent"])
    t0 = int(fm["timestep"])

    states = log["states"]
    t_start_before = max(0, t0 - window_before)
    t_end_before = t0
    t_start_after = t0 + 1
    t_end_after = min(len(states), t0 + 1 + window_after)

    aid = str(agent_id)

    def coin_at(t):
        s = states[t][aid]
        return s["inventory"]["Coin"] + s["escrow"]["Coin"]

    def labor_at(t):
        return states[t][aid]["endogenous"]["Labor"]

    def utility_series(t1, t2):
        vals = []
        for t in range(t1, t2):
            vals.append(states[t][aid].get("utility", np.nan))
        return np.array(vals, dtype=float)

    def count_builds(t1, t2):
        n = 0
        income = 0.0
        for t, builds in enumerate(log.get("Build", [])):
            if not (t1 <= t < t2):
                continue
            builds_ = builds.get("builds", []) if isinstance(builds, dict) else builds
            for b in builds_:
                if int(b["builder"]) == agent_id:
                    n += 1
                    income += float(b.get("income", 0.0))
        return n, income

    def count_trades(t1, t2):
        out = {
            "n_buy_wood": 0,
            "n_sell_wood": 0,
            "n_buy_stone": 0,
            "n_sell_stone": 0,
            "trade_cashflow": 0.0,
        }
        for t, trades in enumerate(log.get("Trade", [])):
            if not (t1 <= t < t2):
                continue
            trades_ = trades.get("trades", []) if isinstance(trades, dict) else trades
            for tr in trades_:
                commodity = tr["commodity"]
                if int(tr["buyer"]) == agent_id:
                    out[f"n_buy_{commodity.lower()}"] += 1
                    out["trade_cashflow"] -= float(tr["cost"])
                if int(tr["seller"]) == agent_id:
                    out[f"n_sell_{commodity.lower()}"] += 1
                    out["trade_cashflow"] += float(tr["income"])
        return out

    coin_before = coin_at(t_end_before - 1) - coin_at(t_start_before)
    coin_after = coin_at(t_end_after - 1) - coin_at(t_start_after)

    labor_before = labor_at(t_end_before - 1) - labor_at(t_start_before)
    labor_after = labor_at(t_end_after - 1) - labor_at(t_start_after)

    util_before = utility_series(t_start_before, t_end_before)
    util_after = utility_series(t_start_after, t_end_after)

    n_build_before, build_inc_before = count_builds(t_start_before, t_end_before)
    n_build_after, build_inc_after = count_builds(t_start_after, t_end_after)

    trade_before = count_trades(t_start_before, t_end_before)
    trade_after = count_trades(t_start_after, t_end_after)

    df = pd.DataFrame([
        {
            "period": "before_move",
            "agent": agent_id,
            "t_start": t_start_before,
            "t_end": t_end_before - 1,
            "coin_change": coin_before,
            "labor_change": labor_before,
            "avg_utility": np.nanmean(util_before),
            "n_builds": n_build_before,
            "build_income": build_inc_before,
            **trade_before,
        },
        {
            "period": "after_move",
            "agent": agent_id,
            "t_start": t_start_after,
            "t_end": t_end_after - 1,
            "coin_change": coin_after,
            "labor_change": labor_after,
            "avg_utility": np.nanmean(util_after),
            "n_builds": n_build_after,
            "build_income": build_inc_after,
            **trade_after,
        },
    ])

    return df

def forced_move_event_table(log, window=100):
    fm = log.get("forced_move", None)
    if fm is None:
        raise ValueError("No forced_move entry found in log.")

    agent_id = int(fm["agent"])
    t0 = int(fm["timestep"])

    rows = []

    for t in range(max(0, t0 - window), min(len(log["states"]), t0 + window + 1)):
        state = log["states"][t][str(agent_id)]

        row = {
            "t": t,
            "relative_t": t - t0,
            "region": state.get("region", np.nan),
            "coin": state["inventory"]["Coin"] + state["escrow"]["Coin"],
            "wood": state["inventory"]["Wood"] + state["escrow"]["Wood"],
            "stone": state["inventory"]["Stone"] + state["escrow"]["Stone"],
            "labor": state["endogenous"]["Labor"],
            "utility": state.get("utility", np.nan),
            "built": 0,
            "bought_wood": 0,
            "sold_wood": 0,
            "bought_stone": 0,
            "sold_stone": 0,
        }

        if "Build" in log and t < len(log["Build"]):
            builds = log["Build"][t]
            builds_ = builds.get("builds", []) if isinstance(builds, dict) else builds
            for b in builds_:
                if int(b["builder"]) == agent_id:
                    row["built"] += 1

        if "Trade" in log and t < len(log["Trade"]):
            trades = log["Trade"][t]
            trades_ = trades.get("trades", []) if isinstance(trades, dict) else trades
            for tr in trades_:
                commodity = tr["commodity"].lower()
                if int(tr["buyer"]) == agent_id:
                    row[f"bought_{commodity}"] += 1
                if int(tr["seller"]) == agent_id:
                    row[f"sold_{commodity}"] += 1

        rows.append(row)

    return pd.DataFrame(rows)



def plot_forced_move_timeseries(log, window=100):
    df = forced_move_event_table(log, window=window)
    fm = log["forced_move"]
    agent_id = fm["agent"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(df["relative_t"], df["coin"], label="Coin")
    axes[0].plot(df["relative_t"], df["utility"], label="Utility")
    axes[0].axvline(0, linestyle="--")
    axes[0].set_title(f"Agent {agent_id}: coin and utility around forced move")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["relative_t"], df["wood"], label="Wood")
    axes[1].plot(df["relative_t"], df["stone"], label="Stone")
    axes[1].axvline(0, linestyle="--")
    axes[1].set_title("Inventory")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(df["relative_t"], df["labor"], label="Labor")
    axes[2].axvline(0, linestyle="--")
    axes[2].set_title("Labor")
    axes[2].grid(True)

    axes[3].plot(df["relative_t"], df["bought_wood"] + df["sold_wood"], label="Wood trades")
    axes[3].plot(df["relative_t"], df["bought_stone"] + df["sold_stone"], label="Stone trades")
    axes[3].plot(df["relative_t"], df["built"], label="Builds")
    axes[3].axvline(0, linestyle="--")
    axes[3].set_title("Economic activity")
    axes[3].set_xlabel("Timesteps relative to forced move")
    axes[3].legend()
    axes[3].grid(True)

    fig.tight_layout()
    return fig

import numpy as np
import pandas as pd

def forced_move_system_summary_table(log, window_before=100, window_after=100):
    """
    Summary table for moved agent, other agents, and both planners
    before vs after a forced move.

    Returns a pandas DataFrame.
    """
    fm = log.get("forced_move", None)
    if fm is None:
        raise ValueError("No forced_move entry found in log.")

    moved_agent = int(fm["agent"])
    t0 = int(fm["timestep"])

    states = log["states"]
    first_state = states[0]
    agent_ids = sorted(int(k) for k in first_state.keys() if str(k).isdigit())

    t_before_0 = max(0, t0 - window_before)
    t_before_1 = t0
    t_after_0 = t0 + 1
    t_after_1 = min(len(states), t0 + 1 + window_after)

    def total_coin(state, aid):
        s = state[str(aid)]
        return s["inventory"]["Coin"] + s["escrow"]["Coin"]

    def labor(state, aid):
        return state[str(aid)]["endogenous"]["Labor"]

    def utility_series(aids, t_start, t_end):
        vals = []
        for t in range(t_start, t_end):
            for aid in aids:
                vals.append(states[t][str(aid)].get("utility", np.nan))
        return np.array(vals, dtype=float)

    def coin_change(aids, t_start, t_end):
        start = sum(total_coin(states[t_start], aid) for aid in aids)
        end = sum(total_coin(states[t_end - 1], aid) for aid in aids)
        return end - start

    def labor_change(aids, t_start, t_end):
        start = sum(labor(states[t_start], aid) for aid in aids)
        end = sum(labor(states[t_end - 1], aid) for aid in aids)
        return end - start

    def build_stats(aids, t_start, t_end):
        n_builds = 0
        build_income = 0.0
        for t, builds in enumerate(log.get("Build", [])):
            if not (t_start <= t < t_end):
                continue
            builds_ = builds.get("builds", []) if isinstance(builds, dict) else builds
            for b in builds_:
                if int(b["builder"]) in aids:
                    n_builds += 1
                    build_income += float(b.get("income", 0.0))
        return n_builds, build_income

    def trade_stats(aids, t_start, t_end):
        out = {
            "n_trades": 0,
            "trade_cashflow": 0.0,
            "n_buy_wood": 0,
            "n_sell_wood": 0,
            "n_buy_stone": 0,
            "n_sell_stone": 0,
        }
        for t, trades in enumerate(log.get("Trade", [])):
            if not (t_start <= t < t_end):
                continue
            trades_ = trades.get("trades", []) if isinstance(trades, dict) else trades
            for tr in trades_:
                buyer = int(tr["buyer"])
                seller = int(tr["seller"])
                commodity = tr["commodity"].lower()

                if buyer in aids:
                    out["n_trades"] += 1
                    out["trade_cashflow"] -= float(tr.get("cost", tr.get("price", 0.0)))
                    out[f"n_buy_{commodity}"] += 1
                if seller in aids:
                    out["n_trades"] += 1
                    out["trade_cashflow"] += float(tr.get("income", tr.get("price", 0.0)))
                    out[f"n_sell_{commodity}"] += 1
        return out



    groups = {
        "moved_agent": [moved_agent],
        "other_agents": [aid for aid in agent_ids if aid != moved_agent],
    }

    rows = []

    for label, aids in groups.items():
        for period_name, t_start, t_end in [
            ("before_move", t_before_0, t_before_1),
            ("after_move", t_after_0, t_after_1),
        ]:
            n_builds, build_income = build_stats(aids, t_start, t_end)
            tr = trade_stats(aids, t_start, t_end)
            util = utility_series(aids, t_start, t_end)

            rows.append({
                "group": label,
                "period": period_name,
                "n_agents": len(aids),
                "t_start": t_start,
                "t_end": t_end - 1,
                "coin_change": coin_change(aids, t_start, t_end),
                "labor_change": labor_change(aids, t_start, t_end),
                "avg_utility": float(np.nanmean(util)),
                "n_builds": n_builds,
                "build_income": build_income,
                **tr,
                "avg_planner_reward": np.nan,
                "sum_planner_reward": np.nan,
            })


    return pd.DataFrame(rows)


def baseline_system_summary_table(log, moved_agent, reference_timestep, window_before=100, window_after=100, episode_key=0):
    """
    Summary table for the original baseline run with NO forced move.
    Accepts either:
      - a single episode log with log["states"], or
      - a dict of episode logs, e.g. log[0]["states"].

    Parameters
    ----------
    log : dict
    moved_agent : int
    reference_timestep : int
    window_before : int
    window_after : int
    episode_key : int or str
        Which episode to use if log is a multi-episode dict.
    """

    # --- normalize to a single episode log ---
    if "states" in log:
        ep_log = log
    elif episode_key in log and isinstance(log[episode_key], dict) and "states" in log[episode_key]:
        ep_log = log[episode_key]
    else:
        # fallback: pick first entry that looks like an episode log
        ep_log = None
        for v in log.values():
            if isinstance(v, dict) and "states" in v:
                ep_log = v
                break
        if ep_log is None:
            raise ValueError("Could not find a single episode log with key 'states'.")

    t0 = int(reference_timestep)

    states = ep_log["states"]
    first_state = states[0]
    agent_ids = sorted(int(k) for k in first_state.keys() if str(k).isdigit())

    if moved_agent not in agent_ids:
        raise ValueError(f"moved_agent={moved_agent} not found in baseline log")

    t_before_0 = max(0, t0 - window_before)
    t_before_1 = t0
    t_after_0 = t0 + 1
    t_after_1 = min(len(states), t0 + 1 + window_after)

    def total_coin(state, aid):
        s = state[str(aid)]
        return s["inventory"]["Coin"] + s["escrow"]["Coin"]

    def labor(state, aid):
        return state[str(aid)]["endogenous"]["Labor"]

    def utility_series(aids, t_start, t_end):
        vals = []
        for t in range(t_start, t_end):
            for aid in aids:
                vals.append(states[t][str(aid)].get("utility", np.nan))
        return np.array(vals, dtype=float)

    def coin_change(aids, t_start, t_end):
        start = sum(total_coin(states[t_start], aid) for aid in aids)
        end = sum(total_coin(states[t_end - 1], aid) for aid in aids)
        return end - start

    def labor_change(aids, t_start, t_end):
        start = sum(labor(states[t_start], aid) for aid in aids)
        end = sum(labor(states[t_end - 1], aid) for aid in aids)
        return end - start

    def build_stats(aids, t_start, t_end):
        n_builds = 0
        build_income = 0.0
        for t, builds in enumerate(ep_log.get("Build", [])):
            if not (t_start <= t < t_end):
                continue
            builds_ = builds.get("builds", []) if isinstance(builds, dict) else builds
            for b in builds_:
                if int(b["builder"]) in aids:
                    n_builds += 1
                    build_income += float(b.get("income", 0.0))
        return n_builds, build_income

    def trade_stats(aids, t_start, t_end):
        out = {
            "n_trades": 0,
            "trade_cashflow": 0.0,
            "n_buy_wood": 0,
            "n_sell_wood": 0,
            "n_buy_stone": 0,
            "n_sell_stone": 0,
        }
        for t, trades in enumerate(ep_log.get("Trade", [])):
            if not (t_start <= t < t_end):
                continue
            trades_ = trades.get("trades", []) if isinstance(trades, dict) else trades
            for tr in trades_:
                buyer = int(tr["buyer"])
                seller = int(tr["seller"])
                commodity = tr["commodity"].lower()

                if buyer in aids:
                    out["n_trades"] += 1
                    out["trade_cashflow"] -= float(tr.get("cost", tr.get("price", 0.0)))
                    out[f"n_buy_{commodity}"] += 1
                if seller in aids:
                    out["n_trades"] += 1
                    out["trade_cashflow"] += float(tr.get("income", tr.get("price", 0.0)))
                    out[f"n_sell_{commodity}"] += 1
        return out

    def planner_stats(pid, t_start, t_end):
        arr = np.array(ep_log.get("planner_rewards", {}).get(pid, []), dtype=float)
        if len(arr) == 0:
            return {"avg_planner_reward": np.nan, "sum_planner_reward": np.nan}

        t_start = min(t_start, len(arr))
        t_end = min(t_end, len(arr))
        if t_start >= t_end:
            return {"avg_planner_reward": np.nan, "sum_planner_reward": np.nan}

        seg = arr[t_start:t_end]
        return {
            "avg_planner_reward": float(np.nanmean(seg)),
            "sum_planner_reward": float(np.nansum(seg)),
        }

    groups = {
        "moved_agent": [moved_agent],
        "other_agents": [aid for aid in agent_ids if aid != moved_agent],
    }

    rows = []

    for label, aids in groups.items():
        for period_name, t_start, t_end in [
            ("before_move", t_before_0, t_before_1),
            ("after_move", t_after_0, t_after_1),
        ]:
            n_builds, build_income = build_stats(aids, t_start, t_end)
            tr = trade_stats(aids, t_start, t_end)
            util = utility_series(aids, t_start, t_end)

            rows.append({
                "group": label,
                "period": period_name,
                "n_agents": len(aids),
                "t_start": t_start,
                "t_end": t_end - 1,
                "coin_change": coin_change(aids, t_start, t_end),
                "labor_change": labor_change(aids, t_start, t_end),
                "avg_utility": float(np.nanmean(util)),
                "n_builds": n_builds,
                "build_income": build_income,
                **tr,
                "avg_planner_reward": np.nan,
                "sum_planner_reward": np.nan,
            })

    for pid in ["p_top", "p_bottom"]:
        for period_name, t_start, t_end in [
            ("before_move", t_before_0, t_before_1),
            ("after_move", t_after_0, t_after_1),
        ]:
            pr = planner_stats(pid, t_start, t_end)
            rows.append({
                "group": pid,
                "period": period_name,
                "n_agents": np.nan,
                "t_start": t_start,
                "t_end": t_end - 1,
                "coin_change": np.nan,
                "labor_change": np.nan,
                "avg_utility": np.nan,
                "n_builds": np.nan,
                "build_income": np.nan,
                "n_trades": np.nan,
                "trade_cashflow": np.nan,
                "n_buy_wood": np.nan, 
                "n_sell_wood": np.nan,
                "n_buy_stone": np.nan,
                "n_sell_stone": np.nan,
                **pr,
            })

    return pd.DataFrame(rows)


import copy

def _compute_actions_from_obs(trainer, env_obj, obs, agent_states, p_top_state, p_bottom_state, explore=False):
    def _compute_action(pid, obs_i, state):
        out = trainer.compute_action(
            observation=obs_i,
            state=state,
            policy_id=pid,
            full_fetch=False,
            explore=explore,
        )
        if isinstance(out, tuple):
            if len(out) >= 2:
                return out[0], out[1]
            return out[0], state
        return out, state

    actions = {}

    for i in range(env_obj.env.n_agents):
        aid = str(i)
        a, ns = _compute_action("a", obs[aid], agent_states[aid])
        actions[aid] = a
        agent_states[aid] = ns

    a_top, p_top_state = _compute_action("p_top", obs["p_top"], p_top_state)
    a_bottom, p_bottom_state = _compute_action("p_bottom", obs["p_bottom"], p_bottom_state)

    actions["p_top"] = a_top
    actions["p_bottom"] = a_bottom

    return actions, agent_states, p_top_state, p_bottom_state


def _inject_utility_into_dense_log(dense_log, agent_utility_history):
    if "states" not in dense_log:
        return dense_log

    n_states = len(dense_log["states"])
    n_utils = len(agent_utility_history)
    n_match = min(n_states, n_utils)

    for t in range(n_match):
        for aid, util in agent_utility_history[t].items():
            if aid in dense_log["states"][t]:
                dense_log["states"][t][aid]["utility"] = util

    return dense_log


def force_single_agent_move(env_obj, agent_id, target_region, max_tries=500):
    world = env_obj.env.world
    agent = world.agents[int(agent_id)]

    H, W = world.world_size
    split = H // 2

    if target_region == "top":
        r_min, r_max = 0, split
    elif target_region == "bottom":
        r_min, r_max = split + 1, H
    else:
        raise ValueError("target_region must be 'top' or 'bottom'")

    old_r, old_c = agent.loc

    if (target_region == "top" and old_r < split) or (target_region == "bottom" and old_r > split):
        return {
            "agent": int(agent.idx),
            "from": (int(old_r), int(old_c)),
            "to": (int(old_r), int(old_c)),
            "new_region": target_region,
            "moved": False,
            "reason": "already_in_target_region",
        }

    for _ in range(max_tries):
        r = np.random.randint(r_min, r_max)
        c = np.random.randint(0, W)

        if world.maps.unoccupied[r, c] and world.can_agent_occupy(r, c, agent):
            new_r, new_c = world.set_agent_loc(agent, r, c)

            if "region" in agent.state:
                agent.state["region"] = "top" if int(new_r) < split else "bottom"

            try:
                cda = env_obj.env.get_component("ContinuousDoubleAuction")
                if cda is not None and hasattr(cda, "cancel_all_orders_for_agent"):
                    cda.cancel_all_orders_for_agent(agent.idx)
            except Exception:
                pass

            return {
                "agent": int(agent.idx),
                "from": (int(old_r), int(old_c)),
                "to": (int(new_r), int(new_c)),
                "new_region": agent.state.get("region", target_region),
                "moved": True,
            }

    raise RuntimeError(f"Could not find valid relocation target for agent {agent_id} to region {target_region}")


def generate_paired_baseline_forced_rollouts(
    trainer,
    env_obj,
    forced_agent_id=0,
    forced_timestep=200,
    forced_target_region="bottom",
    explore=False,
):
    """
    Create two rollouts with identical pre-move trajectories:
      1) baseline continuation
      2) forced-move continuation

    Returns
    -------
    baseline_log, forced_log
    """

    # --- reset once ---
    obs = env_obj.reset(force_dense_logging=True)

    agent_states = {
        str(i): trainer.get_policy("a").get_initial_state()
        for i in range(env_obj.env.n_agents)
    }
    p_top_state = trainer.get_policy("p_top").get_initial_state()
    p_bottom_state = trainer.get_policy("p_bottom").get_initial_state()

    pre_move_utility_history = []
    pre_move_planner_rewards_top = []
    pre_move_planner_rewards_bottom = []

    # --- run until forced_timestep ---
    for t in range(forced_timestep + 1):
        actions, agent_states, p_top_state, p_bottom_state = _compute_actions_from_obs(
            trainer, env_obj, obs, agent_states, p_top_state, p_bottom_state, explore=explore
        )

        obs, rew, done, info = env_obj.step(actions)

        util_t = {}
        for agent in env_obj.env.world.agents:
            aid = str(agent.idx)
            util_t[aid] = float(env_obj.env.curr_optimization_metric.get(agent.idx, np.nan))
        pre_move_utility_history.append(util_t)

        pre_move_planner_rewards_top.append(rew.get("p_top", np.nan))
        pre_move_planner_rewards_bottom.append(rew.get("p_bottom", np.nan))

        if done.get("__all__", False):
            raise RuntimeError("Episode ended before forced_timestep; choose a smaller forced_timestep.")

    # --- snapshot branch point ---
    env_baseline = copy.deepcopy(env_obj)
    env_forced = copy.deepcopy(env_obj)

    obs_baseline = copy.deepcopy(obs)
    obs_forced = copy.deepcopy(obs)

    agent_states_baseline = copy.deepcopy(agent_states)
    agent_states_forced = copy.deepcopy(agent_states)

    p_top_state_baseline = copy.deepcopy(p_top_state)
    p_top_state_forced = copy.deepcopy(p_top_state)

    p_bottom_state_baseline = copy.deepcopy(p_bottom_state)
    p_bottom_state_forced = copy.deepcopy(p_bottom_state)

    # =========================
    # BASELINE continuation
    # =========================
    baseline_utility_history = copy.deepcopy(pre_move_utility_history)
    baseline_top_rewards = copy.deepcopy(pre_move_planner_rewards_top)
    baseline_bottom_rewards = copy.deepcopy(pre_move_planner_rewards_bottom)

    for t in range(forced_timestep + 1, env_baseline.env.episode_length):
        actions, agent_states_baseline, p_top_state_baseline, p_bottom_state_baseline = _compute_actions_from_obs(
            trainer,
            env_baseline,
            obs_baseline,
            agent_states_baseline,
            p_top_state_baseline,
            p_bottom_state_baseline,
            explore=explore,
        )

        obs_baseline, rew, done, info = env_baseline.step(actions)

        util_t = {}
        for agent in env_baseline.env.world.agents:
            aid = str(agent.idx)
            util_t[aid] = float(env_baseline.env.curr_optimization_metric.get(agent.idx, np.nan))
        baseline_utility_history.append(util_t)

        baseline_top_rewards.append(rew.get("p_top", np.nan))
        baseline_bottom_rewards.append(rew.get("p_bottom", np.nan))

        if done.get("__all__", False):
            break

    baseline_log = dict(env_baseline.env.dense_log)
    baseline_log = _inject_utility_into_dense_log(baseline_log, baseline_utility_history)
    baseline_log["planner_rewards"] = {
        "p_top": baseline_top_rewards,
        "p_bottom": baseline_bottom_rewards,
    }
    baseline_log["paired_reference"] = {
        "agent": int(forced_agent_id),
        "timestep": int(forced_timestep),
        "target_region": forced_target_region,
        "type": "baseline",
    }

    # =========================
    # FORCED continuation
    # =========================
    forced_utility_history = copy.deepcopy(pre_move_utility_history)
    forced_top_rewards = copy.deepcopy(pre_move_planner_rewards_top)
    forced_bottom_rewards = copy.deepcopy(pre_move_planner_rewards_bottom)

    move_log = force_single_agent_move(
        env_forced,
        agent_id=forced_agent_id,
        target_region=forced_target_region,
    )
    move_log["timestep"] = int(forced_timestep)

    for t in range(forced_timestep + 1, env_forced.env.episode_length):
        actions, agent_states_forced, p_top_state_forced, p_bottom_state_forced = _compute_actions_from_obs(
            trainer,
            env_forced,
            obs_forced,
            agent_states_forced,
            p_top_state_forced,
            p_bottom_state_forced,
            explore=explore,
        )

        obs_forced, rew, done, info = env_forced.step(actions)

        util_t = {}
        for agent in env_forced.env.world.agents:
            aid = str(agent.idx)
            util_t[aid] = float(env_forced.env.curr_optimization_metric.get(agent.idx, np.nan))
        forced_utility_history.append(util_t)

        forced_top_rewards.append(rew.get("p_top", np.nan))
        forced_bottom_rewards.append(rew.get("p_bottom", np.nan))

        if done.get("__all__", False):
            break

    forced_log = dict(env_forced.env.dense_log)
    forced_log = _inject_utility_into_dense_log(forced_log, forced_utility_history)
    forced_log["planner_rewards"] = {
        "p_top": forced_top_rewards,
        "p_bottom": forced_bottom_rewards,
    }
    forced_log["forced_move"] = move_log

    return baseline_log, forced_log


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_forced_move_utilities_system(log, window=100):
    """
    Plot utility development before and after forced move for:
    - moved agent
    - average of other agents
    - p_top reward
    - p_bottom reward
    """
    fm = log.get("forced_move", None)
    if fm is None:
        raise ValueError("No forced_move entry found in log.")

    moved_agent = int(fm["agent"])
    t0 = int(fm["timestep"])

    states = log["states"]
    first_state = states[0]
    agent_ids = sorted(int(k) for k in first_state.keys() if str(k).isdigit())
    other_agents = [aid for aid in agent_ids if aid != moved_agent]

    t_start = max(0, t0 - window)
    t_end = min(len(states), t0 + window + 1)

    rows = []
    for t in range(t_start, t_end):
        rel_t = t - t0

        moved_u = states[t][str(moved_agent)].get("utility", np.nan)

        others_u = []
        for aid in other_agents:
            others_u.append(states[t][str(aid)].get("utility", np.nan))
        others_u = np.array(others_u, dtype=float)

        p_top_arr = np.array(log.get("planner_rewards", {}).get("p_top", []), dtype=float)
        p_bottom_arr = np.array(log.get("planner_rewards", {}).get("p_bottom", []), dtype=float)

        p_top_val = p_top_arr[t] if t < len(p_top_arr) else np.nan
        p_bottom_val = p_bottom_arr[t] if t < len(p_bottom_arr) else np.nan

        rows.append({
            "t": t,
            "relative_t": rel_t,
            "moved_agent_utility": moved_u,
            "other_agents_avg_utility": float(np.nanmean(others_u)),
            "p_top_reward": p_top_val,
            "p_bottom_reward": p_bottom_val,
        })

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df["relative_t"], df["moved_agent_utility"], label="Moved agent utility")
    axes[0].plot(df["relative_t"], df["other_agents_avg_utility"], label="Other agents avg utility")
    axes[0].axvline(0, linestyle="--")
    axes[0].set_title("Agent utility before and after forced move")
    axes[0].set_ylabel("Utility")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(df["relative_t"], df["p_top_reward"], label="p_top reward")
    axes[1].plot(df["relative_t"], df["p_bottom_reward"], label="p_bottom reward")
    axes[1].axvline(0, linestyle="--")
    axes[1].set_title("Planner rewards before and after forced move")
    axes[1].set_xlabel("Timesteps relative to forced move")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    return fig, df

def forced_move_delta_table(log, window_before=100, window_after=100):
    df = forced_move_system_summary_table(
        log,
        window_before=window_before,
        window_after=window_after,
    )

    numeric_cols = [
        c for c in df.columns
        if c not in ["group", "period"]
    ]

    before = df[df["period"] == "before_move"].set_index("group")
    after = df[df["period"] == "after_move"].set_index("group")

    delta = after[numeric_cols] - before[numeric_cols]
    delta = delta.reset_index()
    return delta