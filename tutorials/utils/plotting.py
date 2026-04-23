# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pickle
import pandas as pd

from ai_economist.foundation import landmarks, resources

def numeric_agent_ids_from_states(state_dict):
    """
    Return sorted list of numeric agent IDs (ints), ignoring planners like 'p', 'p_top', 'p_bottom'.
    Works for both legacy (single planner) and your 2-planner extension.
    """
    return sorted([int(k) for k in state_dict.keys() if str(k).isdigit()])


def plot_map(maps, locs, ax=None, cmap_order=None, show_water=True):
    """Universal map renderer that works for live env (Maps) and dense logs (dict).
    Handles large worlds cleanly and can optionally hide water for debugging.
    """
    # Helpers
    def _map_keys(m):
        # Both Maps and dict expose .keys()
        return list(m.keys()) if hasattr(m, "keys") else []

    def _map_get(m, key, default=None):
        try:
            return m.get(key)
        except Exception:
            return default

    keys = _map_keys(maps)
    if not keys:
        raise ValueError("plot_map: No map keys found to infer world size.")

    # Pick any entity to infer world size
    example_map = _map_get(maps, keys[0])
    world_size = np.array(example_map).shape

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(min(0.4*world_size[1], 16), min(0.4*world_size[0], 16)))
    else:
        ax.cla()

    tmp = np.zeros((3, world_size[0], world_size[1]), dtype=float)
    n_agents = len(locs)
    cmap = plt.get_cmap("jet", n_agents)

    if cmap_order is None:
        cmap_order = list(range(n_agents))

    # Dynamically draw all non-source entities
    for key in keys:
        if key is None:
            continue
        if "source" in str(key).lower():
            continue

        arr = _map_get(maps, key)
        if arr is None:
            continue

        # Skip water if we don't want to show it
        if not show_water and str(key).lower() == "water":
            continue

        if resources.has(key):
            rdef = resources.get(key)
            if rdef.collectible:
                a = np.array(arr, dtype=float)
                tmp += rdef.color[:, None, None] * a[None]

        elif landmarks.has(key):
            ldef = landmarks.get(key)
            a = np.array(arr)
            if a.ndim == 2:
                tmp += ldef.color[:, None, None] * a[None].astype(float)
            elif isinstance(arr, dict):  # e.g., House in logs
                health = np.array(arr.get("health", np.zeros(world_size)), dtype=float)
                tmp += ldef.color[:, None, None] * health[None]

    # Agent-owned houses with per-agent colors
    if "House" in keys:
        house = _map_get(maps, "House")
        if isinstance(house, dict):
            house_idx = np.array(house.get("owner", np.zeros(world_size)), dtype=int)
            house_health = np.array(house.get("health", np.zeros(world_size)), dtype=float)
        else:
            h = np.array(house, dtype=float)
            house_idx = np.zeros_like(h, dtype=int)
            house_health = h

        for i in range(n_agents):
            houses = (house_idx == cmap_order[i]) * house_health
            col = np.array(cmap(i)[:3])
            tmp += col[:, None, None] * houses[None]

    # brighten + clip
    tmp = 0.7 * tmp + 0.3
    tmp = np.transpose(np.minimum(tmp, 1.0), [1, 2, 0])

    im = ax.imshow(tmp, vmax=1.0, interpolation='nearest')  # keep grid crisp
    ax.set_aspect('equal')  # no stretching

    # Agent markers scale with axes size, not world size
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    pix_h = bbox.height * ax.figure.dpi
    for i in range(n_agents):
        r, c = locs[cmap_order[i]]
        col = np.array(cmap(i)[:3])
        ax.plot(c, r, "o", markersize=max(4, pix_h * 0.03), color="w")
        ax.plot(c, r, "*", markersize=max(3, pix_h * 0.02), color=col)

    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_env_state(env, ax=None, remap_key=None):
    maps = env.world.maps
    locs = [agent.loc for agent in env.world.agents]

    if remap_key is None:
        cmap_order = None
    else:
        assert isinstance(remap_key, str)
        cmap_order = np.argsort(
            [agent.state[remap_key] for agent in env.world.agents]
        ).tolist()

    plot_map(maps, locs, ax, cmap_order)

def plot_log_state(dense_log, t, ax=None, remap_key=None):
    maps = dense_log["world"][t]
    states = dense_log["states"][t]

    # --- MULTI-PLANNER SAFE: only numeric agent ids ---
    agent_ids = numeric_agent_ids_from_states(states)
    n_agents = len(agent_ids)

    # Agent locations in the order of agent_ids
    locs = [states[str(i)]["loc"] for i in agent_ids]

    # Build color order (optional remap)
    if remap_key is None:
        # No remap: keep the agent_ids’ order; colormap expects 0..n_agents-1 positions
        cmap_order = None
    else:
        assert isinstance(remap_key, str)
        key_val = np.array([dense_log["states"][0][str(i)][remap_key] for i in agent_ids])
        # Order refers to positions within the current agent_ids array
        cmap_order = np.argsort(key_val).tolist()

    plot_map(maps, locs, ax, cmap_order)



def _format_logs_and_eps(dense_logs, eps):
    if isinstance(dense_logs, dict):
        return [dense_logs], [0]
    else:
        assert isinstance(dense_logs, (list, tuple))

    if isinstance(eps, (list, tuple)):
        return dense_logs, list(eps)
    elif isinstance(eps, (int, float)):
        return dense_logs, [int(eps)]
    elif eps is None:
        return dense_logs, list(range(np.minimum(len(dense_logs), 16)))
    else:
        raise NotImplementedError


def vis_world_array(dense_logs, ts, eps=None, axes=None, remap_key=None):
    dense_logs, eps = _format_logs_and_eps(dense_logs, eps)
    if isinstance(ts, (int, float)):
        ts = [ts]

    if axes is None:
        fig, axes = plt.subplots(
            len(eps),
            len(ts),
            figsize=(np.minimum(3.2 * len(ts), 16), 3 * len(eps)),
            squeeze=False,
        )

    else:
        fig = None

        if len(ts) == 1 and len(eps) == 1:
            axes = np.array([[axes]]).reshape(1, 1)
        else:
            try:
                axes = np.array(axes).reshape(len(eps), len(ts))
            except ValueError:
                print("Could not reshape provided axes array into the necessary shape!")
                raise

    for ti, t in enumerate(ts):
        for ei, ep in enumerate(eps):
            plot_log_state(dense_logs[ep], t, ax=axes[ei, ti], remap_key=remap_key)

    for ax, t in zip(axes[0], ts):
        ax.set_title("T = {}".format(t))
    for ax, ep in zip(axes[:, 0], eps):
        ax.set_ylabel("Episode {}".format(ep))

    return fig


def vis_world_range(
    dense_logs, t0=0, tN=None, N=5, eps=None, axes=None, remap_key=None
):
    dense_logs, eps = _format_logs_and_eps(dense_logs, eps)

    viable_ts = np.array([i for i, w in enumerate(dense_logs[0]["world"]) if w])
    if tN is None:
        tN = viable_ts[-1]
    assert 0 <= t0 < tN
    target_ts = np.linspace(t0, tN, N).astype(np.int32)

    ts = set()
    for tt in target_ts:
        closest = np.argmin(np.abs(tt - viable_ts))
        ts.add(viable_ts[closest])
    ts = sorted(list(ts))
    if axes is not None:
        axes = axes[: len(ts)]
    return vis_world_array(dense_logs, ts, axes=axes, eps=eps, remap_key=remap_key)


def vis_builds(dense_logs, eps=None, ax=None):
    dense_logs, eps = _format_logs_and_eps(dense_logs, eps)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 3))
    cmap = plt.get_cmap("jet", len(eps))
    for i, ep in enumerate(eps):
        ax.plot(
            np.cumsum([len(b["builds"]) for b in dense_logs[ep]["Build"]]),
            color=cmap(i),
            label="Ep {}".format(ep),
        )
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=0)


def trade_str(c_trades, resource, agent, income=True):
    if income:
        p = [x["income"] for x in c_trades[resource] if x["seller"] == agent]
    else:
        p = [x["cost"] for x in c_trades[resource] if x["buyer"] == agent]
    if len(p) > 0:
        return "{:6.2f} (n={:3d})".format(np.mean(p), len(p))
    else:
        tmp = "~" * 8
        tmp = (" ") * 3 + tmp + (" ") * 3
        return tmp


def full_trade_str(c_trades, resource, a_indices, income=True):
    s_head = "{} ({})".format("Income" if income else "Cost", resource)
    ac_strings = [trade_str(c_trades, resource, buyer, income) for buyer in a_indices]
    s_tail = " | ".join(ac_strings)
    return "{:<15}: {}".format(s_head, s_tail)


def build_str(all_builds, agent):
    p = [x["income"] for x in all_builds if x["builder"] == agent]
    if len(p) > 0:
        return "{:6.2f} (n={:3d})".format(np.mean(p), len(p))
    else:
        tmp = "~" * 8
        tmp = (" ") * 3 + tmp + (" ") * 3
        return tmp


def full_build_str(all_builds, a_indices):
    s_head = "Income (Build)"
    ac_strings = [build_str(all_builds, builder) for builder in a_indices]
    s_tail = " | ".join(ac_strings)
    return "{:<15}: {}".format(s_head, s_tail)


def header_str(n_agents):
    s_head = ("_" * 15) + ":_"
    s_tail = "_|_".join([" Agent {:2d} ____".format(i) for i in range(n_agents)])
    return s_head + s_tail


def report(c_trades, all_builds, n_agents, a_indices=None):
    if a_indices is None:
        a_indices = list(range(n_agents))
    print(header_str(n_agents))
    resources = ["Wood", "Stone"]
    if c_trades is not None:
        for resource in resources:
            print(full_trade_str(c_trades, resource, a_indices, income=False))
        print("")
        for resource in resources:
            print(full_trade_str(c_trades, resource, a_indices, income=True))
    print(full_build_str(all_builds, a_indices))

def breakdown(log, remap_key=None):
    """
    Multi-planner safe breakdown plotter:
      - Counts only numeric agent IDs (ignores planners).
      - Supports legacy single-planner and your 2-planner extension.
    """
    # Snapshot montage figure (uses plot_log_state which is already updated to ignore planners)
    fig0 = vis_world_range(log, remap_key=remap_key)

    # --- Only numeric mobile agents ---
    agent_ids = numeric_agent_ids_from_states(log["states"][0])
    n = len(agent_ids)
    trading_active = "Trade" in log

    # Agent ordering (optionally by remap_key)
    if remap_key is None:
        aidx = agent_ids[:]  # keep numeric agent IDs as-is
    else:
        assert isinstance(remap_key, str)
        key_vals = np.array([log["states"][0][str(i)][remap_key] for i in agent_ids])
        order = np.argsort(key_vals).tolist()
        aidx = [agent_ids[j] for j in order]

    # --- Collect builds over time ---
    all_builds = []
    for t, builds in enumerate(log.get("Build", [])):
        if isinstance(builds, dict):
            builds_ = builds.get("builds", [])
        else:
            builds_ = builds
        for build in builds_:
            this_build = {"t": t}
            this_build.update(build)
            all_builds.append(this_build)

    # --- Collect trades if present ---
    if trading_active:
        c_trades = {"Stone": [], "Wood": []}
        for t, trades in enumerate(log["Trade"]):
            if isinstance(trades, dict):
                trades_ = trades.get("trades", [])
            else:
                trades_ = trades
            for trade in trades_:
                this_trade = {
                    "t": t,
                    "t_ask": t - trade.get("ask_lifetime", 0),
                    "t_bid": t - trade.get("bid_lifetime", 0),
                }
                this_trade.update(trade)
                c_trades[trade["commodity"]].append(this_trade)

        incomes = {
            "Sell Stone": [
                sum([tr["income"] for tr in c_trades["Stone"] if tr["seller"] == aidx[i]])
                for i in range(n)
            ],
            "Buy Stone": [
                sum([-tr["price"] for tr in c_trades["Stone"] if tr["buyer"] == aidx[i]])
                for i in range(n)
            ],
            "Sell Wood": [
                sum([tr["income"] for tr in c_trades["Wood"] if tr["seller"] == aidx[i]])
                for i in range(n)
            ],
            "Buy Wood": [
                sum([-tr["price"] for tr in c_trades["Wood"] if tr["buyer"] == aidx[i]])
                for i in range(n)
            ],
            "Build": [
                sum([b["income"] for b in all_builds if b["builder"] == aidx[i]])
                for i in range(n)
            ],
        }
    else:
        c_trades = None
        incomes = {
            "Build": [
                sum([b["income"] for b in all_builds if b["builder"] == aidx[i]])
                for i in range(n)
            ],
        }

    # Total income per agent (position-aligned with aidx)
    incomes["Total"] = np.stack([v for v in incomes.values()]).sum(axis=0)

    # Endowments at episode end
    endows = [
        int(
            log["states"][-1][str(aidx[i])]["inventory"]["Coin"]
            + log["states"][-1][str(aidx[i])]["escrow"]["Coin"]
        )
        for i in range(n)
    ]

    # Text report
    report(c_trades, all_builds, n, aidx)

    # --- Time series plots: resources + labor ---
    cmap = plt.get_cmap("jet", n)
    rs = ["Wood", "Stone", "Coin"]
    fig1, axes = plt.subplots(1, len(rs) + 1, figsize=(16, 4), sharey=False)

    for r, ax in zip(rs, axes):
        for i in range(n):
            ax.plot(
                [
                    x[str(aidx[i])]["inventory"][r] + x[str(aidx[i])]["escrow"][r]
                    for x in log["states"]
                ],
                label=i,
                color=cmap(i),
            )
        ax.set_title(r)
        ax.legend()
        ax.grid(True)

    ax = axes[-1]
    for i in range(n):
        ax.plot(
            [x[str(aidx[i])]["endogenous"]["Labor"] for x in log["states"]],
            label=i,
            color=cmap(i),
        )
    ax.set_title("Labor")
    ax.legend()
    ax.grid(True)

    # --- Movement tracks (subsampled) ---
    tmp = np.array(log["world"][0]["Stone"])
    n_small = np.minimum(4, n)
    fig2, axes = plt.subplots(
        2 if trading_active else 1,
        n_small,
        figsize=(16, 8 if trading_active else 4),
        sharex="row",
        sharey="row",
        squeeze=False,
    )

    # Trajectories
    for i, ax in enumerate(axes[0]):
        rows = np.array([x[str(aidx[i])]["loc"][0] for x in log["states"]]) * -1
        cols = np.array([x[str(aidx[i])]["loc"][1] for x in log["states"]])
        ax.plot(cols[::20], rows[::20])
        ax.plot(cols[0], rows[0], "r*", markersize=15)
        ax.plot(cols[-1], rows[-1], "g*", markersize=15)
        ax.set_title("Agent {}".format(i))
        ax.set_xlim([-1, 1 + tmp.shape[1]])
        ax.set_ylim([-(1 + tmp.shape[0]), 1])

    # Trade timelines (if any)
    if trading_active:
        for i, ax in enumerate(axes[1]):
            for r in ["Wood", "Stone"]:
                # Seller incomes
                tmp = [(s["t"], s["income"]) for s in c_trades[r] if s["seller"] == aidx[i]]
                if tmp:
                    ts, prices = [np.array(x) for x in zip(*tmp)]
                    ax.plot(
                        np.stack([ts, ts]),
                        np.stack([np.zeros_like(prices), prices]),
                        color=resources.get(r).color,
                    )
                    ax.plot(ts, prices, ".", color=resources.get(r).color, markersize=12)

                # Buyer costs (negative)
                tmp = [(s["t"], -s["cost"]) for s in c_trades[r] if s["buyer"] == aidx[i]]
                if tmp:
                    ts, prices = [np.array(x) for x in zip(*tmp)]
                    ax.plot(
                        np.stack([ts, ts]),
                        np.stack([np.zeros_like(prices), prices]),
                        color=resources.get(r).color,
                    )
                    ax.plot(ts, prices, ".", color=resources.get(r).color, markersize=12)

            ax.plot([-20, len(log["states"]) + 19], [0, 0], "w-")
            ax.set_xlim([-20, len(log["states"]) + 19])
            ax.grid(True)
            ax.set_facecolor([0.3, 0.3, 0.3])

    return (fig0, fig1, fig2), incomes, endows, c_trades, all_builds

def plot_for_each_n(y_fun, n, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    cmap = plt.get_cmap("jet", n)
    for i in range(n):
        ax.plot(y_fun(i), color=cmap(i), label=i)
    ax.legend()
    ax.grid(True)

def breakdown_all_agents(log, remap_key="build_payment", n_cols=4):
    """
    Like breakdown(...), but:
      - shows ALL mobile agents
      - uses up to n_cols columns
      - marks travel as lower-opacity movement
      - defines travel simply as crossing the middle divider
      - avoids shadow lines by plotting each step separately
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Snapshot montage figure
    fig0 = vis_world_range(log, remap_key=remap_key)

    # --- Only numeric mobile agents ---
    agent_ids = numeric_agent_ids_from_states(log["states"][0])
    n = len(agent_ids)
    trading_active = "Trade" in log

    # Agent ordering
    if remap_key is None:
        aidx = agent_ids[:]
    else:
        key_vals = np.array([log["states"][0][str(i)][remap_key] for i in agent_ids])
        order = np.argsort(key_vals).tolist()
        aidx = [agent_ids[j] for j in order]

    # Labels with skill-rank annotation
    rank_labels = []
    build_payment = {}
    gather_mults = {}

    for aid in aidx:
        s = log["states"][0][str(aid)]
        build_payment[aid] = s.get("build_payment", np.nan)
        p = s.get("bonus_gather_prob", np.nan)
        gather_mults[aid] = 1.0 + p if np.isfinite(p) else np.nan

    for i, aid in enumerate(aidx):
        base = f"Agent {aid}"
        if i == 0:
            base += " (Lowest Skill)"
        elif i == len(aidx) - 1:
            base += " (Highest Skill)"

        build = build_payment.get(aid, np.nan)
        gather = gather_mults.get(aid, np.nan)
        skill_line = f"\nBuild: {build:.2f} | Gather: {gather:.2f}"
        rank_labels.append(base + skill_line)

    # --- Collect builds over time ---
    all_builds = []
    for t, builds in enumerate(log.get("Build", [])):
        builds_ = builds.get("builds", []) if isinstance(builds, dict) else builds
        for build in builds_:
            this_build = {"t": t}
            this_build.update(build)
            all_builds.append(this_build)

    # --- Collect trades if present ---
    if trading_active:
        c_trades = {"Stone": [], "Wood": []}
        for t, trades in enumerate(log.get("Trade", [])):
            trades_ = trades.get("trades", []) if isinstance(trades, dict) else trades
            for trade in trades_:
                this_trade = {
                    "t": t,
                    "t_ask": t - trade.get("ask_lifetime", 0),
                    "t_bid": t - trade.get("bid_lifetime", 0),
                }
                this_trade.update(trade)
                c_trades[trade["commodity"]].append(this_trade)

        incomes = {
            "Sell Stone": [
                sum(tr["income"] for tr in c_trades["Stone"] if tr["seller"] == aidx[i])
                for i in range(n)
            ],
            "Buy Stone": [
                sum(-tr["price"] for tr in c_trades["Stone"] if tr["buyer"] == aidx[i])
                for i in range(n)
            ],
            "Sell Wood": [
                sum(tr["income"] for tr in c_trades["Wood"] if tr["seller"] == aidx[i])
                for i in range(n)
            ],
            "Buy Wood": [
                sum(-tr["price"] for tr in c_trades["Wood"] if tr["buyer"] == aidx[i])
                for i in range(n)
            ],
            "Build": [
                sum(b["income"] for b in all_builds if b["builder"] == aidx[i])
                for i in range(n)
            ],
        }
    else:
        c_trades = None
        incomes = {
            "Build": [
                sum(b["income"] for b in all_builds if b["builder"] == aidx[i])
                for i in range(n)
            ],
        }

    incomes["Total"] = np.stack([v for v in incomes.values()]).sum(axis=0)

    endows = [
        int(
            log["states"][-1][str(aidx[i])]["inventory"]["Coin"]
            + log["states"][-1][str(aidx[i])]["escrow"]["Coin"]
        )
        for i in range(n)
    ]

    report(c_trades, all_builds, n, aidx)

    # --- Time series plots: resources + labor + utility ---
    cmap = plt.get_cmap("jet", n)
    rs = ["Wood", "Stone", "Coin"]

    fig1, axes = plt.subplots(1, len(rs) + 2, figsize=(22, 4), sharey=False)

    for r, ax in zip(rs, axes[:3]):
        for i in range(n):
            ax.plot(
                [
                    x[str(aidx[i])]["inventory"][r] + x[str(aidx[i])]["escrow"][r]
                    for x in log["states"]
                ],
                label=rank_labels[i],
                color=cmap(i),
            )
        ax.set_title(r)
        ax.grid(True)

    ax = axes[3]
    for i in range(n):
        ax.plot(
            [x[str(aidx[i])]["endogenous"]["Labor"] for x in log["states"]],
            label=rank_labels[i],
            color=cmap(i),
        )
    ax.set_title("Labor")
    ax.grid(True)

    ax = axes[4]
    utility_ok = False
    try:
        for i in range(n):
            vals = [x[str(aidx[i])].get("utility", np.nan) for x in log["states"]]
            if np.any(np.isfinite(vals)):
                utility_ok = True
                ax.plot(vals, label=rank_labels[i], color=cmap(i))
        if utility_ok:
            ax.set_title("Utility")
        else:
            for i in range(n):
                vals = [
                    x[str(aidx[i])]["inventory"]["Coin"] + x[str(aidx[i])]["escrow"]["Coin"]
                    for x in log["states"]
                ]
                ax.plot(vals, label=rank_labels[i], color=cmap(i))
            ax.set_title("Coin (duplicate)")
    except Exception:
        for i in range(n):
            vals = [
                x[str(aidx[i])]["inventory"]["Coin"] + x[str(aidx[i])]["escrow"]["Coin"]
                for x in log["states"]
            ]
            ax.plot(vals, label=rank_labels[i], color=cmap(i))
        ax.set_title("Coin (duplicate)")
    ax.grid(True)

    # --- Separate planner reward figure ---
    fig1_planner, ax_planner = plt.subplots(1, 1, figsize=(8, 4))
    try:
        p_top = log.get("planner_rewards", {}).get("p_top", [])
        p_bot = log.get("planner_rewards", {}).get("p_bottom", [])

        if len(p_top) > 0:
            p_top = np.array(p_top, dtype=float)
            mask_top = np.isfinite(p_top)
            ax_planner.plot(np.where(mask_top)[0], p_top[mask_top], label="Planner Top", linestyle="--")
        if len(p_bot) > 0:
            p_bot = np.array(p_bot, dtype=float)
            mask_bot = np.isfinite(p_bot)
            ax_planner.plot(np.where(mask_bot)[0], p_bot[mask_bot], label="Planner Bottom", linestyle="--")

        ax_planner.set_title("Planner Rewards")
        ax_planner.set_xlabel("Timestep")
        ax_planner.set_ylabel("Reward")
        ax_planner.grid(True)
        ax_planner.legend()
    except Exception:
        ax_planner.set_title("Planner Rewards (missing)")
        ax_planner.grid(True)

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    # --- Movement tracks / trade timelines for ALL agents ---
    tmp_map = np.array(log["world"][0]["Stone"])
    map_h, map_w = tmp_map.shape
    middle_col = map_w // 2

    n_plot_cols = min(n_cols, n)
    n_agent_rows = int(math.ceil(n / n_plot_cols))
    total_rows = n_agent_rows + (n_agent_rows if trading_active else 0)

    # Make the movement rows taller than the trade rows
    if trading_active:
        height_ratios = [1.6] * n_agent_rows + [0.8] * n_agent_rows
        fig_height = 4.5 * n_agent_rows + 2.5 * n_agent_rows
    else:
        height_ratios = [1.6] * n_agent_rows
        fig_height = 4.5 * n_agent_rows

    fig2, axes = plt.subplots(
        total_rows,
        n_plot_cols,
        figsize=(4.0 * n_plot_cols, fig_height),
        sharex=False,
        sharey=False,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig2.subplots_adjust(hspace=0.5)

    # Turn off unused axes
    for rr in range(total_rows):
        for cc in range(n_plot_cols):
            idx = rr * n_plot_cols + cc
            if rr < n_agent_rows:
                if idx >= n:
                    axes[rr, cc].axis("off")
            else:
                trade_idx = (rr - n_agent_rows) * n_plot_cols + cc
                if trade_idx >= n:
                    axes[rr, cc].axis("off")

    # --- Trajectories: simple and clean ---
    for i in range(n):
        rr = i // n_plot_cols
        cc = i % n_plot_cols
        ax = axes[rr, cc]

        aid = aidx[i]
        locs = np.array([x[str(aid)]["loc"] for x in log["states"]], dtype=int)

        rows = locs[:, 0]
        cols = locs[:, 1]

        # Plot each step separately so there are NO shadow lines
        for t in range(len(locs) - 1):
            r0, c0 = locs[t]
            r1, c1 = locs[t + 1]

            # skip no-op steps
            if r0 == r1 and c0 == c1:
                continue

            # treat any crossing of middle as travel
            is_travel = (c0 < middle_col and c1 > middle_col) or (c0 > middle_col and c1 < middle_col)

            ax.plot(
                [c0, c1],
                [-r0, -r1],
                color=cmap(i),
                linewidth=1.2,
                alpha=0.25 if is_travel else 0.9,
            )

        # start/end markers
        ax.plot(cols[0], -rows[0], "r*", markersize=12)
        ax.plot(cols[-1], -rows[-1], "g*", markersize=12)

        # middle divider
        ax.axvline(middle_col, color="gray", linestyle=":", linewidth=0.8, alpha=0.8)

        ax.set_title(rank_labels[i], fontsize=10)
        ax.set_xlim([-1, 1 + map_w])
        ax.set_ylim([-(1 + map_h), 1])
        ax.grid(True)

    movement_legend = [
        Line2D([0], [0], color="black", lw=1.2, alpha=0.9, label="Movement"),
        Line2D([0], [0], color="black", lw=1.2, alpha=0.25, label="Travel"),
    ]
    axes[0, 0].legend(handles=movement_legend, loc="upper right", fontsize=8)

    # --- Trade timelines ---
    if trading_active:
        for i in range(n):
            rr = n_agent_rows + (i // n_plot_cols)
            cc = i % n_plot_cols
            ax = axes[rr, cc]

            for r in ["Wood", "Stone"]:
                tmp = [(s["t"], s["income"]) for s in c_trades[r] if s["seller"] == aidx[i]]
                if tmp:
                    ts, prices = [np.array(x) for x in zip(*tmp)]
                    ax.plot(
                        np.stack([ts, ts]),
                        np.stack([np.zeros_like(prices), prices]),
                        color=resources.get(r).color,
                    )
                    ax.plot(ts, prices, ".", color=resources.get(r).color, markersize=10)

                tmp = [(s["t"], -s["cost"]) for s in c_trades[r] if s["buyer"] == aidx[i]]
                if tmp:
                    ts, prices = [np.array(x) for x in zip(*tmp)]
                    ax.plot(
                        np.stack([ts, ts]),
                        np.stack([np.zeros_like(prices), prices]),
                        color=resources.get(r).color,
                    )
                    ax.plot(ts, prices, ".", color=resources.get(r).color, markersize=10)

            ax.plot([-20, len(log["states"]) + 19], [0, 0], "w-")
            ax.set_xlim([-20, len(log["states"]) + 19])
            ax.grid(True)
            ax.set_facecolor([0.3, 0.3, 0.3])
            ax.set_title(rank_labels[i], fontsize=10)

    fig2.tight_layout(pad=2.0)

    return (fig0, fig1, fig1_planner, fig2), incomes, endows, c_trades, all_builds

#----------------------------------------
# Tax Plot
#----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from simulation import get_disc_rates

def plot_avg_final_tax_schedules_two_planners_from_dense_logs(
    dense_logs,
    env_obj,
    brackets,
    top_first=True,
    title="Average Final Marginal Tax Schedules (p_top vs p_bottom)",
    figsize=(9, 7),
    errorbar="std",        # "std", "sem", or None
    last_bin_width=None,   # controls width of final bracket
    capsize=3,
):
    """
    Average final schedules across all dense_logs.
    Uses the last logged planner-action row from each episode.

    - Error bars are centered in each bracket
    - Final bracket is extended so it is fully visible
    """

    def _split_last_row(arr, top_first=True):
        if arr.size == 0:
            return None, None
        last = arr[-1].reshape(1, -1)
        half = last.shape[1] // 2
        if top_first:
            return last[:, :half][0], last[:, half:][0]
        else:
            return last[:, half:][0], last[:, :half][0]

    idx_top_list = []
    idx_bottom_list = []

    for ep in dense_logs:
        actions_top = dense_logs[ep]["planner_actions"]["p_top"]
        actions_bot = dense_logs[ep]["planner_actions"]["p_bottom"]

        if actions_top.size == 0 or actions_bot.size == 0:
            continue

        top_last, _ = _split_last_row(actions_top, top_first=top_first)
        _, bottom_last = _split_last_row(actions_bot, top_first=top_first)

        idx_top_list.append(top_last)
        idx_bottom_list.append(bottom_last)

    if len(idx_top_list) == 0 or len(idx_bottom_list) == 0:
        raise ValueError("No planner actions logged in dense_logs.")

    idx_top_arr = np.stack(idx_top_list, axis=0)
    idx_bottom_arr = np.stack(idx_bottom_list, axis=0)

    disc_rates = get_disc_rates(env_obj)

    top_rates_all = disc_rates[np.clip(idx_top_arr, 0, len(disc_rates) - 1)]
    bottom_rates_all = disc_rates[np.clip(idx_bottom_arr, 0, len(disc_rates) - 1)]

    top_rates_mean = np.mean(top_rates_all, axis=0)
    bottom_rates_mean = np.mean(bottom_rates_all, axis=0)

    # --- variability ---
    if errorbar == "std":
        top_err = np.std(top_rates_all, axis=0)
        bottom_err = np.std(bottom_rates_all, axis=0)
    elif errorbar == "sem":
        top_err = np.std(top_rates_all, axis=0) / np.sqrt(top_rates_all.shape[0])
        bottom_err = np.std(bottom_rates_all, axis=0) / np.sqrt(bottom_rates_all.shape[0])
    else:
        top_err = None
        bottom_err = None

    # --- extend last bracket ---
    brackets = np.asarray(brackets, dtype=float)

    if last_bin_width is None:
        last_bin_width = brackets[-1] - brackets[-2]

    right_edge = brackets[-1] + last_bin_width

    step_x = np.append(brackets, right_edge)

    top_step_y = np.append(top_rates_mean, top_rates_mean[-1])
    bottom_step_y = np.append(bottom_rates_mean, bottom_rates_mean[-1])

    # --- bracket centers for error bars ---
    centers = np.empty(len(brackets))
    centers[:-1] = 0.5 * (brackets[:-1] + brackets[1:])
    centers[-1] = 0.5 * (brackets[-1] + right_edge)

    # --- colors ---
    color_top = "#1f77b4"
    color_bottom = "#ff7f0e"

    # --- plot ---
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, y=0.98)

    # ---- TOP ----
    axes[0].step(step_x, top_step_y, where="post", color=color_top)
    axes[0].fill_between(step_x, top_step_y, step="post", alpha=0.25, color=color_top)

    if top_err is not None:
        axes[0].errorbar(
            centers,
            top_rates_mean,
            yerr=top_err,
            fmt="none",
            ecolor=color_top,
            elinewidth=1.4,
            capsize=capsize,
        )

    axes[0].set_ylabel("Marginal rate")
    axes[0].set_title("p_top (Top Region)")
    axes[0].grid(True)

    # ---- BOTTOM ----
    axes[1].step(step_x, bottom_step_y, where="post", color=color_bottom)
    axes[1].fill_between(step_x, bottom_step_y, step="post", alpha=0.25, color=color_bottom)

    if bottom_err is not None:
        axes[1].errorbar(
            centers,
            bottom_rates_mean,
            yerr=bottom_err,
            fmt="none",
            ecolor=color_bottom,
            elinewidth=1.4,
            capsize=capsize,
        )

    axes[1].set_ylabel("Marginal rate")
    axes[1].set_title("p_bottom (Bottom Region)")
    axes[1].set_xlabel("Income (k USD)")
    axes[1].grid(True)

    # --- shared formatting ---
    ymax = 1.05 * np.max(disc_rates)
    axes[0].set_ylim(0, ymax)
    axes[1].set_ylim(0, ymax)

    axes[1].set_xlim(brackets[0], right_edge)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    return fig

def plot_planner_tax_lines(
    dense_logs,
    env_obj,
    brackets,
    top_first=True,
    mode="average",      # "single" or "average"
    errorbar="std",      # "std", "sem", or None (only used in average mode)
    figsize=(14, 10),
    title=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def _split(A):
        if A is None or A.size == 0:
            return A, A
        half = A.shape[1] // 2
        if top_first:
            return A[:, :half], A[:, half:]
        else:
            return A[:, half:], A[:, :half]

    def _extract_episode_logs(obj):
        if obj is None:
            return []

        # single episode
        if isinstance(obj, dict) and "planner_actions" in obj:
            return [obj]

        # dict of episodes
        if isinstance(obj, dict):
            vals = list(obj.values())
            return [v for v in vals if isinstance(v, dict) and "planner_actions" in v]

        # list of episodes
        if isinstance(obj, list):
            return [v for v in obj if isinstance(v, dict) and "planner_actions" in v]

        return []

    # def _get_disc_rates(env_obj):
    #     for comp in env_obj.env.components:
    #         if "BracketTax" in comp.name and hasattr(comp, "disc_rates"):
    #             return np.array(comp.disc_rates, dtype=float)
    #     return np.arange(0.0, 1.0 + 1e-9, 0.05)

    eps = _extract_episode_logs(dense_logs)
    if len(eps) == 0:
        raise ValueError("No episode logs with planner_actions found.")

    top_list = []
    bot_list = []

    for ep in eps:
        A_top = ep["planner_actions"]["p_top"]
        A_bot = ep["planner_actions"]["p_bottom"]

        if A_top.size == 0 or A_bot.size == 0:
            continue

        top7_top, _ = _split(A_top)
        _, bottom7_bot = _split(A_bot)

        top_list.append(top7_top)
        bot_list.append(bottom7_bot)

    if len(top_list) == 0 or len(bot_list) == 0:
        raise ValueError("No planner actions logged in dense_logs.")

    # align episode lengths
    min_T = min(a.shape[0] for a in top_list)

    top_stack = np.stack([a[:min_T] for a in top_list], axis=0)   # (E, T, 7)
    bot_stack = np.stack([a[:min_T] for a in bot_list], axis=0)

    disc_rates = get_disc_rates(env_obj)

    top_rates = disc_rates[np.clip(top_stack.astype(int), 0, len(disc_rates) - 1)]
    bot_rates = disc_rates[np.clip(bot_stack.astype(int), 0, len(disc_rates) - 1)]

    X = np.arange(min_T)

    # consistent style with your other plots
    color_top = "#1f77b4"
    color_bottom = "#ff7f0e"

    if title is None:
        if mode == "single":
            title = "Planner Tax Choices Over Time (Single Dense Log)"
        else:
            title = "Planner Tax Choices Over Time (Mean Across Dense Logs)"

    fig, axes = plt.subplots(7, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, y=0.98)

    if mode == "single":
        top_plot = top_rates[0]   # (T, 7)
        bot_plot = bot_rates[0]

        for b in range(7):
            ax = axes[b]

            ax.plot(
                X,
                top_plot[:, b],
                color=color_top,
                linewidth=2.2,
                label="p_top" if b == 0 else None,
            )
            ax.plot(
                X,
                bot_plot[:, b],
                color=color_bottom,
                linewidth=2.2,
                linestyle="--",
                label="p_bottom" if b == 0 else None,
            )

            ax.set_ylabel(f"{brackets[b]:.1f}k")
            ax.set_ylim(0, 1.05 * np.max(disc_rates))
            ax.grid(True, alpha=0.3)

    elif mode == "average":
        top_mean = np.mean(top_rates, axis=0)   # (T, 7)
        bot_mean = np.mean(bot_rates, axis=0)

        if errorbar == "std":
            top_err = np.std(top_rates, axis=0)
            bot_err = np.std(bot_rates, axis=0)
        elif errorbar == "sem":
            top_err = np.std(top_rates, axis=0) / np.sqrt(top_rates.shape[0])
            bot_err = np.std(bot_rates, axis=0) / np.sqrt(bot_rates.shape[0])
        else:
            top_err = None
            bot_err = None

        for b in range(7):
            ax = axes[b]

            ax.plot(
                X,
                top_mean[:, b],
                color=color_top,
                linewidth=2.2,
                label="p_top" if b == 0 else None,
            )
            ax.plot(
                X,
                bot_mean[:, b],
                color=color_bottom,
                linewidth=2.2,
                linestyle="--",
                label="p_bottom" if b == 0 else None,
            )

            if top_err is not None:
                ax.fill_between(
                    X,
                    top_mean[:, b] - top_err[:, b],
                    top_mean[:, b] + top_err[:, b],
                    color=color_top,
                    alpha=0.18,
                )

            if bot_err is not None:
                ax.fill_between(
                    X,
                    bot_mean[:, b] - bot_err[:, b],
                    bot_mean[:, b] + bot_err[:, b],
                    color=color_bottom,
                    alpha=0.18,
                )

            ax.set_ylabel(f"{brackets[b]:.1f}k")
            ax.set_ylim(0, 1.05 * np.max(disc_rates))
            ax.grid(True, alpha=0.3)

    else:
        raise ValueError("mode must be 'single' or 'average'")

    axes[-1].set_xlabel("Decision index")

    # legend below, same style as your other plots
    legend_handles = [
        Line2D([0], [0], color=color_top, lw=2.2, label="p_top"),
        Line2D([0], [0], color=color_bottom, lw=2.2, linestyle="--", label="p_bottom"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        fontsize=10,
    )

    fig.subplots_adjust(bottom=0.08, top=0.93, hspace=0.25)
    return fig
#----------------------------------------
# Tables
#----------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def diagnose_resource_market(log, resource="Wood", period=100):

    n_steps = len(log["states"])

    # --- trades per timestep ---
    trade_count = np.zeros(n_steps, dtype=int)
    trade_price = np.full(n_steps, np.nan)

    if "Trade" in log:
        for t, trades in enumerate(log["Trade"]):
            trades_ = trades.get("trades", []) if isinstance(trades, dict) else trades
            resource_trades = [tr for tr in trades_ if tr.get("commodity") == resource]

            trade_count[t] = len(resource_trades)
            if len(resource_trades) > 0:
                trade_price[t] = np.mean([tr["price"] for tr in resource_trades])

    # --- inventories ---
    total_inventory = np.zeros(n_steps)
    total_escrow = np.zeros(n_steps)

    for t, state in enumerate(log["states"]):
        agents = [k for k in state.keys() if str(k).isdigit()]

        total_inventory[t] = sum(
            state[str(a)]["inventory"].get(resource, 0.0) for a in agents
        )
        total_escrow[t] = sum(
            state[str(a)]["escrow"].get(resource, 0.0) for a in agents
        )

    # --- resource on map ---
    resource_on_map = np.full(n_steps, np.nan)
    if "world" in log:
        for t in range(min(n_steps, len(log["world"]))):
            world_t = log["world"][t]
            if resource in world_t:
                resource_on_map[t] = np.sum(np.array(world_t[resource]))

    # fill missing timesteps with last known map value
    resource_on_map = pd.Series(resource_on_map).ffill().bfill().to_numpy()

    # --- builds ---
    builds_per_t = np.zeros(n_steps)
    build_income_per_t = np.zeros(n_steps)

    if "Build" in log:
        for t, builds in enumerate(log["Build"]):
            builds_ = builds.get("builds", []) if isinstance(builds, dict) else builds
            builds_per_t[t] = len(builds_)
            build_income_per_t[t] = sum(b.get("income", 0.0) for b in builds_)

    # --- dataframe ---
    df_timestep = pd.DataFrame({
        "t": np.arange(n_steps),
        f"{resource.lower()}_trades": trade_count,
        f"{resource.lower()}_inventory_total": total_inventory,
        f"{resource.lower()}_escrow_total": total_escrow,
        f"{resource.lower()}_on_map": resource_on_map,
        "builds": builds_per_t,
        "tax_period": np.arange(n_steps) // period,
    })

    df_period = df_timestep.groupby("tax_period").agg({
        f"{resource.lower()}_trades": "sum",
        f"{resource.lower()}_inventory_total": "mean",
        f"{resource.lower()}_escrow_total": "mean",
        f"{resource.lower()}_on_map": "mean",
        "builds": "sum",
    }).reset_index()

    # =========================
    # PLOTTING
    # =========================

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- 1. trades ---
    axes[0].plot(df_timestep["t"], df_timestep[f"{resource.lower()}_trades"])
    axes[0].set_title(f"{resource} trades per timestep")
    axes[0].grid(True)

    # --- 2. STOCKS (FIXED) ---
    ax = axes[1]

    # left axis
    ax.plot(df_timestep["t"], df_timestep[f"{resource.lower()}_inventory_total"], label="Inventory")
    ax.plot(df_timestep["t"], df_timestep[f"{resource.lower()}_escrow_total"], label="Escrow")
    ax.set_ylabel("Inventory / Escrow")
    ax.grid(True)

    # right axis (on_map)
    if not np.all(np.isnan(df_timestep[f"{resource.lower()}_on_map"])):
        ax2 = ax.twinx()
        ax2.plot(
            df_timestep["t"],
            df_timestep[f"{resource.lower()}_on_map"],
            color="green",
            linewidth=2,
            label="On map",
        )
        ax2.set_ylabel("On map")

        # combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax.legend()

    ax.set_title(f"{resource} stock over time")

    # --- 3. builds ---
    axes[2].plot(df_timestep["t"], df_timestep["builds"])
    axes[2].set_title("Builds per timestep")
    axes[2].grid(True)

    # --- 4. period summary ---
    # axes[3].plot(df_period["tax_period"], df_period[f"{resource.lower()}_trades"], label="Trades / period")
    # axes[3].plot(df_period["tax_period"], df_period["builds"], label="Builds / period")
    # axes[3].set_title("Market vs building (per tax period)")
    # axes[3].legend()
    # axes[3].grid(True)
    # axes[3].set_xlabel("Tax period")

    fig.tight_layout()

    return df_timestep, df_period, fig



def tax_day_income_report(
    log,
    period=100,
    bracket_cutoffs=(-np.inf, 0, 9.7, 39.475, 84.2, 160.725, 204.1, 510.3, np.inf),
    #[0,\;10,\;25,\;50,\;80,\;120,\;180,\;\infty]
):
    """
    Reconstructs per-agent income at each tax day and assigns tax brackets.

    Income is computed as change in coin endowment over the period:
        income_t = coin_t - coin_{t-period}

    Returns:
        df_agent: one row per agent per tax day
        df_counts: number of agents in each bracket at each tax day
    """

    def total_coin(state, aid):
        return (
            state[str(aid)]["inventory"].get("Coin", 0.0)
            + state[str(aid)]["escrow"].get("Coin", 0.0)
        )

    def get_region(state, aid):
        # Prefer explicit region if present
        if "region" in state[str(aid)]:
            return state[str(aid)]["region"]
        # fallback from row location
        row = state[str(aid)]["loc"][0]
        waterline = 25  # for 51x25 world split at row 25
        return "top" if row < waterline else "bottom"

    def bracket_label(x, cutoffs):
        for i in range(len(cutoffs) - 1):
            lo, hi = cutoffs[i], cutoffs[i + 1]
            if lo <= x < hi:
                hi_label = "inf" if np.isinf(hi) else f"{hi:.3f}"
                return f"[{lo:.3f}, {hi_label})"
        return "unknown"

    states = log["states"]
    first_state = states[0]
    agent_ids = sorted([int(k) for k in first_state.keys() if str(k).isdigit()])

    # tax day assumed at end of each period: period-1, 2*period-1, ...
    tax_days = list(range(period - 1, len(states), period))

    rows = []
    prev_snapshot_idx = 0

    for td_idx, t in enumerate(tax_days):
        state_t = states[t]
        state_prev = states[prev_snapshot_idx]

        for aid in agent_ids:
            coin_now = total_coin(state_t, aid)
            coin_prev = total_coin(state_prev, aid)
            income = coin_now - coin_prev

            rows.append({
                "tax_day_number": td_idx + 1,
                "timestep": t,
                "agent": aid,
                "region": get_region(state_t, aid),
                "coin_start": coin_prev,
                "coin_end": coin_now,
                "income": income,
                "tax_bracket": bracket_label(income, bracket_cutoffs),
            })

        prev_snapshot_idx = t

    df_agent = pd.DataFrame(rows)

    df_counts = (
        df_agent.groupby(["tax_day_number", "region", "tax_bracket"])
        .size()
        .reset_index(name="n_agents")
        .sort_values(["tax_day_number", "region", "tax_bracket"])
    )

    return df_agent, df_counts

import numpy as np
import pandas as pd

def inventory_table_by_tax_period(log, period=100, include_escrow=True):
    """
    Build a table of agent inventories at the end of each tax period.

    Parameters
    ----------
    log : dict
        Dense log with log["states"]
    period : int
        Tax period length
    include_escrow : bool
        If True, also include escrow and total (= inventory + escrow)

    Returns
    -------
    df : pd.DataFrame
        One row per (tax_period, agent)
    """
    states = log["states"]
    n_steps = len(states)

    # tax-day indices: 99, 199, 299, ...
    tax_end_steps = list(range(period - 1, n_steps, period))

    rows = []

    for tax_day_number, t in enumerate(tax_end_steps, start=1):
        state = states[t]

        agent_ids = sorted(int(k) for k in state.keys() if str(k).isdigit())

        for aid in agent_ids:
            s = state[str(aid)]

            inv = s.get("inventory", {})
            esc = s.get("escrow", {})

            row = {
                "tax_day_number": tax_day_number,
                "timestep": t,
                "agent": aid,
                "region": s.get("region", None),
                "coin_inventory": inv.get("Coin", 0.0),
                "wood_inventory": inv.get("Wood", 0.0),
                "stone_inventory": inv.get("Stone", 0.0),
            }

            if include_escrow:
                row.update({
                    "coin_escrow": esc.get("Coin", 0.0),
                    "wood_escrow": esc.get("Wood", 0.0),
                    "stone_escrow": esc.get("Stone", 0.0),
                    "coin_total": inv.get("Coin", 0.0) + esc.get("Coin", 0.0),
                    "wood_total": inv.get("Wood", 0.0) + esc.get("Wood", 0.0),
                    "stone_total": inv.get("Stone", 0.0) + esc.get("Stone", 0.0),
                })

            rows.append(row)

    df = pd.DataFrame(rows)
    return df

def diagnose_agent_activity_df(dense_log):
    import numpy as np
    import pandas as pd

    states = dense_log["states"]
    n_steps = len(states) - 1

    agent_ids = sorted(
        [k for k in states[0].keys() if str(k).isdigit()],
        key=int
    )

    rows = []

    # --- per-agent stats ---
    for aid in agent_ids:

        # movement
        d = 0.0
        for t in range(1, len(states)):
            p0 = np.array(states[t-1][aid]["loc"], dtype=float)
            p1 = np.array(states[t][aid]["loc"], dtype=float)
            d += np.abs(p1 - p0).sum()

        s = states[-1][aid]

        row = {
            "agent": int(aid),
            "movement": d,
            "coin": float(s["inventory"]["Coin"] + s["escrow"]["Coin"]),
            "wood": float(s["inventory"]["Wood"] + s["escrow"]["Wood"]),
            "stone": float(s["inventory"]["Stone"] + s["escrow"]["Stone"]),
            "labor": float(s["endogenous"]["Labor"]),
        }

        rows.append(row)

    df_agents = pd.DataFrame(rows).sort_values("agent")

    # --- aggregate counts ---
    def count_events(events, key=None):
        total = 0
        for x in events:
            if isinstance(x, dict):
                if key is not None:
                    total += len(x.get(key, []))
                else:
                    total += sum(len(v) for v in x.values() if isinstance(v, list))
            elif isinstance(x, list):
                total += len(x)
        return total

    n_builds = count_events(dense_log.get("Build", []), key="builds")
    n_trades = count_events(dense_log.get("Trade", []), key="trades")

    # gather is messy → handle separately
    n_gathers = 0
    for x in dense_log.get("Gather", []):
        if isinstance(x, dict):
            if "gathers" in x:
                n_gathers += len(x["gathers"])
            elif "events" in x:
                n_gathers += len(x["events"])
        elif isinstance(x, list):
            n_gathers += len(x)

    # --- summary table ---
    df_summary = pd.DataFrame([{
        "steps": n_steps,
        "n_builds": n_builds,
        "n_trades": n_trades,
        "n_gathers": n_gathers,
        "avg_movement": df_agents["movement"].mean(),
        "total_coin": df_agents["coin"].sum(),
        "total_wood": df_agents["wood"].sum(),
        "total_stone": df_agents["stone"].sum(),
    }])

    return df_agents, df_summary

def agent_region_time_table(log, split_row=25, use_region_field=True, normalize=False):
    """
    Count how many timesteps each agent spends in each region.

    Parameters
    ----------
    log : dict
        Dense log containing log["states"].
    split_row : int
        Row that separates top and bottom region. For a 51x25 world, 25 is typical.
    use_region_field : bool
        If True, use state["region"] when available. Otherwise infer from loc.
    normalize : bool
        If True, also return shares of time instead of only counts.

    Returns
    -------
    df : pandas.DataFrame
        Index = agent id
        Columns include:
            - timesteps_top
            - timesteps_bottom
            - total_timesteps
            - share_top (optional)
            - share_bottom (optional)
            - final_region
    """
    states = log["states"]
    first_state = states[0]

    agent_ids = sorted(int(k) for k in first_state.keys() if str(k).isdigit())

    rows = []

    for aid in agent_ids:
        top_count = 0
        bottom_count = 0
        final_region = None

        for state in states:
            s = state[str(aid)]

            if use_region_field and "region" in s:
                region = s["region"]
            else:
                row = int(s["loc"][0])
                region = "top" if row < split_row else "bottom"

            if region == "top":
                top_count += 1
            elif region == "bottom":
                bottom_count += 1

            final_region = region

        total = top_count + bottom_count

        row_out = {
            "agent": aid,
            "timesteps_top": top_count,
            "timesteps_bottom": bottom_count,
            "total_timesteps": total,
            "final_region": final_region,
        }

        if normalize and total > 0:
            row_out["share_top"] = top_count / total
            row_out["share_bottom"] = bottom_count / total

        rows.append(row_out)

    df = pd.DataFrame(rows).set_index("agent")
    return df

def agent_region_spell_table(log, split_row=25, use_region_field=True):
    """
    For each agent, compute:
      - timesteps spent in top/bottom
      - number of region switches
      - average spell length in top
      - average spell length in bottom
      - overall average spell length

    A spell is a consecutive run of timesteps spent in the same region.
    """
    states = log["states"]
    agent_ids = sorted(int(k) for k in states[0].keys() if str(k).isdigit())

    def get_region(state, aid):
        s = state[str(aid)]
        if use_region_field and "region" in s:
            return s["region"]
        row = int(s["loc"][0])
        return "top" if row < split_row else "bottom"

    rows = []

    for aid in agent_ids:
        regions = [get_region(state, aid) for state in states]

        top_time = sum(r == "top" for r in regions)
        bottom_time = sum(r == "bottom" for r in regions)

        spells_top = []
        spells_bottom = []

        current_region = regions[0]
        current_len = 1
        switches = 0

        for r in regions[1:]:
            if r == current_region:
                current_len += 1
            else:
                if current_region == "top":
                    spells_top.append(current_len)
                else:
                    spells_bottom.append(current_len)
                switches += 1
                current_region = r
                current_len = 1

        # append final spell
        if current_region == "top":
            spells_top.append(current_len)
        else:
            spells_bottom.append(current_len)

        all_spells = spells_top + spells_bottom

        rows.append({
            "agent": aid,
            "timesteps_top": top_time,
            "timesteps_bottom": bottom_time,
            "n_region_switches": switches,
            "n_top_spells": len(spells_top),
            "n_bottom_spells": len(spells_bottom),
            "avg_top_spell": sum(spells_top) / len(spells_top) if spells_top else float("nan"),
            "avg_bottom_spell": sum(spells_bottom) / len(spells_bottom) if spells_bottom else float("nan"),
            "avg_spell_overall": sum(all_spells) / len(all_spells) if all_spells else float("nan"),
            "final_region": regions[-1],
        })

    return pd.DataFrame(rows).set_index("agent")


#----------------------------------------
# Comparison plots
#----------------------------------------


def load_experiment_run(run_dir):
    with open(os.path.join(run_dir, "summary.json"), "r") as f:
        summary = json.load(f)

        
    with open(os.path.join(run_dir, "dense_logs_final.pkl"), "rb") as f:
        dense_logs = pickle.load(f)

    metrics = pd.read_csv(os.path.join(run_dir, "training_metrics.csv"))

    with open(os.path.join(run_dir, "dense_logs_final.pkl"), "rb") as f:
        dense_log = pickle.load(f)


        
    #dense_log = dense_log[0] #maybe change
    if isinstance(dense_logs, dict) and 0 in dense_logs:
        dense_log = dense_logs[0]
    else:
        dense_log = dense_logs
    

    return {
        "run_dir": run_dir,
        "name": summary.get("experiment_name", os.path.basename(run_dir)),
        "summary": summary,
        "metrics": metrics,
        "dense_log": dense_log,
    }

def load_experiment_runs(run_dirs):
    return [load_experiment_run(rd) for rd in run_dirs]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compare_training_curves(
    runs,
    metric="episode_reward_mean",
    by_phase=False,
    show_phase_boundaries=True,
    short_labels=None,
    smooth_window=1,
    figsize=(10, 5),
):
    """
    Compare training curves across runs on a shared x-axis.

    Parameters
    ----------
    runs : list of dicts
        Each run must contain:
            run["name"]
            run["metrics"]   # DataFrame with columns ["phase", "iter", metric]

    metric : str
        Metric column to plot.

    by_phase : bool
        If True, plot each phase separately for each run.
        If False, concatenate phases into one cumulative training curve per run.

    show_phase_boundaries : bool
        Draw vertical lines between phases.

    short_labels : None, list, or dict
        Short legend labels. If None, uses "Run 1", "Run 2", ...

    smooth_window : int
        Rolling mean window for smoothing. Use 1 for no smoothing.

    figsize : tuple
        Figure size.
    """

    phase_order = ["PHASE 1", "PHASE 2", "PHASE 3A", "PHASE 3B"]

    # Build labels
    run_names = [run["name"] for run in runs]
    if short_labels is None:
        short_labels = {name: f"Run {i+1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    # Better contrast than default
    colors = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#8c564b",  # brown
    ]
    linestyles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]

    fig, ax = plt.subplots(figsize=figsize)

    all_boundaries = None
    max_x = 0

    for i, run in enumerate(runs):
        df = run["metrics"].copy()

        if metric not in df.columns:
            print(f"Skipping {run['name']}: metric '{metric}' not found.")
            continue

        df["phase"] = pd.Categorical(df["phase"], categories=phase_order, ordered=True)
        df = df.sort_values(["phase", "iter"]).reset_index(drop=True)

        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        label = short_labels[run["name"]]

        cumulative_offset = 0
        x_all = []
        y_all = []
        boundaries = []

        for phase in phase_order:
            sdf = df[df["phase"] == phase].copy()
            if sdf.empty:
                continue

            y_phase = sdf[metric].astype(float).to_numpy()

            if smooth_window > 1 and len(y_phase) >= smooth_window:
                y_phase = (
                    pd.Series(y_phase)
                    .rolling(window=smooth_window, min_periods=1, center=False)
                    .mean()
                    .to_numpy()
                )

            x_phase = np.arange(len(sdf)) + cumulative_offset

            if by_phase:
                ax.plot(
                    x_phase,
                    y_phase,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    alpha=0.95,
                    label=f"{label} | {phase}",
                )
            else:
                x_all.extend(x_phase.tolist())
                y_all.extend(y_phase.tolist())

            cumulative_offset = x_phase[-1] + 1
            boundaries.append(cumulative_offset)

        if not by_phase and len(x_all) > 0:
            ax.plot(
                x_all,
                y_all,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                alpha=0.95,
                label=label,
            )

        max_x = max(max_x, cumulative_offset)

        if all_boundaries is None:
            all_boundaries = boundaries

    # Draw phase boundaries once
    if show_phase_boundaries and all_boundaries is not None:
        for b in all_boundaries[:-1]:
            ax.axvline(b, linestyle="--", alpha=0.35, color="gray", linewidth=1)

    # Nicer labels
    pretty_title = metric.replace("_", " ").title()
    ax.set_title(pretty_title)
    ax.set_xlabel("Training iteration (cumulative across phases)")
    ax.set_ylabel(pretty_title)
    ax.grid(True, alpha=0.3)

    # Keep x-axis tight to data
    ax.set_xlim(0, max_x if max_x > 0 else 1)

    # Put legend below, not to the side
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(3, len(labels)),
            frameon=True,
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        fig.tight_layout()

    return fig

def compare_summary_bars(
    runs,
    metrics=None,
    short_labels=None,
    show_legend=True,
    errorbar="std",   # "std", "sem", or None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if metrics is None:
        metrics = [
            "mean_final_coin",
            "std_final_coin",
            "mean_final_labor",
            "n_trades",
            "n_builds",
        ]

    def summarize_one_dense_log(log):
        states = log["states"]
        last_state = states[-1]

        agent_ids = sorted([int(k) for k in last_state.keys() if str(k).isdigit()])

        final_coin = []
        final_labor = []

        for aid in agent_ids:
            s = last_state[str(aid)]
            coin = s["inventory"]["Coin"] + s["escrow"]["Coin"]
            labor = s["endogenous"]["Labor"]

            final_coin.append(float(coin))
            final_labor.append(float(labor))

        n_trades = 0
        if "Trade" in log:
            for t in log["Trade"]:
                trades = t.get("trades", []) if isinstance(t, dict) else t
                n_trades += len(trades)

        n_builds = 0
        if "Build" in log:
            for t in log["Build"]:
                builds = t.get("builds", []) if isinstance(t, dict) else t
                n_builds += len(builds)

        return {
            "mean_final_coin": float(np.mean(final_coin)) if final_coin else np.nan,
            "std_final_coin": float(np.std(final_coin)) if final_coin else np.nan,
            "mean_final_labor": float(np.mean(final_labor)) if final_labor else np.nan,
            "n_trades": float(n_trades),
            "n_builds": float(n_builds),
        }

    def extract_episode_logs(dense_logs_obj):
        if dense_logs_obj is None:
            return []

        if isinstance(dense_logs_obj, list):
            if len(dense_logs_obj) > 0 and isinstance(dense_logs_obj[0], dict):
                return dense_logs_obj
            return []

        if isinstance(dense_logs_obj, dict):
            if "states" in dense_logs_obj:
                return [dense_logs_obj]

            vals = list(dense_logs_obj.values())
            if len(vals) > 0 and isinstance(vals[0], dict):
                return vals

        return []

    # Main summary dataframe
    summary_df = pd.DataFrame(
        [{"name": r["name"], **r["summary"]} for r in runs]
    ).set_index("name")

    metrics = [m for m in metrics if m in summary_df.columns]
    run_names = list(summary_df.index)

    if short_labels is None:
        short_labels = {name: f"E{i+1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    colors_list = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#8c564b",  # brown
    ]
    colors = {name: colors_list[i % len(colors_list)] for i, name in enumerate(run_names)}

    # Compute episode-level metric variability for each run
    err_lookup = {name: {} for name in run_names}

    for run in runs:
        run_name = run["name"]

        dense_logs_obj = None
        for key in ["dense_logs", "dense_log", "logs"]:
            if key in run:
                dense_logs_obj = run[key]
                break

        eps = extract_episode_logs(dense_logs_obj)
        if len(eps) == 0:
            continue

        ep_rows = []
        for ep in eps:
            try:
                ep_rows.append(summarize_one_dense_log(ep))
            except Exception:
                continue

        if len(ep_rows) == 0:
            continue

        ep_df = pd.DataFrame(ep_rows)

        for metric in metrics:
            if metric not in ep_df.columns:
                err_lookup[run_name][metric] = np.nan
                continue

            vals = ep_df[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                err_lookup[run_name][metric] = np.nan
            elif errorbar == "std":
                err_lookup[run_name][metric] = float(np.std(vals))
            elif errorbar == "sem":
                err_lookup[run_name][metric] = float(np.std(vals) / np.sqrt(len(vals)))
            else:
                err_lookup[run_name][metric] = np.nan

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3 * len(metrics)), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[i, 0]

        vals = summary_df[metric]
        x = np.arange(len(vals))
        yerr = np.array([err_lookup[name].get(metric, np.nan) for name in vals.index], dtype=float)

        ax.bar(
            x,
            vals.values,
            color=[colors[name] for name in vals.index],
            width=0.6,
            alpha=0.9,
            yerr=None if errorbar is None else yerr,
            capsize=5 if errorbar is not None else 0,
            ecolor="black",
        )

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels([short_labels[name] for name in vals.index])
        ax.grid(True, axis="y", alpha=0.3)

    if show_legend:
        legend_handles = [
            Patch(facecolor=colors[name], label=f"{short_labels[name]}")
            for name in run_names
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(3, len(run_names)),
            frameon=True,
            fontsize=10,
        )

        fig.subplots_adjust(bottom=0.15, top=0.95)
    else:
        fig.tight_layout()

    out_df = summary_df.copy()
    out_df.insert(0, "label", [short_labels[name] for name in out_df.index])

    return fig, out_df

def extract_trade_count_over_time(log):
    if "Trade" not in log:
        return []

    counts = []
    for t in log["Trade"]:
        trades = t.get("trades", []) if isinstance(t, dict) else t
        counts.append(len(trades))
    return counts

def compare_trade_dynamics(
    runs,
    short_labels=None,
    mode="single",        # "single" or "average"
    smooth_window=1,      # >1 smooths curves
    show_legend=True,
    figsize=(10, 5),
):
    import numpy as np
    import matplotlib.pyplot as plt

    def extract_trade_count_over_time(log):
        if "Trade" not in log:
            return []
        counts = []
        for t in log["Trade"]:
            trades = t.get("trades", []) if isinstance(t, dict) else t
            counts.append(len(trades))
        return np.array(counts, dtype=float)

    def extract_episode_logs(obj):
        if obj is None:
            return []
        if isinstance(obj, dict) and "Trade" in obj:
            return [obj]
        if isinstance(obj, dict):
            return [v for v in obj.values() if isinstance(v, dict) and "Trade" in v]
        if isinstance(obj, list):
            return [v for v in obj if isinstance(v, dict) and "Trade" in v]
        return []

    def smooth(x, w):
        if w <= 1:
            return x
        return np.convolve(x, np.ones(w)/w, mode="same")

    run_names = [run["name"] for run in runs]

    if short_labels is None:
        short_labels = {name: f"E{i+1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    # consistent color palette (same as other plots)
    colors_list = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
        "#ff7f0e", "#8c564b", "#e377c2", "#17becf"
    ]
    colors = {name: colors_list[i % len(colors_list)] for i, name in enumerate(run_names)}

    fig, ax = plt.subplots(figsize=figsize)

    for run in runs:
        name = run["name"]
        color = colors[name]

        if mode == "single":
            log = run.get("dense_log", None)
            if log is None and "dense_logs" in run:
                eps = extract_episode_logs(run["dense_logs"])
                log = eps[0] if len(eps) > 0 else None

            if log is None:
                continue

            counts = extract_trade_count_over_time(log)
            counts = smooth(counts, smooth_window)

            ax.plot(
                counts,
                color=color,
                linewidth=2,
                alpha=0.9,
                label=short_labels[name],
            )

        elif mode == "average":
            dense_logs_obj = run.get("dense_logs", None)
            if dense_logs_obj is None and "dense_log" in run:
                dense_logs_obj = run["dense_log"]

            eps = extract_episode_logs(dense_logs_obj)
            series = []

            for ep in eps:
                counts = extract_trade_count_over_time(ep)
                if len(counts) > 0:
                    series.append(counts)

            if len(series) == 0:
                continue

            # align lengths
            min_len = min(len(s) for s in series)
            series = np.array([s[:min_len] for s in series])

            mean = np.mean(series, axis=0)
            std = np.std(series, axis=0)

            mean = smooth(mean, smooth_window)
            std = smooth(std, smooth_window)

            x = np.arange(len(mean))

            ax.plot(
                x,
                mean,
                color=color,
                linewidth=2.5,
                label=short_labels[name],
            )

            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.2,
            )

        else:
            raise ValueError("mode must be 'single' or 'average'")

    # titles
    if mode == "single":
        ax.set_title("Trade Count Over Time (Single Rollout)")
    else:
        ax.set_title("Trade Count Over Time (Mean ± SD Across Dense Logs)")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of trades")
    ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, len(runs)),
            frameon=True,
        )
        fig.subplots_adjust(bottom=0.25)
    else:
        fig.tight_layout()

    return fig


def compare_gini(
    runs,
    short_labels=None,
    show_legend=True,
    mode="single",      # "single" or "average"
    errorbar=None,      # None, "std", or "sem"
    figsize=(7, 4),
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def gini(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.nan
        if np.sum(x) <= 0:
            return np.nan
        x = np.sort(x)
        n = len(x)
        return (2 * np.sum((np.arange(1, n + 1) * x))) / (n * np.sum(x)) - (n + 1) / n

    def extract_episode_logs(obj):
        if obj is None:
            return []

        # single episode
        if isinstance(obj, dict) and "states" in obj:
            return [obj]

        # dict of episodes
        if isinstance(obj, dict):
            vals = list(obj.values())
            return [v for v in vals if isinstance(v, dict) and "states" in v]

        # list of episodes
        if isinstance(obj, list):
            return [v for v in obj if isinstance(v, dict) and "states" in v]

        return []

    def gini_from_one_log(log):
        final_state = log["states"][-1]
        coins = []

        for k, s in final_state.items():
            if str(k).isdigit():
                coins.append(
                    float(s["inventory"]["Coin"]) + float(s["escrow"]["Coin"])
                )

        return gini(np.array(coins, dtype=float))

    run_names = [run["name"] for run in runs]

    if short_labels is None:
        short_labels = {name: f"E{i+1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    # Same palette as other plots
    colors_list = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#17becf",  # cyan
    ]
    colors = {name: colors_list[i % len(colors_list)] for i, name in enumerate(run_names)}

    values = {}
    errors = {}
    n_used = {}

    for run in runs:
        name = run["name"]

        if mode == "single":
            log = run.get("dense_log", None)
            if log is None and "dense_logs" in run:
                eps = extract_episode_logs(run["dense_logs"])
                log = eps[0] if len(eps) > 0 else None

            val = gini_from_one_log(log) if log is not None else np.nan
            values[name] = val
            errors[name] = np.nan
            n_used[name] = 1 if np.isfinite(val) else 0

        elif mode == "average":
            dense_logs_obj = run.get("dense_logs", None)
            if dense_logs_obj is None and "dense_log" in run:
                dense_logs_obj = run["dense_log"]

            eps = extract_episode_logs(dense_logs_obj)
            gini_vals = []

            for ep in eps:
                try:
                    gini_vals.append(gini_from_one_log(ep))
                except Exception:
                    continue

            gini_vals = np.array(gini_vals, dtype=float)
            gini_vals = gini_vals[np.isfinite(gini_vals)]

            if len(gini_vals) == 0:
                values[name] = np.nan
                errors[name] = np.nan
                n_used[name] = 0
            else:
                values[name] = float(np.mean(gini_vals))
                n_used[name] = len(gini_vals)

                if errorbar == "std":
                    errors[name] = float(np.std(gini_vals))
                elif errorbar == "sem":
                    errors[name] = float(np.std(gini_vals) / np.sqrt(len(gini_vals)))
                else:
                    errors[name] = np.nan
        else:
            raise ValueError("mode must be 'single' or 'average'")

    df = pd.Series(values, name="gini")
    err = pd.Series(errors, name="error")

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    yerr = None if errorbar is None else err.values

    ax.bar(
        x,
        df.values,
        color=[colors[name] for name in df.index],
        width=0.6,
        alpha=0.9,
        yerr=yerr,
        capsize=5 if errorbar is not None else 0,
        ecolor="black",
        error_kw={"elinewidth": 2.0, "capthick": 2.0},
    )

    if mode == "single":
        ax.set_title("Gini Coefficient (Final Wealth, Single Rollout)")
    else:
        title = "Gini Coefficient (Final Wealth, Mean Across Dense Logs)"
        if errorbar == "std":
            title += " ± SD"
        elif errorbar == "sem":
            title += " ± SEM"
        ax.set_title(title)

    ax.set_xticks(x)
    ax.set_xticklabels([short_labels[name] for name in df.index])
    ax.set_ylabel("Gini")
    ax.grid(True, axis="y", alpha=0.3)

    if show_legend:
        legend_handles = [
            Patch(facecolor=colors[name], label=f"{short_labels[name]}")
            for name in df.index
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(3, len(legend_handles)),
            frameon=True,
            fontsize=10,
        )
        fig.subplots_adjust(bottom=0.18, top=0.88)
    else:
        fig.tight_layout()

    out_df = pd.DataFrame({
        "label": [short_labels[name] for name in df.index],
        "gini": df.values,
        "error": err.values,
        "n_dense_logs": [n_used[name] for name in df.index],
    }, index=df.index)

    return fig, out_df




from simulation import get_disc_rates

def compare_avg_final_tax_schedules_two_planners(
    runs,
    env_obj,
    brackets,
    top_first=True,
    short_labels=None,
    show_legend=True,
    figsize=(10, 7),
    errorbar="std",        # "std", "sem", or None
    last_bin_width=None,   # width used to extend the final bracket
    capsize=5,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def _split_last_row(arr, top_first=True):
        if arr is None or np.size(arr) == 0:
            return None, None
        last = np.asarray(arr)[-1].reshape(1, -1)
        half = last.shape[1] // 2
        if top_first:
            return last[:, :half][0], last[:, half:][0]
        else:
            return last[:, half:][0], last[:, :half][0]

    def _extract_episode_logs(dense_logs_obj):
        if dense_logs_obj is None:
            return []

        if isinstance(dense_logs_obj, list):
            if len(dense_logs_obj) > 0 and isinstance(dense_logs_obj[0], dict):
                return dense_logs_obj
            return []

        if isinstance(dense_logs_obj, dict):
            if "planner_actions" in dense_logs_obj:
                return [dense_logs_obj]

            for key in ["episodes", "dense_logs", "logs", "data"]:
                if key in dense_logs_obj:
                    val = dense_logs_obj[key]
                    if isinstance(val, list):
                        return [x for x in val if isinstance(x, dict)]
                    if isinstance(val, dict):
                        return [v for v in val.values() if isinstance(v, dict)]

            vals = list(dense_logs_obj.values())
            if len(vals) > 0 and isinstance(vals[0], dict):
                return vals

        return []

    def _final_schedule_stats_from_dense_logs(dense_logs_obj):
        eps = _extract_episode_logs(dense_logs_obj)

        idx_top_list = []
        idx_bottom_list = []

        for ep in eps:
            if not isinstance(ep, dict):
                continue
            if "planner_actions" not in ep:
                continue
            if "p_top" not in ep["planner_actions"] or "p_bottom" not in ep["planner_actions"]:
                continue

            actions_top = ep["planner_actions"]["p_top"]
            actions_bot = ep["planner_actions"]["p_bottom"]

            if np.size(actions_top) == 0 or np.size(actions_bot) == 0:
                continue

            top_last, _ = _split_last_row(actions_top, top_first=top_first)
            _, bottom_last = _split_last_row(actions_bot, top_first=top_first)

            if top_last is None or bottom_last is None:
                continue

            idx_top_list.append(top_last)
            idx_bottom_list.append(bottom_last)

        if len(idx_top_list) == 0 or len(idx_bottom_list) == 0:
            return None

        idx_top_arr = np.stack(idx_top_list, axis=0)
        idx_bottom_arr = np.stack(idx_bottom_list, axis=0)

        disc_rates = get_disc_rates(env_obj)

        top_rates_all = disc_rates[np.clip(idx_top_arr.astype(int), 0, len(disc_rates) - 1)]
        bottom_rates_all = disc_rates[np.clip(idx_bottom_arr.astype(int), 0, len(disc_rates) - 1)]

        top_mean = np.mean(top_rates_all, axis=0)
        bottom_mean = np.mean(bottom_rates_all, axis=0)

        if errorbar == "std":
            top_err = np.std(top_rates_all, axis=0)
            bottom_err = np.std(bottom_rates_all, axis=0)
        elif errorbar == "sem":
            top_err = np.std(top_rates_all, axis=0) / np.sqrt(top_rates_all.shape[0])
            bottom_err = np.std(bottom_rates_all, axis=0) / np.sqrt(bottom_rates_all.shape[0])
        else:
            top_err = None
            bottom_err = None

        return {
            "top_mean": top_mean,
            "bottom_mean": bottom_mean,
            "top_err": top_err,
            "bottom_err": bottom_err,
            "n_episodes": len(idx_top_list),
        }

    run_names = [run["name"] for run in runs]

    if short_labels is None:
        short_labels = {name: f"E{i+1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    brackets = np.asarray(brackets, dtype=float)
    disc_rates = get_disc_rates(env_obj)

    if last_bin_width is None:
        last_bin_width = brackets[-1] - brackets[-2]

    right_edge = brackets[-1] + last_bin_width
    step_x = np.append(brackets, right_edge)

    centers = np.empty(len(brackets))
    centers[:-1] = 0.5 * (brackets[:-1] + brackets[1:])
    centers[-1] = 0.5 * (brackets[-1] + right_edge)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle("Average Final Marginal Tax Schedules Across Runs", fontsize=14, y=0.98)

    summary_rows = []

    # high-contrast palette + distinct linestyles
    colors_list = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#17becf",  # cyan
    ]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (4, 1, 1, 1))]

    plotted_names = []

    for i, run in enumerate(runs):
        run_name = run["name"]

        dense_logs_obj = None
        for key in ["dense_logs", "dense_log", "logs"]:
            if key in run:
                dense_logs_obj = run[key]
                break

        if dense_logs_obj is None:
            print(f"Skipping {run_name}: no dense_log(s) found.")
            continue

        stats = _final_schedule_stats_from_dense_logs(dense_logs_obj)
        if stats is None:
            print(f"Skipping {run_name}: could not find planner_actions/p_top/p_bottom in this run.")
            continue

        color = colors_list[i % len(colors_list)]
        linestyle = linestyles[i % len(linestyles)]
        label = short_labels[run_name]

        top_step_y = np.append(stats["top_mean"], stats["top_mean"][-1])
        bottom_step_y = np.append(stats["bottom_mean"], stats["bottom_mean"][-1])

        # small horizontal offset so error bars from multiple runs do not sit exactly on top of each other
        if len(runs) > 1:
            offset_scale = 0.015 * (right_edge - brackets[0])
            offset = (i - (len(runs) - 1) / 2.0) * offset_scale
        else:
            offset = 0.0
        centers_offset = centers + offset

        # Top region
        axes[0].step(
            step_x,
            top_step_y,
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            alpha=0.95,
            label=label,
            zorder=2,
        )
        axes[0].fill_between(
            step_x,
            top_step_y,
            step="post",
            alpha=0.10,
            color=color,
            zorder=1,
        )

        if stats["top_err"] is not None:
            axes[0].errorbar(
                centers_offset,
                stats["top_mean"],
                yerr=stats["top_err"],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2.2,
                capsize=capsize,
                markersize=4.5,
                markeredgewidth=0.0,
                alpha=1.0,
                zorder=5,
            )

        # Bottom region
        axes[1].step(
            step_x,
            bottom_step_y,
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            alpha=0.95,
            label=label,
            zorder=2,
        )
        axes[1].fill_between(
            step_x,
            bottom_step_y,
            step="post",
            alpha=0.10,
            color=color,
            zorder=1,
        )

        if stats["bottom_err"] is not None:
            axes[1].errorbar(
                centers_offset,
                stats["bottom_mean"],
                yerr=stats["bottom_err"],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2.2,
                capsize=capsize,
                markersize=4.5,
                markeredgewidth=0.0,
                alpha=1.0,
                zorder=5,
            )

        plotted_names.append(run_name)

        summary_rows.append({
            "run": run_name,
            "label": label,
            "n_episodes": stats["n_episodes"],
            "top_schedule": stats["top_mean"],
            "bottom_schedule": stats["bottom_mean"],
            "top_error": stats["top_err"],
            "bottom_error": stats["bottom_err"],
        })

    ymax = 1.05 * np.max(disc_rates)

    axes[0].set_ylabel("Marginal rate")
    axes[0].set_title("p_top (Top Region)")
    axes[0].set_ylim(0, ymax)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Marginal rate")
    axes[1].set_title("p_bottom (Bottom Region)")
    axes[1].set_xlabel("Income (k USD)")
    axes[1].set_ylim(0, ymax)
    axes[1].grid(True, alpha=0.3)

    axes[1].set_xlim(brackets[0], right_edge)

    if show_legend:
        legend_handles = []
        for i, name in enumerate(run_names):
            if name not in plotted_names:
                continue
            color = colors_list[i % len(colors_list)]
            linestyle = linestyles[i % len(linestyles)]
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color=color,
                    linestyle=linestyle,
                    lw=2.2,
                    label=f"{short_labels[name]}"
                )
            )

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(2, max(1, len(legend_handles))),
            frameon=True,
            fontsize=10,
        )

        fig.subplots_adjust(bottom=0.22, top=0.90)
    else:
        fig.subplots_adjust(bottom=0.10, top=0.90)

    out_df = pd.DataFrame(summary_rows)
    return fig, out_df