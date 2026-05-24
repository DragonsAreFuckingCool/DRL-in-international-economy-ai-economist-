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

try:
    from ai_economist.foundation import landmarks, resources
except ModuleNotFoundError:
    class _MissingRegistry:
        def has(self, key):
            return False

        def get(self, key):
            raise ModuleNotFoundError(
                "ai_economist optional dependencies are needed for map/resource plotting."
            )

    landmarks = _MissingRegistry()
    resources = _MissingRegistry()

def numeric_agent_ids_from_states(state_dict):
    """
    Return sorted list of numeric agent IDs (ints), ignoring planners like 'p', 'p_top', 'p_bottom'.
    Works for both legacy (single planner) and your 2-planner extension.
    """
    return sorted([int(k) for k in state_dict.keys() if str(k).isdigit()])

def _make_agent_colors(agent_ids):
    """Stable, unique color per actual agent id; no gray."""
    agent_ids = sorted([int(a) for a in agent_ids])

    base_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#17becf",  # cyan, 
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#d62728",  # red,
        "#bcbd22",  # olive
        "#aec7e8",  # light blue
        "#ffbb78",  # light orange
        "#98df8a",  # light green
        "#c5b0d5",  # light purple
        "#c49c94",  # light brown
        "#f7b6d2",  # light pink
        "#9edae5",  # light cyan
    ]

    return {
        aid: base_colors[i % len(base_colors)]
        for i, aid in enumerate(agent_ids)
    }



def plot_map(maps, locs, ax=None, cmap_order=None, show_water=True, agent_ids=None, agent_colors=None, house_alpha=0.35):
    import matplotlib.colors as mcolors

    """Universal map renderer with stable per-agent colors and owned houses tinted by builder."""
    def _map_keys(m):
        return list(m.keys()) if hasattr(m, "keys") else []

    def _map_get(m, key, default=None):
        try:
            return m.get(key)
        except Exception:
            return default

    keys = _map_keys(maps)
    if not keys:
        raise ValueError("plot_map: No map keys found to infer world size.")

    world_size = None
    for key in keys:
        arr = _map_get(maps, key)
        if isinstance(arr, dict):
            if "health" in arr:
                world_size = np.array(arr["health"]).shape
                break
        else:
            a = np.array(arr)
            if a.ndim == 2:
                world_size = a.shape
                break

    if world_size is None:
        raise ValueError("plot_map: Could not infer world size.")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(min(0.4 * world_size[1], 16), min(0.4 * world_size[0], 16)))
    else:
        ax.cla()

    n_agents = len(locs)
    if agent_ids is None:
        agent_ids = list(range(n_agents))
    agent_ids = [int(a) for a in agent_ids]

    if cmap_order is None:
        cmap_order = list(range(n_agents))

    if agent_colors is None:
        agent_colors = _make_agent_colors(agent_ids)

    tmp = np.zeros((3, world_size[0], world_size[1]), dtype=float)

    # Draw resources and non-house landmarks.
    for key in keys:
        if key is None or "source" in str(key).lower():
            continue

        if str(key) == "House":
            continue

        arr = _map_get(maps, key)
        if arr is None:
            continue

        if not show_water and str(key).lower() == "water":
            continue

        if resources.has(key):
            rdef = resources.get(key)
            if rdef.collectible:
                a = np.array(arr, dtype=float)
                tmp += rdef.color[:, None, None] * a[None]

        elif landmarks.has(key):
            ldef = landmarks.get(key)
            if isinstance(arr, dict):
                health = np.array(arr.get("health", np.zeros(world_size)), dtype=float)
                tmp += ldef.color[:, None, None] * health[None]
            else:
                a = np.array(arr)
                if a.ndim == 2:
                    tmp += ldef.color[:, None, None] * a[None].astype(float)

    # Draw houses in the same color as the builder, but lighter / less opaque.
    if "House" in keys:
        house = _map_get(maps, "House")

        if isinstance(house, dict):
            house_owner = np.array(house.get("owner", -np.ones(world_size)), dtype=int)
            house_health = np.array(house.get("health", np.zeros(world_size)), dtype=float)
        else:
            house_health = np.array(house, dtype=float)
            house_owner = -np.ones_like(house_health, dtype=int)

        for aid in agent_ids:
            mask = (house_owner == aid) & (house_health > 0)
            if not np.any(mask):
                continue

            base = np.array(mcolors.to_rgb(agent_colors[aid]), dtype=float)
            light = (1.0 - house_alpha) * np.ones(3) + house_alpha * base
            tmp[:, mask] = light[:, None]

        # Fallback for old logs without owner info.
        unowned = (house_owner < 0) & (house_health > 0)
        if np.any(unowned):
            tmp[:, unowned] = np.array([0.85, 0.85, 0.85])[:, None]

    tmp = 0.7 * tmp + 0.3
    tmp = np.transpose(np.minimum(tmp, 1.0), [1, 2, 0])

    im = ax.imshow(tmp, vmax=1.0, interpolation="nearest")
    ax.set_aspect("equal")

    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    pix_h = bbox.height * ax.figure.dpi

    for pos in cmap_order:
        aid = agent_ids[pos]
        r, c = locs[pos]
        col = agent_colors[aid]
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

def plot_log_state(dense_log, t, ax=None, remap_key=None, agent_colors=None):
    maps = dense_log["world"][t]
    states = dense_log["states"][t]

    agent_ids = numeric_agent_ids_from_states(states)
    locs = [states[str(i)]["loc"] for i in agent_ids]

    if remap_key is None:
        cmap_order = None
    else:
        key_val = np.array([dense_log["states"][0][str(i)][remap_key] for i in agent_ids])
        cmap_order = np.argsort(key_val).tolist()

    plot_map(maps, locs, ax, cmap_order, agent_ids=agent_ids, agent_colors=agent_colors)



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



def vis_world_array(dense_logs, ts, eps=None, axes=None, remap_key=None, agent_colors=None):
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
            axes = np.array(axes).reshape(len(eps), len(ts))

    for ti, t in enumerate(ts):
        for ei, ep in enumerate(eps):
            plot_log_state(dense_logs[ep], t, ax=axes[ei, ti], remap_key=remap_key, agent_colors=agent_colors)

    for ax, t in zip(axes[0], ts):
        ax.set_title("T = {}".format(t))
    for ax, ep in zip(axes[:, 0], eps):
        ax.set_ylabel("Episode {}".format(ep))

    return fig




def vis_world_range(dense_logs, t0=0, tN=None, N=5, eps=None, axes=None, remap_key=None, agent_colors=None):
    dense_logs, eps = _format_logs_and_eps(dense_logs, eps)

    viable_ts = np.array([i for i, w in enumerate(dense_logs[0]["world"]) if w])
    if tN is None:
        tN = viable_ts[-1]

    target_ts = np.linspace(t0, tN, N).astype(np.int32)

    ts = set()
    for tt in target_ts:
        closest = np.argmin(np.abs(tt - viable_ts))
        ts.add(viable_ts[closest])
    ts = sorted(list(ts))

    if axes is not None:
        axes = axes[: len(ts)]

    return vis_world_array(dense_logs, ts, axes=axes, eps=eps, remap_key=remap_key, agent_colors=agent_colors)

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


def header_str(n_agents, a_indices=None):
    if a_indices is None:
        a_indices = list(range(n_agents))
    s_head = ("_" * 15) + ":_"
    s_tail = "_|_".join([" Agent {:2d} ____".format(i) for i in a_indices])
    return s_head + s_tail


def report(c_trades, all_builds, n_agents, a_indices=None):
    if a_indices is None:
        a_indices = list(range(n_agents))
    print(header_str(n_agents, a_indices))
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
      - uses stable, unique colors per actual agent id
      - colors houses as lighter versions of the builder's agent color
      - marks travel as lower-opacity movement
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    agent_ids = numeric_agent_ids_from_states(log["states"][0])
    n = len(agent_ids)
    trading_active = "Trade" in log

    agent_colors = _make_agent_colors(agent_ids)

    fig0 = vis_world_range(
        log,
        remap_key=remap_key,
        agent_colors=agent_colors,
    )

    if remap_key is None:
        aidx = agent_ids[:]
    else:
        key_vals = np.array([log["states"][0][str(i)][remap_key] for i in agent_ids])
        order = np.argsort(key_vals).tolist()
        aidx = [agent_ids[j] for j in order]

    rank_labels = []
    build_payment = {}
    #gather_mults = {}

    for aid in aidx:
        s = log["states"][0][str(aid)]
        build_payment[aid] = s.get("build_payment", np.nan)
        #p = s.get("bonus_gather_prob", np.nan)
        #gather_mults[aid] = 1.0 + p if np.isfinite(p) else np.nan

    finite_skills = sorted({v for v in build_payment.values() if np.isfinite(v)})
    skill_rank = {v: i for i, v in enumerate(finite_skills)}

    skill_vals = np.array([build_payment.get(aid, np.nan) for aid in aidx], dtype=float)
    lowest_skill = np.nanmin(skill_vals)
    highest_skill = np.nanmax(skill_vals)

    for aid in aidx:
        base = f"Agent {aid}"
        build = build_payment.get(aid, np.nan)

        if np.isfinite(build) and np.isclose(build, lowest_skill):
            base += " (Lowest Skill)"
        elif np.isfinite(build) and np.isclose(build, highest_skill):
            base += " (Highest Skill)"

        rank = skill_rank.get(build, np.nan)
        rank_text = f"Skill level {rank + 1}/{len(finite_skills)}" if np.isfinite(rank) else "Skill level ?"
        skill_line = f"\n{rank_text} | Build: {build:.2f}"
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
    # cmap = plt.get_cmap("jet", max(1, len(finite_skills)))
    # agent_colors = {}
    # for aid in aidx:
    #     build = build_payment.get(aid, np.nan)
    #     rank = skill_rank.get(build, 0)
    #     agent_colors[aid] = cmap(rank)
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
                color=agent_colors[aidx[i]],
            )
        ax.set_title(r)
        ax.grid(True)

    ax = axes[3]
    for i in range(n):
        ax.plot(
            [x[str(aidx[i])]["endogenous"]["Labor"] for x in log["states"]],
            label=rank_labels[i],
            color=agent_colors[aidx[i]],
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
                ax.plot(vals, label=rank_labels[i], color=agent_colors[aidx[i]])
        if utility_ok:
            ax.set_title("Utility")
        else:
            for i in range(n):
                vals = [
                    x[str(aidx[i])]["inventory"]["Coin"] + x[str(aidx[i])]["escrow"]["Coin"]
                    for x in log["states"]
                ]
                ax.plot(vals, label=rank_labels[i], color=agent_colors[aidx[i]])
            ax.set_title("Coin (duplicate)")
    except Exception:
        for i in range(n):
            vals = [
                x[str(aidx[i])]["inventory"]["Coin"] + x[str(aidx[i])]["escrow"]["Coin"]
                for x in log["states"]
            ]
            ax.plot(vals, label=rank_labels[i], color=agent_colors[aidx[i]])
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
                color=agent_colors[aidx[i]],
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
try:
    from simulation import get_disc_rates
except ModuleNotFoundError:
    def get_disc_rates(env_obj=None):
        return np.arange(0.0, 1.0 + 1e-9, 0.05)

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
    #bracket_cutoffs=(-np.inf, 0, 13.2, 22.05, 31.1, 40.3, 51.75, 234.85, np.inf),
    bracket_cutoffs=(-np.inf, 0, 9.7, 39.475, 84.2, 160.725, 204.1, 510.3, np.inf),
    split_row=None,
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

    def infer_split_row():
        if split_row is not None:
            return int(split_row)

        for world_state in log.get("world", []):
            if not world_state:
                continue
            for arr in world_state.values():
                try:
                    shape = np.asarray(arr).shape
                except Exception:
                    continue
                if len(shape) >= 2 and shape[0] > 0:
                    return int(shape[0] // 2)

        rows = [
            int(s["loc"][0])
            for state in log.get("states", [])
            for k, s in state.items()
            if str(k).isdigit() and "loc" in s
        ]
        if rows:
            return int((max(rows) + 1) // 2)

        return 25

    waterline = infer_split_row()

    def get_region_from_loc(state, aid):
        row = int(state[str(aid)]["loc"][0])
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
                "region": get_region_from_loc(state_t, aid),
                "state_region": state_t[str(aid)].get("region", None),
                "location_region": state_t[str(aid)].get("location_region", None),
                "split_row": waterline,
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

    config_path = os.path.join(run_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        
    with open(os.path.join(run_dir, "dense_logs_final.pkl"), "rb") as f:
        dense_logs = pickle.load(f)

    metrics = pd.read_csv(os.path.join(run_dir, "training_metrics.csv"))

    #dense_log = dense_log[0] #maybe change
    if isinstance(dense_logs, dict) and 0 in dense_logs:
        dense_log = dense_logs[0]
    else:
        dense_log = dense_logs
    

    return {
        "run_dir": run_dir,
        "name": summary.get("experiment_name", os.path.basename(run_dir)),
        "summary": summary,
        "config": config,
        "metrics": metrics,
        "dense_logs": dense_logs,
        "dense_log": dense_log,
    }

def load_experiment_runs(run_dirs):
    return [load_experiment_run(rd) for rd in run_dirs]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _regional_tax_component_config(config, phase_name, planner_id):
    phase_key = phase_name.lower().replace(" ", "")
    env_config = config.get(f"env_config_dict_{phase_key}", {})

    for component in env_config.get("components", []):
        if not isinstance(component, (list, tuple)) or len(component) != 2:
            continue
        component_name, component_config = component
        if component_name != "RegionalPeriodicBracketTax":
            continue
        if str(component_config.get("planner_id")) == str(planner_id):
            return component_config

    return None


def _reconstruct_tax_annealing_cap(metrics, config):
    metrics = metrics.copy()
    episode_length = float(config.get("episode_length", 1000))
    fragment_length = float(config.get("rollout_fragment_length", 1))
    min_band = int(config.get("min_band", 4))
    rate_disc = 0.05

    for planner_id in ["p_top", "p_bottom"]:
        values = []
        for _, row in metrics.iterrows():
            phase_name = str(row.get("phase", ""))
            component = _regional_tax_component_config(config, phase_name, planner_id)

            if component is None:
                values.append(np.nan)
                continue

            if bool(component.get("disable_taxes", False)):
                values.append(0.0)
                continue

            fixed_rates = component.get("fixed_bracket_rates")
            if fixed_rates is not None:
                arr = np.asarray(fixed_rates, dtype=float)
                values.append(float(np.nanmean(arr)) if arr.size else np.nan)
                continue

            schedule = component.get("tax_annealing_schedule")
            if schedule is None:
                values.append(float(component.get("rate_max", 1.0)))
                continue

            warmup, slope = float(schedule[0]), float(schedule[1])
            completions = np.floor((float(row.get("iter", 0)) + 1.0) * fragment_length / episode_length)
            cap = np.maximum(0.0, np.minimum(1.0, slope * (completions - warmup)))
            cap *= float(component.get("rate_max", 1.0))

            # The component also keeps the first min_band discrete rates open.
            min_band_cap = max(0, min_band - 1) * rate_disc
            values.append(float(max(cap, min_band_cap)))

        metrics[f"tax_annealing_cap/{planner_id}"] = values

    return metrics


def compare_training_curves(
    runs,
    metric="episode_reward_mean",
    by_phase=False,
    show_phase_boundaries=True,
    short_labels=None,
    smooth_window=1,
    episode_range=None,
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

    episode_range : None or tuple
        Optional cumulative training-iteration window to show, e.g.
        ``(2500, 4000)`` for the final 1500 iterations.

    figsize : tuple
        Figure size.
    """

    phase_order = ["PHASE 1", "PHASE 2", "PHASE 3A", "PHASE 3B"]
    metric_key = str(metric).lower().replace(" ", "_")
    metric_series = [(metric, None)]
    metric_title = metric.replace("_", " ").title()
    if metric_key == "tax_annealing_cap":
        metric_series = [
            ("tax_annealing_cap/p_top", "top planner"),
            ("tax_annealing_cap/p_bottom", "bottom planner"),
        ]
        metric_title = "Tax Annealing Cap"
    elif metric_key == "mean_tax":
        metric_series = [
            ("mean_tax/p_top", "top planner"),
            ("mean_tax/p_bottom", "bottom planner"),
        ]
        metric_title = "Mean Tax"

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
    linestyles = ["-"]

    fig, ax = plt.subplots(figsize=figsize)

    all_boundaries = None
    max_x = 0
    if episode_range is not None:
        if len(episode_range) != 2:
            raise ValueError("episode_range must be None or a (start, end) tuple.")
        episode_start, episode_end = [float(v) for v in episode_range]
        if episode_end <= episode_start:
            raise ValueError("episode_range end must be greater than start.")
    else:
        episode_start = None
        episode_end = None

    for i, run in enumerate(runs):
        df = run["metrics"].copy()
        run_metric_series = list(metric_series)
        if metric_key == "tax_annealing_cap":
            df = _reconstruct_tax_annealing_cap(df, run.get("config", {}))

        missing = [column for column, _ in run_metric_series if column not in df.columns]
        if missing:
            if metric_key == "mean_tax":
                print(
                    f"{run['name']}: mean_tax was not logged in training_metrics.csv; "
                    "cannot plot the actual applied mean tax for this run."
                )
            else:
                print(f"Skipping {run['name']}: metric '{metric}' needs missing column(s): {missing}.")
            continue

        if metric_key == "mean_tax":
            mean_tax_cols = [column for column, _ in run_metric_series]
            zero_both = df[mean_tax_cols].fillna(0.0).eq(0.0).all(axis=1)
            df.loc[zero_both, mean_tax_cols] = np.nan

        df["phase"] = pd.Categorical(df["phase"], categories=phase_order, ordered=True)
        df = df.sort_values(["phase", "iter"]).reset_index(drop=True)

        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        run_label = short_labels[run["name"]]

        for series_idx, (metric_column, series_label) in enumerate(run_metric_series):
            cumulative_offset = 0
            x_all = []
            y_all = []
            boundaries = []
            series_color = (
                colors[(i * len(run_metric_series) + series_idx) % len(colors)]
                if len(run_metric_series) > 1
                else color
            )
            series_linestyle = linestyle
            label = (
                f"{run_label} | {series_label}"
                if series_label is not None and len(runs) > 1
                else (series_label or run_label)
            )

            for phase in phase_order:
                sdf = df[df["phase"] == phase].copy()
                if sdf.empty:
                    continue

                y_series = sdf[metric_column].astype(float)
                if metric_key == "mean_tax":
                    y_series = y_series.interpolate(limit_direction="both")

                if smooth_window > 1 and len(y_series) >= smooth_window:
                    y_phase = (
                        y_series
                        .rolling(window=smooth_window, min_periods=1, center=False)
                        .mean()
                        .to_numpy()
                    )
                else:
                    y_phase = y_series.to_numpy()

                x_phase = np.arange(len(sdf)) + cumulative_offset

                if by_phase:
                    ax.plot(
                        x_phase,
                        y_phase,
                        color=series_color,
                        linestyle=series_linestyle,
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
                y_all_arr = np.asarray(y_all, dtype=float)
                if not np.any(np.isfinite(y_all_arr)):
                    print(f"Skipping {run['name']} {label}: no finite values for '{metric_column}'.")
                    continue

                ax.plot(
                    x_all,
                    y_all_arr,
                    color=series_color,
                    linestyle=series_linestyle,
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
            if episode_range is not None and not (episode_start <= b <= episode_end):
                continue
            ax.axvline(b, linestyle="--", alpha=0.35, color="gray", linewidth=1)

    # Nicer labels
    pretty_title = metric_title
    ax.set_title(pretty_title)
    ax.set_xlabel("Training iteration (cumulative across phases)")
    ax.set_ylabel(pretty_title)
    ax.grid(True, alpha=0.3)

    # Keep x-axis tight to data
    if episode_range is not None:
        ax.set_xlim(episode_start, episode_end)
    else:
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
        ax.text(
            0.5,
            0.5,
            f"No finite values found for {metric!r}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.35",
        )
        fig.tight_layout()

    return fig


def compare_equality_production_over_time(
    runs,
    short_labels=None,
    show_std=True,
    smooth_window=1,
    max_timestep=None,
    figsize=(12, 7),
):
    """
    Compare equality and production over episode timesteps across runs.

    Equality and production are computed from each dense log's per-timestep
    agent coin holdings. Production is total nonnegative coin; equality is
    1 - Gini over the same coin vector.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def gini(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) <= 1:
            return 0.0
        values = np.maximum(values, 0.0)
        total = float(np.sum(values))
        if total <= 0:
            return 0.0
        values = np.sort(values)
        n = len(values)
        return float((2.0 * np.sum(np.arange(1, n + 1) * values)) / (n * total) - (n + 1.0) / n)

    def coin_total(agent_state):
        inventory = agent_state.get("inventory", {})
        escrow = agent_state.get("escrow", {})
        return float(inventory.get("Coin", 0.0)) + float(escrow.get("Coin", 0.0))

    def dense_logs_for_run(run):
        if isinstance(run, dict):
            logs = _extract_logs_from_run(run)
            if logs:
                return [(k, v) for k, v in logs.items() if isinstance(v, dict) and "states" in v]
            if isinstance(run.get("states"), list):
                return [(0, run)]
        return []

    run_names = [run.get("name", f"Run {i+1}") for i, run in enumerate(runs)]
    if short_labels is None:
        short_labels = {name: f"Run {i+1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#8c564b",
    ]

    rows = []
    for run_idx, run in enumerate(runs):
        run_name = run_names[run_idx]
        for rollout_id, log in dense_logs_for_run(run):
            states = log.get("states", [])
            if max_timestep is not None:
                states = states[: int(max_timestep) + 1]
            for timestep, state in enumerate(states):
                coins = [
                    coin_total(agent_state)
                    for agent_id, agent_state in state.items()
                    if str(agent_id).isdigit() and isinstance(agent_state, dict)
                ]
                if not coins:
                    continue
                coins = np.asarray(coins, dtype=float)
                production = float(np.sum(np.maximum(coins, 0.0)))
                equality = float(1.0 - gini(coins))
                rows.append({
                    "run": run_name,
                    "label": short_labels.get(run_name, run_name),
                    "rollout_id": rollout_id,
                    "timestep": timestep,
                    "production": production,
                    "equality": equality,
                })

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError("No dense-log state data found for equality/production comparison.")

    summary_df = (
        raw_df
        .groupby(["run", "label", "timestep"], as_index=False)
        .agg(
            production=("production", "mean"),
            production_std=("production", "std"),
            equality=("equality", "mean"),
            equality_std=("equality", "std"),
            n_dense_logs=("production", "count"),
        )
    )
    for col in ["production_std", "equality_std"]:
        summary_df[col] = summary_df[col].fillna(0.0)

    if smooth_window and smooth_window > 1:
        for metric in ["production", "production_std", "equality", "equality_std"]:
            summary_df[metric] = (
                summary_df
                .sort_values(["run", "timestep"])
                .groupby("run")[metric]
                .transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
            )

    fig, (ax_eq, ax_prod) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )

    for run_idx, run_name in enumerate(run_names):
        dfr = summary_df[summary_df["run"] == run_name].sort_values("timestep")
        if dfr.empty:
            continue
        color = colors[run_idx % len(colors)]
        label = short_labels.get(run_name, run_name)
        x = dfr["timestep"].to_numpy(dtype=float)

        eq = dfr["equality"].to_numpy(dtype=float)
        eq_std = dfr["equality_std"].to_numpy(dtype=float)
        ax_eq.plot(x, eq, color=color, linewidth=2.0, label=label)
        if show_std:
            ax_eq.fill_between(
                x,
                np.clip(eq - eq_std, 0.0, 1.0),
                np.clip(eq + eq_std, 0.0, 1.0),
                color=color,
                alpha=0.13,
                linewidth=0,
            )

        prod = dfr["production"].to_numpy(dtype=float)
        prod_std = dfr["production_std"].to_numpy(dtype=float)
        ax_prod.plot(x, prod, color=color, linewidth=2.0, label=label)
        if show_std:
            ax_prod.fill_between(
                x,
                prod - prod_std,
                prod + prod_std,
                color=color,
                alpha=0.13,
                linewidth=0,
            )

    ax_eq.set_title("Equality Over Episode Timesteps")
    ax_eq.set_ylabel("Equality (1 - Gini)")
    ax_eq.set_ylim(0, 1)
    ax_eq.grid(True, alpha=0.25)

    ax_prod.set_title("Production Over Episode Timesteps")
    ax_prod.set_ylabel("Production (total coin)")
    ax_prod.set_xlabel("Episode timestep")
    ax_prod.grid(True, alpha=0.25)

    max_x = float(summary_df["timestep"].max())
    ax_prod.set_xlim(0, max_x if max_x > 0 else 1)

    handles, labels = ax_eq.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(labels)),
            frameon=True,
            fontsize=10,
        )
        fig.set_constrained_layout_pads(h_pad=0.08, w_pad=0.08)

    fig.suptitle("Equality and Production Comparison Across Runs", fontsize=14, fontweight="bold")
    return fig, summary_df, raw_df


def compare_redistribution_over_time(
    runs,
    short_labels=None,
    period=100,
    rate_disc=0.05,
    show_std=True,
    smooth_window=1,
    figsize=(12, 7),
):
    """Compare coin redistributed across runs by tax period."""
    import json
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def dense_logs_for_run(run):
        if isinstance(run, dict):
            logs = _extract_logs_from_run(run)
            if logs:
                return [(k, v) for k, v in logs.items() if isinstance(v, dict) and "states" in v]
            if isinstance(run.get("states"), list):
                return [(0, run)]
        return []

    def infer_travel_cost_coin(run):
        config = run.get("config", {}) if isinstance(run, dict) else {}
        if not config and isinstance(run, dict) and run.get("run_dir"):
            config_path = os.path.join(run["run_dir"], "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)

        if not isinstance(config, dict):
            return 0.0
        for key in ["travel_cost_coin_phase3b", "travel_cost_coin_phase3a", "travel_cost_coin"]:
            if key in config:
                return float(config[key])
        phase_configs = [
            value for _, value in config.items()
            if isinstance(value, dict) and "components" in value
        ]
        for phase_config in reversed(phase_configs):
            for component in phase_config.get("components", []):
                if (
                    isinstance(component, (list, tuple))
                    and len(component) >= 2
                    and component[0] == "CrossWaterTravel"
                    and isinstance(component[1], dict)
                    and "travel_cost_coin" in component[1]
                ):
                    return float(component[1]["travel_cost_coin"])
        return 0.0

    run_names = [run.get("name", f"Run {i+1}") for i, run in enumerate(runs)]
    if short_labels is None:
        short_labels = {name: f"Run {i+1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    rows = []
    production_rows = []
    for run_idx, run in enumerate(runs):
        run_name = run_names[run_idx]
        travel_cost_coin = infer_travel_cost_coin(run)
        for rollout_id, log in dense_logs_for_run(run):
            redist = extract_planner_redistribution_by_period(
                log,
                period=period,
                rate_disc=rate_disc,
                travel_cost_coin=travel_cost_coin,
            )
            if redist.empty:
                continue
            redist = redist.copy()
            redist["run"] = run_name
            redist["label"] = short_labels.get(run_name, run_name)
            redist["rollout_id"] = rollout_id
            rows.append(redist)

            states = log.get("states", [])
            aids = _numeric_agent_ids(log) if states else []
            tax_days = list(range(period - 1, len(states), period))
            prev_t = 0
            for tax_period, tax_t in enumerate(tax_days, start=1):
                state_prev = states[prev_t]
                state_now = states[tax_t]
                production = float(np.sum([
                    max(0.0, _coin(state_now[str(aid)]) - _coin(state_prev[str(aid)]))
                    for aid in aids
                    if str(aid) in state_now and str(aid) in state_prev
                ]))
                production_rows.append({
                    "run": run_name,
                    "label": short_labels.get(run_name, run_name),
                    "rollout_id": rollout_id,
                    "tax_period": tax_period,
                    "production": production,
                })
                prev_t = tax_t

    raw_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if raw_df.empty:
        raise ValueError("No redistribution rows could be constructed from these runs.")
    production_df = pd.DataFrame(production_rows)
    if production_df.empty:
        raise ValueError("No production rows could be constructed from these runs.")

    planner_summary = (
        raw_df
        .groupby(["run", "label", "tax_period", "planner_region"], as_index=False)
        .agg(
            redistributed=("redistributed", "mean"),
            redistributed_std=("redistributed", "std"),
            income_tax_collected=("income_tax_collected", "mean"),
            tariff_revenue=("tariff_revenue", "mean"),
            travel_tax_revenue=("travel_tax_revenue", "mean"),
            n_dense_logs=("redistributed", "count"),
        )
    )
    planner_summary["redistributed_std"] = planner_summary["redistributed_std"].fillna(0.0)

    per_rollout_total = (
        raw_df
        .groupby(["run", "label", "rollout_id", "tax_period"], as_index=False)
        .agg(redistributed=("redistributed", "sum"))
    )
    per_rollout_total = per_rollout_total.merge(
        production_df,
        on=["run", "label", "rollout_id", "tax_period"],
        how="left",
    )
    per_rollout_total["redistribution_share"] = np.divide(
        per_rollout_total["redistributed"],
        per_rollout_total["production"],
        out=np.full(len(per_rollout_total), np.nan, dtype=float),
        where=per_rollout_total["production"].to_numpy(dtype=float) > 0,
    )
    total_summary = (
        per_rollout_total
        .groupby(["run", "label", "tax_period"], as_index=False)
        .agg(
            redistributed=("redistributed", "mean"),
            redistributed_std=("redistributed", "std"),
            production=("production", "mean"),
            production_std=("production", "std"),
            redistribution_share=("redistribution_share", "mean"),
            redistribution_share_std=("redistribution_share", "std"),
            n_dense_logs=("redistributed", "count"),
        )
    )
    total_summary[["redistributed_std", "production_std", "redistribution_share_std"]] = (
        total_summary[["redistributed_std", "production_std", "redistribution_share_std"]].fillna(0.0)
    )

    if smooth_window and smooth_window > 1:
        for df in [planner_summary, total_summary]:
            group_cols = ["run"] if "planner_region" not in df.columns else ["run", "planner_region"]
            smooth_cols = ["redistributed", "redistributed_std"]
            if "redistribution_share" in df.columns:
                smooth_cols += ["production", "production_std", "redistribution_share", "redistribution_share_std"]
            for col in smooth_cols:
                df[col] = (
                    df
                    .sort_values(group_cols + ["tax_period"])
                    .groupby(group_cols)[col]
                    .transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
                )

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#8c564b",
    ]

    fig, (ax_total, ax_share) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )

    for run_idx, run_name in enumerate(run_names):
        color = colors[run_idx % len(colors)]
        label = short_labels.get(run_name, run_name)

        dft = total_summary[total_summary["run"] == run_name].sort_values("tax_period")
        if not dft.empty:
            x = dft["tax_period"].to_numpy(dtype=float)
            y = dft["redistributed"].to_numpy(dtype=float)
            yerr = dft["redistributed_std"].to_numpy(dtype=float)
            ax_total.plot(x, y, color=color, linewidth=2.2, marker="o", label=label)
            if show_std:
                ax_total.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.13, linewidth=0)

            share = dft["redistribution_share"].to_numpy(dtype=float) * 100.0
            share_err = dft["redistribution_share_std"].to_numpy(dtype=float) * 100.0
            ax_share.plot(x, share, color=color, linewidth=2.2, marker="o", label=label)
            if show_std:
                ax_share.fill_between(
                    x,
                    share - share_err,
                    share + share_err,
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                )

    ax_total.set_title("Total Redistribution by Tax Period")
    ax_total.set_ylabel("Coin redistributed")
    ax_total.grid(True, alpha=0.25)

    ax_share.set_title("Redistribution as Share of Period Production")
    ax_share.set_ylabel("% of period production redistributed")
    ax_share.set_xlabel("Tax period")
    ax_share.grid(True, alpha=0.25)

    max_period = float(raw_df["tax_period"].max())
    ax_share.set_xlim(1, max_period if max_period > 1 else 1)

    handles, labels = ax_total.get_legend_handles_labels()
    if handles:
        ax_total.legend(handles, labels, loc="upper left", ncol=min(4, len(labels)), frameon=True)
    handles, labels = ax_share.get_legend_handles_labels()
    if handles:
        ax_share.legend(handles, labels, loc="upper left", ncol=min(4, len(labels)), frameon=True, fontsize=8)

    fig.suptitle("Redistribution Comparison Across Runs", fontsize=14, fontweight="bold")
    summary_df = {"total": total_summary, "by_planner": planner_summary, "production": production_df}
    return fig, summary_df, raw_df


def compare_builds_by_skill(
    runs,
    short_labels=None,
    show_std=True,
    figsize=(10, 5),
):
    """Compare total houses built by agent build-skill group across runs."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def dense_logs_for_run(run):
        if isinstance(run, dict):
            logs = _extract_logs_from_run(run)
            if logs:
                return [(k, v) for k, v in logs.items() if isinstance(v, dict) and "states" in v]
            if isinstance(run.get("states"), list):
                return [(0, run)]
        return []

    def build_events(log):
        for item in log.get("Build", []):
            builds = item.get("builds", []) if isinstance(item, dict) else item
            if not isinstance(builds, list):
                continue
            for build in builds:
                if isinstance(build, dict) and "builder" in build:
                    yield build

    run_names = [run.get("name", f"Run {i+1}") for i, run in enumerate(runs)]
    if short_labels is None:
        short_labels = {name: f"Run {i+1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    all_skill_values = []
    run_log_items = []
    for run_idx, run in enumerate(runs):
        log_items = dense_logs_for_run(run)
        run_log_items.append(log_items)
        for _, log in log_items:
            states = log.get("states", [])
            if not states:
                continue
            for aid in _numeric_agent_ids(log):
                skill = states[0].get(str(aid), {}).get("build_payment", np.nan)
                if np.isfinite(skill):
                    all_skill_values.append(float(skill))

    skill_values = sorted(set(all_skill_values))
    if not skill_values:
        raise ValueError("No finite build_payment skill values found in dense logs.")
    skill_rank = {skill: i + 1 for i, skill in enumerate(skill_values)}
    skill_labels = {
        skill: f"skill {skill_rank[skill]} ({skill:.0f})"
        for skill in skill_values
    }

    rows = []
    for run_idx, run in enumerate(runs):
        run_name = run_names[run_idx]
        for rollout_id, log in run_log_items[run_idx]:
            states = log.get("states", [])
            if not states:
                continue
            agent_skill = {}
            for aid in _numeric_agent_ids(log):
                skill = states[0].get(str(aid), {}).get("build_payment", np.nan)
                agent_skill[aid] = float(skill) if np.isfinite(skill) else np.nan

            counts = {skill: 0 for skill in skill_values}
            for build in build_events(log):
                aid = int(build.get("builder", -1))
                skill = agent_skill.get(aid, np.nan)
                if np.isfinite(skill):
                    counts[float(skill)] = counts.get(float(skill), 0) + 1

            for skill in skill_values:
                rows.append({
                    "run": run_name,
                    "label": short_labels.get(run_name, run_name),
                    "rollout_id": rollout_id,
                    "build_payment": skill,
                    "skill_group": skill_labels[skill],
                    "n_builds": float(counts.get(skill, 0)),
                })

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError("No build rows could be constructed from these runs.")

    summary_df = (
        raw_df
        .groupby(["run", "label", "build_payment", "skill_group"], as_index=False)
        .agg(
            n_builds=("n_builds", "mean"),
            n_builds_std=("n_builds", "std"),
            n_dense_logs=("n_builds", "count"),
        )
    )
    summary_df["n_builds_std"] = summary_df["n_builds_std"].fillna(0.0)

    x = np.arange(len(run_names), dtype=float)
    n_skills = len(skill_values)
    total_width = min(0.82, 0.18 * max(1, n_skills))
    bar_width = total_width / max(1, n_skills)
    offsets = (np.arange(n_skills) - (n_skills - 1) / 2.0) * bar_width
    colors = _make_agent_colors(range(n_skills))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for skill_idx, skill in enumerate(skill_values):
        vals = []
        errs = []
        for run_name in run_names:
            match = summary_df[
                (summary_df["run"] == run_name)
                & (summary_df["build_payment"] == skill)
            ]
            vals.append(float(match["n_builds"].iloc[0]) if len(match) else 0.0)
            errs.append(float(match["n_builds_std"].iloc[0]) if len(match) else 0.0)

        ax.bar(
            x + offsets[skill_idx],
            vals,
            width=bar_width * 0.92,
            color=colors[skill_idx % len(colors)],
            edgecolor="white",
            linewidth=0.8,
            label=skill_labels[skill],
            yerr=errs if show_std else None,
            error_kw=dict(ecolor="0.25", elinewidth=0.9, capsize=3, capthick=0.9),
        )

    ax.set_title("Total Houses Built by Skill Group")
    ax.set_ylabel("Houses built per dense log")
    ax.set_xticks(x)
    ax.set_xticklabels([short_labels.get(name, name) for name in run_names])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(n_skills, 4), frameon=True)

    return fig, summary_df, raw_df


def compare_skill_group_mechanisms(
    runs,
    short_labels=None,
    period=100,
    rate_disc=0.05,
    visible_radius=5,
    show_std=True,
    figsize=(15, 8),
):
    """
    Compare possible mechanisms by run and build-skill group.

    Metrics:
    builds, bought trade units, gathered units, visible resources, coin before
    tax day, and net tax paid after redistribution.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def dense_logs_for_run(run):
        if isinstance(run, dict):
            logs = _extract_logs_from_run(run)
            if logs:
                return [(k, v) for k, v in logs.items() if isinstance(v, dict) and "states" in v]
            if isinstance(run.get("states"), list):
                return [(0, run)]
        return []

    def events_from_timeline(log, key, inner_key=None):
        for t, item in enumerate(log.get(key, [])):
            events = item.get(inner_key, []) if inner_key and isinstance(item, dict) else item
            if not isinstance(events, list):
                continue
            for event in events:
                if isinstance(event, dict):
                    yield t, event

    def world_at(log, t):
        worlds = log.get("world", [])
        if not worlds:
            return {}
        t = min(max(0, int(t)), len(worlds) - 1)
        if worlds[t]:
            return worlds[t]
        for j in range(t, -1, -1):
            if worlds[j]:
                return worlds[j]
        for item in worlds:
            if item:
                return item
        return {}

    def map_array(world, key):
        value = world.get(key, None) if isinstance(world, dict) else None
        if value is None or isinstance(value, dict):
            return None
        return np.asarray(value)

    def visible_sum(log, t, loc, key):
        arr = map_array(world_at(log, t), key)
        if arr is None:
            return np.nan
        r, c = int(loc[0]), int(loc[1])
        r0 = max(0, r - visible_radius)
        r1 = min(arr.shape[0], r + visible_radius + 1)
        c0 = max(0, c - visible_radius)
        c1 = min(arr.shape[1], c + visible_radius + 1)
        return float(np.nansum(arr[r0:r1, c0:c1]))

    run_names = [run.get("name", f"Run {i+1}") for i, run in enumerate(runs)]
    if short_labels is None:
        short_labels = {name: f"Run {i+1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    all_skill_values = []
    run_log_items = []
    for run in runs:
        log_items = dense_logs_for_run(run)
        run_log_items.append(log_items)
        for _, log in log_items:
            states = log.get("states", [])
            if not states:
                continue
            for aid in _numeric_agent_ids(log):
                skill = states[0].get(str(aid), {}).get("build_payment", np.nan)
                if np.isfinite(skill):
                    all_skill_values.append(float(skill))

    skill_values = sorted(set(all_skill_values))
    if not skill_values:
        raise ValueError("No finite build_payment skill values found in dense logs.")
    skill_rank = {skill: i + 1 for i, skill in enumerate(skill_values)}
    skill_labels = {skill: f"skill {skill_rank[skill]} ({skill:.0f})" for skill in skill_values}

    rows = []
    for run_idx, run in enumerate(runs):
        run_name = run_names[run_idx]
        for rollout_id, log in run_log_items[run_idx]:
            states = log.get("states", [])
            if not states:
                continue

            aids = _numeric_agent_ids(log)
            agent_skill = {}
            for aid in aids:
                skill = states[0].get(str(aid), {}).get("build_payment", np.nan)
                if np.isfinite(skill):
                    agent_skill[aid] = float(skill)

            waterline = _infer_waterline(log)
            planner_region_by_agent = {
                aid: _planner_region_from_initial_state(log, aid, waterline=waterline)
                for aid in aids
            }
            cutoffs = None
            for tax_event in log.get("PeriodicTax", []):
                if isinstance(tax_event, dict) and tax_event and "cutoffs" in tax_event:
                    cutoffs = np.asarray(tax_event["cutoffs"], dtype=float)
                    break
            if cutoffs is None:
                cutoffs = np.asarray([0.0, 9.7, 39.475, 84.2, 160.725, 204.1, 510.3], dtype=float)
            schedules_by_period = _all_current_planner_schedules_from_actions(
                log,
                period=period,
                rate_disc=rate_disc,
                cutoffs=cutoffs,
            )
            redist_by_period_region = extract_planner_redistribution_by_period(
                log,
                period=period,
                rate_disc=rate_disc,
            )
            redist_lookup = {}
            if not redist_by_period_region.empty:
                for _, row in redist_by_period_region.iterrows():
                    redist_lookup[(int(row["tax_period"]), row["planner_region"])] = {
                        "redistributed": float(row.get("redistributed", 0.0)),
                        "n_agents": max(1, int(row.get("n_agents", 1))),
                    }

            metrics = {
                skill: {
                    "builds": 0.0,
                    "trade_units_bought": 0.0,
                    "gather_units": 0.0,
                    "visible_resources_values": [],
                    "coin_before_tax_values": [],
                    "net_tax_minus_subsidy_values": [],
                }
                for skill in skill_values
            }

            for _, build in events_from_timeline(log, "Build", "builds"):
                aid = int(build.get("builder", -1))
                skill = agent_skill.get(aid, np.nan)
                if np.isfinite(skill):
                    metrics[skill]["builds"] += 1.0

            for _, trade in events_from_timeline(log, "Trade", "trades"):
                aid = int(trade.get("buyer", -1))
                skill = agent_skill.get(aid, np.nan)
                if np.isfinite(skill):
                    metrics[skill]["trade_units_bought"] += float(trade.get("quantity", 1.0))

            for _, gather in events_from_timeline(log, "Gather", "gathers"):
                aid = int(gather.get("agent", -1))
                skill = agent_skill.get(aid, np.nan)
                if np.isfinite(skill):
                    metrics[skill]["gather_units"] += float(gather.get("quantity", gather.get("n", 1.0)))

            tax_days = list(range(period - 1, len(states), period))
            prev_t = 0
            planner_id_by_region = {"top": "p_top", "bottom": "p_bottom"}
            for tax_period, tax_t in enumerate(tax_days, start=1):
                schedules = schedules_by_period.get(tax_period, {})
                for aid in aids:
                    aid_key = str(aid)
                    skill = agent_skill.get(aid, np.nan)
                    if (
                        not np.isfinite(skill)
                        or aid_key not in states[tax_t]
                        or aid_key not in states[prev_t]
                    ):
                        continue
                    state_period_start = states[prev_t][aid_key]
                    state_after_tax = states[tax_t][aid_key]
                    metrics[skill]["coin_before_tax_values"].append(_coin(state_period_start))
                    planner_region = planner_region_by_agent.get(aid)
                    planner_id = planner_id_by_region.get(planner_region)
                    schedule = schedules.get(planner_id, np.zeros(len(cutoffs), dtype=float))
                    income = _coin(state_after_tax) - _coin(state_period_start)
                    tax_due = _tax_due_for_schedule(income, schedule, cutoffs)
                    inventory_coin = float(state_after_tax.get("inventory", {}).get("Coin", 0.0))
                    tax_paid = float(np.minimum(inventory_coin, tax_due))
                    redist_info = redist_lookup.get(
                        (tax_period, planner_region),
                        {"redistributed": 0.0, "n_agents": 1},
                    )
                    subsidy_share = redist_info["redistributed"] / max(1, redist_info["n_agents"])
                    metrics[skill]["net_tax_minus_subsidy_values"].append(
                        tax_paid - subsidy_share
                    )
                    loc = state_after_tax.get("loc", None)
                    if loc is not None:
                        visible_wood = visible_sum(log, tax_t, loc, "Wood")
                        visible_stone = visible_sum(log, tax_t, loc, "Stone")
                        metrics[skill]["visible_resources_values"].append(
                            float(np.nansum([visible_wood, visible_stone]))
                        )
                prev_t = tax_t

            for skill in skill_values:
                values = metrics[skill]
                rows.append({
                    "run": run_name,
                    "label": short_labels.get(run_name, run_name),
                    "rollout_id": rollout_id,
                    "build_payment": skill,
                    "skill_group": skill_labels[skill],
                    "builds": values["builds"],
                    "trade_units_bought": values["trade_units_bought"],
                    "gather_units": values["gather_units"],
                    "visible_resources": (
                        float(np.nanmean(values["visible_resources_values"]))
                        if values["visible_resources_values"] else np.nan
                    ),
                    "coin_before_tax": (
                        float(np.nanmean(values["coin_before_tax_values"]))
                        if values["coin_before_tax_values"] else np.nan
                    ),
                    "net_tax_minus_subsidy": (
                        float(np.nanmean(values["net_tax_minus_subsidy_values"]))
                        if values["net_tax_minus_subsidy_values"] else np.nan
                    ),
                })

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError("No skill mechanism rows could be constructed from these runs.")

    metric_specs = [
        ("builds", "Houses Built", "count per dense log"),
        ("trade_units_bought", "Trade Units Bought", "units per dense log"),
        ("gather_units", "Resources Gathered", "units per dense log"),
        ("visible_resources", "Visible Resources", "avg wood+stone near agent"),
        ("coin_before_tax", "Coin Before Tax Day", "avg coin"),
        ("net_tax_minus_subsidy", "Net Tax After Redistribution", "tax paid - redistribution received"),
    ]

    agg_kwargs = {}
    for metric, _, _ in metric_specs:
        agg_kwargs[metric] = (metric, "mean")
        agg_kwargs[f"{metric}_std"] = (metric, "std")
    summary_df = (
        raw_df
        .groupby(["run", "label", "build_payment", "skill_group"], as_index=False)
        .agg(**agg_kwargs)
    )
    for metric, _, _ in metric_specs:
        summary_df[f"{metric}_std"] = summary_df[f"{metric}_std"].fillna(0.0)

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()

    x = np.arange(len(run_names), dtype=float)
    n_skills = len(skill_values)
    total_width = 0.82
    bar_width = total_width / max(1, n_skills)
    offsets = (np.arange(n_skills) - (n_skills - 1) / 2.0) * bar_width
    colors = _make_agent_colors(range(n_skills))

    for ax, (metric, title, ylabel) in zip(axes_flat, metric_specs):
        for skill_idx, skill in enumerate(skill_values):
            vals = []
            errs = []
            for run_name in run_names:
                match = summary_df[
                    (summary_df["run"] == run_name)
                    & (summary_df["build_payment"] == skill)
                ]
                vals.append(float(match[metric].iloc[0]) if len(match) else np.nan)
                errs.append(float(match[f"{metric}_std"].iloc[0]) if len(match) else 0.0)
            ax.bar(
                x + offsets[skill_idx],
                vals,
                width=bar_width * 0.92,
                color=colors[skill_idx % len(colors)],
                edgecolor="white",
                linewidth=0.7,
                label=skill_labels[skill],
                yerr=errs if show_std else None,
                error_kw=dict(ecolor="0.25", elinewidth=0.8, capsize=2.5, capthick=0.8),
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([short_labels.get(name, name) for name in run_names], rotation=0)
        ax.grid(True, axis="y", alpha=0.25)

    for ax in axes_flat[len(metric_specs):]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(labels)),
            frameon=True,
        )
    fig.suptitle("Skill-Group Mechanisms Across Runs", fontsize=14, fontweight="bold")
    return fig, summary_df, raw_df

def plot_planner_rewards(run, smooth_window=25):
    df = run["metrics"].copy()
    df = df.reset_index(drop=True)
    x = range(len(df))

    fig, ax = plt.subplots(figsize=(10, 5))

    for col in ["policy_reward_mean/p_top", "policy_reward_mean/p_bottom"]:
        y = df[col].astype(float)
        if smooth_window > 1:
            y = y.rolling(smooth_window, min_periods=1).mean()
        ax.plot(x, y, label=col)

    ax.set_title("Planner policy rewards")
    ax.set_xlabel("Training row")
    ax.set_ylabel("Mean policy reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
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
            "gini_final_coin",
            "avg_social_welfare",
            "mean_final_labor",
            "n_trades",
            "n_builds",
        ]

    def gini(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return np.nan
        if np.any(values < 0):
            values = values - np.min(values)
        total = np.sum(values)
        if total <= 0:
            return 0.0
        values = np.sort(values)
        n = len(values)
        return float((2 * np.sum((np.arange(1, n + 1) * values))) / (n * total) - (n + 1) / n)

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

        def social_welfare_from_coins(coins):
            coins = np.asarray(coins, dtype=float)
            coins = coins[np.isfinite(coins)]
            if len(coins) == 0:
                return np.nan
            return float(np.sum(coins) * (1.0 - gini(coins)))

        social_welfare_by_timestep = []
        for state in states:
            coins_t = []
            for aid in agent_ids:
                agent_state = state.get(str(aid), {})
                inventory = agent_state.get("inventory", {})
                escrow = agent_state.get("escrow", {})
                coins_t.append(
                    float(inventory.get("Coin", 0.0)) + float(escrow.get("Coin", 0.0))
                )
            social_welfare_by_timestep.append(social_welfare_from_coins(coins_t))

        mean_social_welfare = (
            float(np.nanmean(social_welfare_by_timestep))
            if len(social_welfare_by_timestep) else np.nan
        )
        final_social_welfare = social_welfare_from_coins(final_coin)

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
            "gini_final_coin": gini(final_coin),
            "avg_social_welfare": mean_social_welfare,
            "mean_social_welfare": mean_social_welfare,
            "final_social_welfare": final_social_welfare,
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
    dense_summary_lookup = {}

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
        dense_summary_lookup[run_name] = ep_df.mean(numeric_only=True).to_dict()

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

    for metric in metrics:
        if metric not in summary_df.columns:
            summary_df[metric] = np.nan
        for run_name in run_names:
            if pd.isna(summary_df.at[run_name, metric]):
                summary_df.at[run_name, metric] = dense_summary_lookup.get(run_name, {}).get(metric, np.nan)

    metrics = [m for m in metrics if m in summary_df.columns and summary_df[m].notna().any()]

    fig_width = max(8, 1.25 * len(metrics))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 4.8), constrained_layout=False)
    welfare_metrics = {
        "avg_social_welfare",
        "mean_social_welfare",
        "final_social_welfare",
        "social_welfare_coin_eq_times_prod",
    }
    social_metrics = [m for m in metrics if m in welfare_metrics]
    ax_social = ax.twinx() if social_metrics else None

    x = np.arange(len(metrics))
    group_width = 0.62
    bar_width = group_width / max(1, len(run_names))
    offsets = (np.arange(len(run_names)) - (len(run_names) - 1) / 2) * bar_width

    for run_i, name in enumerate(run_names):
        for target_ax, target_metrics, alpha in [
            (ax, [m for m in metrics if m not in welfare_metrics], 0.9),
            (ax_social, social_metrics, 0.72),
        ]:
            if target_ax is None or not target_metrics:
                continue
            positions = np.array([metrics.index(metric) for metric in target_metrics], dtype=float)
            vals = summary_df.loc[name, target_metrics].to_numpy(dtype=float)
            yerr = np.array(
                [err_lookup[name].get(metric, np.nan) for metric in target_metrics],
                dtype=float,
            )
            yerr = np.where(np.isfinite(yerr), yerr, 0.0)

            target_ax.bar(
                positions + offsets[run_i],
                vals,
                color=colors[name],
                width=bar_width * 0.82,
                alpha=alpha,
                yerr=None if errorbar is None else yerr,
                capsize=3 if errorbar is not None else 0,
                ecolor="black",
                linewidth=0,
                label=short_labels[name],
            )

    ax.set_title("Summary Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels([metric.replace("_", " ").title() for metric in metrics], rotation=20, ha="right")
    ax.set_ylabel("Summary metrics")
    ax.grid(True, axis="y", alpha=0.3)
    if ax_social is not None:
        ax_social.set_ylabel("Social welfare")
        ax_social.grid(False)

    if show_legend:
        legend_handles = [
            Patch(facecolor=colors[name], label=f"{short_labels[name]}")
            for name in run_names
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(run_names), 6),
            frameon=True,
            fontsize=10,
        )

        fig.subplots_adjust(bottom=0.28, top=0.90, left=0.08, right=0.98)
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

def extract_market_size_price_trade_over_time(log, resources=("Wood", "Stone")):
    """
    Build timestep-level market diagnostics from a dense log.

    Market size is the total amount of the selected resources currently in
    escrow. Prices and trade activity are taken from executed trades.
    """
    import numpy as np
    import pandas as pd

    n_steps = max(len(log.get("states", [])), len(log.get("Trade", [])))
    rows = []

    for t in range(n_steps):
        state = log["states"][t] if t < len(log.get("states", [])) else {}
        trades_t = []
        if t < len(log.get("Trade", [])):
            raw_trades = log["Trade"][t]
            trades_t = raw_trades.get("trades", []) if isinstance(raw_trades, dict) else raw_trades

        agents = [k for k in state.keys() if str(k).isdigit()]
        market_size = 0.0
        for aid in agents:
            agent_state = state[str(aid)]
            escrow = agent_state.get("escrow", {})
            market_size += sum(float(escrow.get(r, 0.0)) for r in resources)

        resource_trades = [
            tr for tr in trades_t
            if isinstance(tr, dict) and tr.get("commodity") in resources
        ]
        prices = [float(tr["price"]) for tr in resource_trades if "price" in tr]

        row = {
            "timestep": t,
            "market_size": market_size,
            "mean_price": np.nan if len(prices) == 0 else float(np.mean(prices)),
            "trade_count": len(resource_trades),
        }

        for resource in resources:
            trades_r = [tr for tr in resource_trades if tr.get("commodity") == resource]
            prices_r = [float(tr["price"]) for tr in trades_r if "price" in tr]
            row[f"{resource.lower()}_price"] = (
                np.nan if len(prices_r) == 0 else float(np.mean(prices_r))
            )
            row[f"{resource.lower()}_trades"] = len(trades_r)

        rows.append(row)

    return pd.DataFrame(rows)

def compare_market_size_prices_trade_activity(
    runs,
    short_labels=None,
    resources=("Wood", "Stone"),
    mode="single",
    smooth_window=1,
    price_fill_method="interpolate",
    show_std=True,
    show_legend=True,
    figsize=(12, 10),
):
    """
    Compare market size, transaction prices, and trade activity across runs.

    Parameters mirror the other comparison helpers: pass a list of loaded runs,
    optionally slice it first (for example, runs[:2]).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def extract_episode_logs(obj):
        if obj is None:
            return []
        if isinstance(obj, dict) and ("states" in obj or "Trade" in obj):
            return [obj]
        if isinstance(obj, dict):
            return [
                v for v in obj.values()
                if isinstance(v, dict) and ("states" in v or "Trade" in v)
            ]
        if isinstance(obj, list):
            return [
                v for v in obj
                if isinstance(v, dict) and ("states" in v or "Trade" in v)
            ]
        return []

    def smooth_series(series, window):
        if window <= 1:
            return series
        return series.rolling(window=window, min_periods=1, center=True).mean()

    def display_price_series(series):
        if price_fill_method is None:
            out = series
        elif price_fill_method == "interpolate":
            out = series.interpolate(limit_direction="both")
        elif price_fill_method == "ffill":
            out = series.ffill().bfill()
        else:
            raise ValueError("price_fill_method must be None, 'interpolate', or 'ffill'")
        return smooth_series(out, smooth_window)

    def mean_frame(frames):
        min_len = min(len(df) for df in frames)
        aligned = [df.iloc[:min_len].reset_index(drop=True) for df in frames]
        numeric_cols = aligned[0].select_dtypes(include=[np.number]).columns
        stacked = np.stack([df[numeric_cols].to_numpy(dtype=float) for df in aligned])
        valid = np.sum(np.isfinite(stacked), axis=0)
        sums = np.nansum(stacked, axis=0)
        means = np.full_like(sums, np.nan, dtype=float)
        np.divide(sums, valid, out=means, where=valid > 0)
        out = pd.DataFrame(means, columns=numeric_cols)
        stds = np.nanstd(stacked, axis=0, ddof=1) if len(frames) > 1 else np.zeros_like(means)
        stds = np.where(valid > 1, stds, 0.0)
        for col_idx, col in enumerate(numeric_cols):
            out[f"{col}_std"] = stds[:, col_idx]
        out["timestep"] = np.arange(min_len)
        return out

    run_names = [run["name"] for run in runs]
    if short_labels is None:
        short_labels = {name: f"Run {i + 1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    colors_list = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
        "#ff7f0e", "#8c564b", "#e377c2", "#17becf"
    ]
    colors = {name: colors_list[i % len(colors_list)] for i, name in enumerate(run_names)}

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    out_frames = []

    for run in runs:
        name = run["name"]
        label = short_labels[name]

        if mode == "single":
            log = run.get("dense_log", None)
            if log is None:
                eps = extract_episode_logs(run.get("dense_logs", None))
                log = eps[0] if len(eps) > 0 else None
            if log is None:
                continue
            df = extract_market_size_price_trade_over_time(log, resources=resources)

        elif mode == "average":
            eps = extract_episode_logs(run.get("dense_logs", None))
            if len(eps) == 0 and run.get("dense_log", None) is not None:
                eps = [run["dense_log"]]
            frames = [
                extract_market_size_price_trade_over_time(ep, resources=resources)
                for ep in eps
            ]
            frames = [df for df in frames if not df.empty]
            if len(frames) == 0:
                continue
            df = mean_frame(frames)

        else:
            raise ValueError("mode must be 'single' or 'average'")

        df = df.copy()
        df["run"] = name
        df["label"] = label
        out_frames.append(df)

        x = df["timestep"]
        color = colors[name]

        market_size_y = smooth_series(df["market_size"], smooth_window)
        axes[0].plot(
            x,
            market_size_y,
            color=color,
            linewidth=2.2,
            label=label,
        )
        if show_std and mode == "average" and "market_size_std" in df:
            market_size_sd = smooth_series(df["market_size_std"], smooth_window).fillna(0.0)
            axes[0].fill_between(
                x,
                market_size_y - market_size_sd,
                market_size_y + market_size_sd,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        price_y = display_price_series(df["mean_price"])
        axes[1].plot(
            x,
            price_y,
            color=color,
            linewidth=2.2,
            label=label,
        )
        if show_std and mode == "average" and "mean_price_std" in df:
            price_sd = display_price_series(df["mean_price_std"]).fillna(0.0)
            axes[1].fill_between(
                x,
                price_y - price_sd,
                price_y + price_sd,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        trade_count_y = smooth_series(df["trade_count"], smooth_window)
        axes[2].plot(
            x,
            trade_count_y,
            color=color,
            linewidth=2.2,
            label=label,
        )
        if show_std and mode == "average" and "trade_count_std" in df:
            trade_count_sd = smooth_series(df["trade_count_std"], smooth_window).fillna(0.0)
            axes[2].fill_between(
                x,
                trade_count_y - trade_count_sd,
                trade_count_y + trade_count_sd,
                color=color,
                alpha=0.16,
                linewidth=0,
            )

    title_suffix = "Single Rollout" if mode == "single" else "Mean Across Dense Logs"
    axes[0].set_title(f"Market Size: Escrowed Resources ({title_suffix})")
    axes[0].set_ylabel("Escrowed Wood + Stone")
    axes[1].set_title("Average Transaction Price")
    axes[1].set_ylabel("Price")
    axes[2].set_title("Trade Activity")
    axes[2].set_ylabel("Trades")
    axes[2].set_xlabel("Timestep")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    if show_legend:
        axes[0].legend(loc="best", frameon=True)

    fig.tight_layout()
    out_df = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    return fig, out_df

def extract_trade_region_distribution(log, volume_field="price"):
    """
    Summarize traded volume by whether the buyer and seller are in the same region.

    Volume defaults to the executed trade price, which is the coin value of a
    one-unit Wood/Stone trade in these logs.
    """
    import numpy as np
    import pandas as pd

    states = log.get("states", [])
    waterline = _infer_waterline(log)
    rows = []

    for t, item in enumerate(log.get("Trade", [])):
        trades = item.get("trades", []) if isinstance(item, dict) else item
        if not isinstance(trades, list):
            continue

        state_t = states[min(t, len(states) - 1)] if states else {}

        for trade in trades:
            if not isinstance(trade, dict):
                continue

            buyer = trade.get("buyer")
            seller = trade.get("seller")
            buyer_region = trade.get("buyer_region", None)
            seller_region = trade.get("seller_region", None)

            if buyer_region is None and buyer is not None and str(buyer) in state_t:
                buyer_region = _location_region_from_state(
                    state_t[str(buyer)], waterline=waterline
                )
            if seller_region is None and seller is not None and str(seller) in state_t:
                seller_region = _location_region_from_state(
                    state_t[str(seller)], waterline=waterline
                )

            if "cross_region" in trade:
                is_cross = bool(trade.get("cross_region", False))
            else:
                is_cross = (
                    buyer_region is not None
                    and seller_region is not None
                    and buyer_region != seller_region
                )

            if is_cross and buyer_region is not None and seller_region is not None:
                route_type = "cross region"
                route = f"{seller_region} to {buyer_region}"
            elif is_cross:
                trade_region = trade.get("region", "unknown")
                route_type = "cross region"
                route = f"cross region: {trade_region}"
            elif buyer_region is not None:
                route_type = "within region"
                route = f"within {buyer_region}"
            else:
                trade_region = trade.get("region", "unknown")
                route_type = "within region"
                route = f"within {trade_region}"

            rows.append({
                "timestep": t,
                "route_type": route_type,
                "route": route,
                "buyer_region": buyer_region,
                "seller_region": seller_region,
                "commodity": trade.get("commodity", "unknown"),
                "units": float(trade.get("quantity", 1.0)),
                "volume": float(trade.get(volume_field, trade.get("price", 1.0))),
                "price": float(trade.get("price", np.nan)),
                "buyer_cost": float(trade.get("cost", trade.get("price", np.nan))),
                "tariff": float(trade.get("tariff", 0.0)),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "timestep", "route_type", "route", "commodity", "units", "volume",
            "price", "buyer_cost", "tariff", "buyer_region", "seller_region"
        ])

    return pd.DataFrame(rows)

def extract_planner_redistribution_table(log, period=100, rate_disc=0.05):
    """
    Summarize each planner region's redistribution and income-tax contribution.

    The saved ``PeriodicTax`` log can contain only one regional tax component, so
    this reconstructs both planners from planner actions, period incomes, and
    the tax bracket cutoffs. ``redistributed`` is income tax plus any logged
    trade-tariff revenue for that region.
    """
    import numpy as np
    import pandas as pd

    waterline = _infer_waterline(log)
    aids = _numeric_agent_ids(log)
    planner_region_by_agent = {
        aid: _planner_region_from_initial_state(log, aid, waterline=waterline)
        for aid in aids
    }
    planner_id_by_region = {"top": "p_top", "bottom": "p_bottom"}

    cutoffs = None
    for item in log.get("PeriodicTax", []):
        if isinstance(item, dict) and "cutoffs" in item:
            cutoffs = np.asarray(item["cutoffs"], dtype=float)
            break
    if cutoffs is None:
        cutoffs = np.asarray([0.0, 9.7, 39.475, 84.2, 160.725, 204.1, 510.3], dtype=float)

    schedules_by_period = _all_current_planner_schedules_from_actions(
        log,
        period=period,
        rate_disc=rate_disc,
        cutoffs=cutoffs,
    )

    rows = []
    states = log.get("states", [])
    tax_days = list(range(period - 1, max(0, len(states) - 1), period))
    prev_t = 0

    for tax_period, tax_t in enumerate(tax_days, start=1):
        schedules = schedules_by_period.get(tax_period, {})
        state_prev = states[prev_t]
        state_tax = states[tax_t]

        start_t = 0 if tax_period == 1 else prev_t + 1
        tariff_by_region = {"top": 0.0, "bottom": 0.0}
        for trade_t in range(start_t, tax_t + 1):
            if trade_t >= len(log.get("Trade", [])):
                continue
            item = log["Trade"][trade_t]
            trades = item.get("trades", []) if isinstance(item, dict) else item
            if not isinstance(trades, list):
                continue
            state_trade = states[min(trade_t, len(states) - 1)]
            for trade in trades:
                buyer = trade.get("buyer")
                if buyer is not None and str(buyer) in state_trade:
                    region = _location_region_from_state(
                        state_trade[str(buyer)], waterline=waterline
                    )
                else:
                    region = trade.get("region")
                if region in tariff_by_region:
                    tariff_by_region[region] += float(trade.get("tariff", 0.0) or 0.0)

        for region, planner_id in planner_id_by_region.items():
            schedule = schedules.get(planner_id, np.zeros(len(cutoffs), dtype=float))
            region_aids = [
                aid for aid in aids
                if planner_region_by_agent[aid] == region
            ]

            income_total = 0.0
            income_tax_total = 0.0
            for aid in region_aids:
                income = _coin(state_tax[str(aid)]) - _coin(state_prev[str(aid)])
                income_total += float(income)
                tax_due = _tax_due_for_schedule(income, schedule, cutoffs)
                inventory_coin = float(
                    state_tax[str(aid)].get("inventory", {}).get("Coin", 0.0)
                )
                income_tax_total += float(np.minimum(inventory_coin, tax_due))

            redistributed = income_tax_total + tariff_by_region[region]
            rows.append({
                "timestep": tax_t,
                "tax_period": tax_period,
                "planner_region": region,
                "income": income_total,
                "income_tax_collected": income_tax_total,
                "tariff_revenue": tariff_by_region[region],
                "redistributed": redistributed,
                "n_agents": len(region_aids),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "planner_region", "income_tax_collected", "redistributed",
            "non_income_tax_redistribution"
        ])

    df_agent = pd.DataFrame(rows)
    df = (
        df_agent
        .groupby("planner_region", as_index=False)
        .agg(
            income=("income", "sum"),
            income_tax_collected=("income_tax_collected", "sum"),
            tariff_revenue=("tariff_revenue", "sum"),
            redistributed=("redistributed", "sum"),
            n_agents=("n_agents", "max"),
        )
    )
    df["income_tax_funded_redistribution"] = df["income_tax_collected"]
    df["non_income_tax_redistribution"] = (
        df["redistributed"] - df["income_tax_funded_redistribution"]
    ).clip(lower=0.0)

    return df

def extract_planner_redistribution_by_period(log, period=100, rate_disc=0.05, travel_cost_coin=0.0):
    """Return planner redistribution and income-tax contribution by tax period."""
    import numpy as np
    import pandas as pd

    waterline = _infer_waterline(log)
    aids = _numeric_agent_ids(log)
    planner_region_by_agent = {
        aid: _planner_region_from_initial_state(log, aid, waterline=waterline)
        for aid in aids
    }
    planner_id_by_region = {"top": "p_top", "bottom": "p_bottom"}

    cutoffs = None
    for item in log.get("PeriodicTax", []):
        if isinstance(item, dict) and "cutoffs" in item:
            cutoffs = np.asarray(item["cutoffs"], dtype=float)
            break
    if cutoffs is None:
        cutoffs = np.asarray([0.0, 9.7, 39.475, 84.2, 160.725, 204.1, 510.3], dtype=float)

    schedules_by_period = _all_current_planner_schedules_from_actions(
        log,
        period=period,
        rate_disc=rate_disc,
        cutoffs=cutoffs,
    )

    states = log.get("states", [])
    tax_days = list(range(period - 1, max(0, len(states) - 1), period))
    prev_t = 0
    rows = []

    for tax_period, tax_t in enumerate(tax_days, start=1):
        schedules = schedules_by_period.get(tax_period, {})
        state_prev = states[prev_t]
        state_tax = states[tax_t]
        start_t = 0 if tax_period == 1 else prev_t + 1

        tariff_by_region = {"top": 0.0, "bottom": 0.0}
        travel_by_region = {"top": 0.0, "bottom": 0.0}
        for trade_t in range(start_t, tax_t + 1):
            if trade_t >= len(log.get("Trade", [])):
                continue
            item = log["Trade"][trade_t]
            trades = item.get("trades", []) if isinstance(item, dict) else item
            if not isinstance(trades, list):
                continue
            state_trade = states[min(trade_t, len(states) - 1)]
            for trade in trades:
                buyer = trade.get("buyer")
                if buyer is not None and str(buyer) in state_trade:
                    region = _location_region_from_state(
                        state_trade[str(buyer)], waterline=waterline
                    )
                else:
                    region = trade.get("region")
                if region in tariff_by_region:
                    tariff_by_region[region] += float(trade.get("tariff", 0.0) or 0.0)

        for event in _iter_travel_events(log):
            event_t = int(event.get("t", -1))
            if event_t < start_t or event_t > tax_t:
                continue
            origin = event.get("from", None)
            if origin is None:
                continue
            origin_region = "top" if int(origin[0]) < waterline else "bottom"
            travel_amount = event.get(
                "travel_cost_coin",
                event.get("cost", event.get("tax", travel_cost_coin)),
            )
            if origin_region in travel_by_region:
                travel_by_region[origin_region] += float(travel_amount or 0.0)

        for region, planner_id in planner_id_by_region.items():
            schedule = schedules.get(planner_id, np.zeros(len(cutoffs), dtype=float))
            region_aids = [
                aid for aid in aids
                if planner_region_by_agent[aid] == region
            ]

            income_total = 0.0
            income_tax_total = 0.0
            for aid in region_aids:
                income = _coin(state_tax[str(aid)]) - _coin(state_prev[str(aid)])
                income_total += float(income)
                tax_due = _tax_due_for_schedule(income, schedule, cutoffs)
                inventory_coin = float(
                    state_tax[str(aid)].get("inventory", {}).get("Coin", 0.0)
                )
                income_tax_total += float(np.minimum(inventory_coin, tax_due))

            redistributed = income_tax_total + tariff_by_region[region] + travel_by_region[region]
            rows.append({
                "tax_period": tax_period,
                "timestep": tax_t,
                "planner_region": region,
                "income": income_total,
                "income_tax_collected": income_tax_total,
                "travel_tax_revenue": travel_by_region[region],
                "tariff_revenue": tariff_by_region[region],
                "redistributed": redistributed,
                "income_tax_funded_redistribution": income_tax_total,
                "travel_tax_funded_redistribution": travel_by_region[region],
                "trade_tariff_funded_redistribution": tariff_by_region[region],
                "non_income_tax_redistribution": max(0.0, redistributed - income_tax_total),
                "n_agents": len(region_aids),
            })

        prev_t = tax_t

    return pd.DataFrame(rows)

def plot_trade_enabled_run_trade_and_redistribution(
    run,
    mode="single",
    period=100,
    rate_disc=0.05,
    dense_log_key=None,
    show_std=True,
    figsize=(14, 10),
):
    """
    Plot one trade-enabled run, either one dense log or the average across logs.

    Top-left: units of Wood and Stone traded within-region versus cross-region
    over the full rollout. Cross-region average trade prices include their
    logged tariff so price comparisons reflect buyer cost. Bottom:
    redistribution by planner and tax period, split into income-tax-funded and
    other redistribution.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def episode_logs_from_run(run):
        if mode == "single":
            if dense_log_key is not None and isinstance(run.get("dense_logs"), dict):
                return [(dense_log_key, run["dense_logs"][dense_log_key])]
            return [(0, run["dense_log"])]

        dense_logs = run.get("dense_logs", None)
        if isinstance(dense_logs, dict):
            return [(k, v) for k, v in dense_logs.items() if isinstance(v, dict)]
        if isinstance(dense_logs, list):
            return [(i, v) for i, v in enumerate(dense_logs) if isinstance(v, dict)]
        return [(0, run["dense_log"])] if run.get("dense_log", None) is not None else []

    logs = episode_logs_from_run(run)
    if not logs:
        raise ValueError("No dense logs found for this run.")

    run_summary = run.get("summary", {}) if isinstance(run, dict) else {}
    run_config = run_summary.get("config", {}) if isinstance(run_summary, dict) else {}
    if not run_config and isinstance(run, dict) and run.get("run_dir"):
        config_path = os.path.join(run["run_dir"], "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                run_config = json.load(f)

    def infer_travel_cost_coin(config):
        if not isinstance(config, dict):
            return 0.0
        for key in ["travel_cost_coin_phase3b", "travel_cost_coin_phase3a", "travel_cost_coin"]:
            if key in config:
                return float(config[key])
        phase_configs = [
            value for key, value in config.items()
            if isinstance(value, dict) and "components" in value
        ]
        for phase_config in reversed(phase_configs):
            for component in phase_config.get("components", []):
                if (
                    isinstance(component, (list, tuple))
                    and len(component) >= 2
                    and component[0] == "CrossWaterTravel"
                    and isinstance(component[1], dict)
                    and "travel_cost_coin" in component[1]
                ):
                    return float(component[1]["travel_cost_coin"])
        return 0.0

    travel_cost_coin = infer_travel_cost_coin(run_config)

    trade_frames = []
    redistrib_frames = []
    for rollout_id, log in logs:
        trade_df = extract_trade_region_distribution(log)
        if not trade_df.empty:
            trade_df = trade_df.copy()
            trade_df["rollout_id"] = rollout_id
            trade_df["route_group"] = np.where(
                trade_df["route_type"].eq("cross region"),
                "cross region",
                "within region",
            )
            trade_frames.append(trade_df)

        redist_df = extract_planner_redistribution_by_period(
            log,
            period=period,
            rate_disc=rate_disc,
            travel_cost_coin=travel_cost_coin,
        )
        if not redist_df.empty:
            redist_df = redist_df.copy()
            redist_df["rollout_id"] = rollout_id
            redistrib_frames.append(redist_df)

    trade_raw = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    redist_raw = (
        pd.concat(redistrib_frames, ignore_index=True)
        if redistrib_frames
        else pd.DataFrame()
    )

    if not trade_raw.empty:
        trade_raw = trade_raw.copy()
        trade_raw["tax_period"] = (trade_raw["timestep"].astype(int) // int(period)) + 1
        trade_raw["price_with_cross_tariff"] = trade_raw["price"]
        cross_trade = trade_raw["route_group"].eq("cross region")
        trade_raw.loc[cross_trade, "price_with_cross_tariff"] = (
            trade_raw.loc[cross_trade, "price"] + trade_raw.loc[cross_trade, "tariff"]
        )

    if trade_raw.empty:
        trade_units = pd.DataFrame(columns=["tax_period", "commodity", "route_group", "units", "units_std"])
        price_summary = pd.DataFrame(columns=["commodity", "route_group", "avg_price", "avg_price_std"])
    elif mode == "average":
        per_rollout_units = (
            trade_raw
            .groupby(["rollout_id", "tax_period", "commodity", "route_group"], as_index=False)
            .agg(units=("units", "sum"))
        )
        trade_units = (
            per_rollout_units
            .groupby(["tax_period", "commodity", "route_group"], as_index=False)
            .agg(units=("units", "mean"), units_std=("units", "std"))
        )
        trade_units["units_std"] = trade_units["units_std"].fillna(0.0)
        per_rollout_prices = (
            trade_raw
            .groupby(["rollout_id", "commodity", "route_group"], as_index=False)
            .agg(avg_price=("price_with_cross_tariff", "mean"))
        )
        price_summary = (
            per_rollout_prices
            .groupby(["commodity", "route_group"], as_index=False)
            .agg(avg_price=("avg_price", "mean"), avg_price_std=("avg_price", "std"))
        )
        price_summary["avg_price_std"] = price_summary["avg_price_std"].fillna(0.0)
    else:
        trade_units = (
            trade_raw
            .groupby(["tax_period", "commodity", "route_group"], as_index=False)
            .agg(units=("units", "sum"))
        )
        trade_units["units_std"] = 0.0
        price_summary = (
            trade_raw
            .groupby(["commodity", "route_group"], as_index=False)
            .agg(avg_price=("price_with_cross_tariff", "mean"))
        )
        price_summary["avg_price_std"] = 0.0

    if redist_raw.empty:
        redist = pd.DataFrame(columns=[
            "tax_period", "planner_region", "income_tax_funded_redistribution",
            "travel_tax_funded_redistribution", "trade_tariff_funded_redistribution",
            "non_income_tax_redistribution", "redistributed", "redistributed_std"
        ])
    elif mode == "average":
        redist = (
            redist_raw
            .groupby(["tax_period", "planner_region"], as_index=False)
            .agg(
                income=("income", "mean"),
                income_tax_collected=("income_tax_collected", "mean"),
                tariff_revenue=("tariff_revenue", "mean"),
                redistributed=("redistributed", "mean"),
                redistributed_std=("redistributed", "std"),
                income_tax_funded_redistribution=("income_tax_funded_redistribution", "mean"),
                income_tax_funded_redistribution_std=("income_tax_funded_redistribution", "std"),
                travel_tax_funded_redistribution=("travel_tax_funded_redistribution", "mean"),
                travel_tax_funded_redistribution_std=("travel_tax_funded_redistribution", "std"),
                trade_tariff_funded_redistribution=("trade_tariff_funded_redistribution", "mean"),
                trade_tariff_funded_redistribution_std=("trade_tariff_funded_redistribution", "std"),
                non_income_tax_redistribution=("non_income_tax_redistribution", "mean"),
                non_income_tax_redistribution_std=("non_income_tax_redistribution", "std"),
            )
        )
        redist[[
            "redistributed_std",
            "income_tax_funded_redistribution_std",
            "travel_tax_funded_redistribution_std",
            "trade_tariff_funded_redistribution_std",
            "non_income_tax_redistribution_std",
        ]] = redist[[
            "redistributed_std",
            "income_tax_funded_redistribution_std",
            "travel_tax_funded_redistribution_std",
            "trade_tariff_funded_redistribution_std",
            "non_income_tax_redistribution_std",
        ]].fillna(0.0)
    else:
        redist = redist_raw.copy()
        redist["redistributed_std"] = 0.0
        redist["income_tax_funded_redistribution_std"] = 0.0
        redist["travel_tax_funded_redistribution_std"] = 0.0
        redist["trade_tariff_funded_redistribution_std"] = 0.0
        redist["non_income_tax_redistribution_std"] = 0.0

    buyer_route_cols = [
        "tax_period", "buyer_region", "route_group", "units", "units_std",
        "avg_price", "avg_price_std", "n_dense_logs",
    ]
    if trade_raw.empty or "buyer_region" not in trade_raw:
        buyer_route = pd.DataFrame(columns=buyer_route_cols)
    else:
        buyer_raw = trade_raw[
            trade_raw["buyer_region"].isin(["top", "bottom"])
            & trade_raw["route_group"].isin(["within region", "cross region"])
        ].copy()
        if buyer_raw.empty:
            buyer_route = pd.DataFrame(columns=buyer_route_cols)
        elif mode == "average":
            per_rollout_buyer_route = (
                buyer_raw
                .groupby(["rollout_id", "tax_period", "buyer_region", "route_group"], as_index=False)
                .agg(
                    units=("units", "sum"),
                    avg_price=("price_with_cross_tariff", "mean"),
                )
            )
            full_index = pd.MultiIndex.from_product(
                [
                    [rollout_id for rollout_id, _ in logs],
                    sorted(trade_raw["tax_period"].dropna().unique()),
                    ["top", "bottom"],
                    ["within region", "cross region"],
                ],
                names=["rollout_id", "tax_period", "buyer_region", "route_group"],
            )
            per_rollout_buyer_route = (
                per_rollout_buyer_route
                .set_index(["rollout_id", "tax_period", "buyer_region", "route_group"])
                .reindex(full_index, fill_value=0.0)
                .reset_index()
            )
            buyer_route = (
                per_rollout_buyer_route
                .groupby(["tax_period", "buyer_region", "route_group"], as_index=False)
                .agg(
                    units=("units", "mean"),
                    units_std=("units", "std"),
                    avg_price=("avg_price", "mean"),
                    avg_price_std=("avg_price", "std"),
                    n_dense_logs=("units", "count"),
                )
            )
            buyer_route["units_std"] = buyer_route["units_std"].fillna(0.0)
            buyer_route["avg_price_std"] = buyer_route["avg_price_std"].fillna(0.0)
        else:
            buyer_route = (
                buyer_raw
                .groupby(["tax_period", "buyer_region", "route_group"], as_index=False)
                .agg(
                    units=("units", "sum"),
                    avg_price=("price_with_cross_tariff", "mean"),
                )
            )
            buyer_route["units_std"] = 0.0
            buyer_route["avg_price_std"] = 0.0
            buyer_route["n_dense_logs"] = 1

    fig = plt.figure(figsize=(figsize[0], max(6, figsize[1] * 0.68)), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.25, 0.9],
    )
    ax_trade = fig.add_subplot(gs[0, :])
    ax_pies = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    commodities = ["Wood", "Stone"]
    route_groups = ["within region", "cross region"]
    commodity_colors = {"Wood": "#8fcf7b", "Stone": "#eadfbd"}
    cross_alpha = 0.45
    all_trade_periods = sorted(trade_units["tax_period"].dropna().unique()) if not trade_units.empty else []
    all_redist_periods = sorted(redist["tax_period"].dropna().unique()) if not redist.empty else []
    plot_periods = sorted(set(all_trade_periods) | set(all_redist_periods))
    if not plot_periods:
        plot_periods = list(range(1, 26))

    period_x = np.arange(len(plot_periods))
    bar_width = 0.34
    commodity_offsets = {"Wood": -bar_width / 2, "Stone": bar_width / 2}

    for commodity in commodities:
        within_vals = []
        cross_vals = []
        within_err = []
        cross_err = []
        for tax_period in plot_periods:
            within_match = trade_units[
                (trade_units["tax_period"] == tax_period)
                & (trade_units["commodity"] == commodity)
                & (trade_units["route_group"] == "within region")
            ]
            cross_match = trade_units[
                (trade_units["tax_period"] == tax_period)
                & (trade_units["commodity"] == commodity)
                & (trade_units["route_group"] == "cross region")
            ]
            within_vals.append(float(within_match["units"].sum()) if len(within_match) else 0.0)
            cross_vals.append(float(cross_match["units"].sum()) if len(cross_match) else 0.0)
            within_err.append(float(within_match["units_std"].sum()) if len(within_match) else 0.0)
            cross_err.append(float(cross_match["units_std"].sum()) if len(cross_match) else 0.0)

        pos = period_x + commodity_offsets[commodity]
        ax_trade.bar(
            pos,
            within_vals,
            width=bar_width,
            color=commodity_colors[commodity],
            edgecolor="0.25",
            linewidth=0.5,
            label=f"{commodity}: within",
            yerr=within_err if mode == "average" and show_std else None,
            error_kw=dict(ecolor="0.25", elinewidth=0.8, capsize=2, capthick=0.8),
        )
        ax_trade.bar(
            pos,
            cross_vals,
            bottom=within_vals,
            width=bar_width,
            color=commodity_colors[commodity],
            alpha=cross_alpha,
            edgecolor="0.25",
            linewidth=0.5,
            hatch="//",
            label=f"{commodity}: cross",
            yerr=cross_err if mode == "average" and show_std else None,
            error_kw=dict(ecolor="0.25", elinewidth=0.8, capsize=2, capthick=0.8),
        )

    title_suffix = "single dense log" if mode == "single" else "average across dense logs"
    ax_trade.set_title(f"Units Traded by Tax Period ({title_suffix})")
    ax_trade.set_ylabel("Units traded")
    ax_trade.set_xlabel("Tax period")
    ax_trade.set_xticks(period_x)
    ax_trade.set_xticklabels([str(int(k)) for k in plot_periods], fontsize=8)
    ax_trade.legend(loc="upper left", ncol=4, frameon=True, fontsize=8)
    ax_trade.grid(True, axis="y", alpha=0.3)

    for ax, commodity in zip(ax_pies, commodities):
        values = []
        price_lines = []
        for route_group in route_groups:
            units_match = trade_units[
                (trade_units["commodity"] == commodity)
                & (trade_units["route_group"] == route_group)
            ]
            values.append(float(units_match["units"].sum()) if len(units_match) else 0.0)
            price_match = price_summary[
                (price_summary["commodity"] == commodity)
                & (price_summary["route_group"] == route_group)
            ]
            price = float(price_match["avg_price"].mean()) if len(price_match) else np.nan
            price_std = (
                float(price_match["avg_price_std"].mean())
                if len(price_match) and "avg_price_std" in price_match
                else np.nan
            )
            price_label = "within" if route_group == "within region" else "cross"
            if pd.isna(price):
                price_text = "n/a"
            elif show_std and mode == "average" and not pd.isna(price_std):
                price_text = f"{price:.2f} +/- {price_std:.2f}"
            else:
                price_text = f"{price:.2f}"
            if route_group == "cross region":
                price_lines.append(f"{price_label} avg price incl tariff: {price_text}")
            else:
                price_lines.append(f"{price_label} avg seller price: {price_text}")

        if sum(values) > 0:
            ax.pie(
                values,
                labels=["within", "cross"],
                autopct="%1.0f%%",
                startangle=90,
                colors=[commodity_colors[commodity], commodity_colors[commodity]],
                wedgeprops=dict(edgecolor="white", linewidth=1.0),
                textprops=dict(fontsize=9),
            )
            ax.patches[1].set_alpha(cross_alpha)
            ax.patches[1].set_hatch("//")
        else:
            ax.text(0.5, 0.55, "no trades", transform=ax.transAxes, ha="center", va="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_title(f"{commodity} Sold: Within vs Cross", fontsize=11)
        ax.text(
            0.5,
            -0.08,
            "\n".join(price_lines),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    fig_redist, redist_axes = plt.subplots(
        1,
        2,
        figsize=(figsize[0], max(4.8, figsize[1] * 0.48)),
        sharey=True,
        constrained_layout=True,
    )

    tax_periods = sorted(redist["tax_period"].dropna().unique()) if not redist.empty else []
    planners = ["top", "bottom"]
    source_specs = [
        ("income_tax_funded_redistribution", "income tax", "#4c78a8"),
        ("travel_tax_funded_redistribution", "travel tax", "#f58518"),
        ("trade_tariff_funded_redistribution", "trade tariff", "#54a24b"),
    ]
    bar_width = 0.68
    period_x = np.arange(len(tax_periods))

    for ax_redist, planner in zip(redist_axes, planners):
        planner_df = redist[redist["planner_region"] == planner] if not redist.empty else redist
        bottoms = np.zeros(len(tax_periods), dtype=float)

        for source_col, label, color in source_specs:
            vals = []
            for tax_period in tax_periods:
                match = planner_df[planner_df["tax_period"] == tax_period]
                vals.append(float(match[source_col].sum()) if len(match) else 0.0)
            vals = np.asarray(vals, dtype=float)
            ax_redist.bar(
                period_x,
                vals,
                bottom=bottoms,
                width=bar_width,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            bottoms += vals

        if show_std and mode == "average" and "redistributed_std" in planner_df:
            total_err = []
            for tax_period in tax_periods:
                match = planner_df[planner_df["tax_period"] == tax_period]
                total_err.append(float(match["redistributed_std"].sum()) if len(match) else 0.0)
            ax_redist.errorbar(
                period_x,
                bottoms,
                yerr=np.asarray(total_err, dtype=float),
                fmt="none",
                ecolor="0.25",
                elinewidth=1.0,
                capsize=3,
                capthick=1.0,
                zorder=5,
            )

        ax_redist.set_title(f"{planner.capitalize()} Planner")
        ax_redist.set_xlabel("Tax period")
        ax_redist.set_xticks(period_x)
        ax_redist.set_xticklabels([str(int(k)) for k in tax_periods], fontsize=8)
        ax_redist.grid(True, axis="y", alpha=0.25)

    redist_axes[0].set_ylabel("Coin redistributed")
    handles, labels = redist_axes[0].get_legend_handles_labels()
    fig_redist.suptitle(f"Redistribution by Planner and Tax Period ({title_suffix})", fontsize=13, fontweight="bold")
    fig_redist.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        frameon=True,
        fontsize=9,
    )
    fig_redist.subplots_adjust(top=0.78)

    fig_cross, (ax_cross, ax_price, ax_share) = plt.subplots(
        3,
        1,
        figsize=(figsize[0], max(6.8, figsize[1] * 0.68)),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.45, 0.9, 0.8]},
    )
    buyer_regions = ["top", "bottom"]
    buyer_route_colors = {
        ("top", "within region"): "#9bd7f0",
        ("top", "cross region"): "#1f77b4",
        ("bottom", "within region"): "#ffd59e",
        ("bottom", "cross region"): "#ff7f0e",
    }
    buyer_route_offsets = {
        ("top", "within region"): -0.27,
        ("top", "cross region"): -0.09,
        ("bottom", "within region"): 0.09,
        ("bottom", "cross region"): 0.27,
    }
    route_label = {"within region": "within", "cross region": "cross"}
    buyer_periods = (
        sorted(set(plot_periods) | set(buyer_route["tax_period"].dropna().unique()))
        if not buyer_route.empty
        else plot_periods
    )
    buyer_x = np.arange(len(buyer_periods))
    buyer_width = 0.16
    totals_by_region_route = {}
    err_by_region_route = {}
    price_by_region_route = {}
    price_err_by_region_route = {}

    for buyer_region in buyer_regions:
        for route_group in route_groups:
            vals = []
            errs = []
            prices = []
            price_errs = []
            for tax_period in buyer_periods:
                match = buyer_route[
                    (buyer_route["tax_period"] == tax_period)
                    & (buyer_route["buyer_region"] == buyer_region)
                    & (buyer_route["route_group"] == route_group)
                ]
                vals.append(float(match["units"].sum()) if len(match) else 0.0)
                errs.append(float(match["units_std"].sum()) if len(match) else 0.0)
                prices.append(float(match["avg_price"].mean()) if len(match) else np.nan)
                price_errs.append(float(match["avg_price_std"].mean()) if len(match) else 0.0)
            vals = np.asarray(vals, dtype=float)
            errs = np.asarray(errs, dtype=float)
            prices = np.asarray(prices, dtype=float)
            price_errs = np.asarray(price_errs, dtype=float)
            totals_by_region_route[(buyer_region, route_group)] = vals
            err_by_region_route[(buyer_region, route_group)] = errs
            price_by_region_route[(buyer_region, route_group)] = prices
            price_err_by_region_route[(buyer_region, route_group)] = price_errs
            ax_cross.bar(
                buyer_x + buyer_route_offsets[(buyer_region, route_group)],
                vals,
                width=buyer_width,
                color=buyer_route_colors[(buyer_region, route_group)],
                alpha=0.88,
                edgecolor="0.25",
                linewidth=0.6,
                hatch="//" if route_group == "cross region" else "",
                label=f"{buyer_region} buys {route_label[route_group]}",
                yerr=errs if show_std and mode == "average" else None,
                error_kw=dict(ecolor="0.25", elinewidth=0.8, capsize=2, capthick=0.8),
            )
            ax_price.plot(
                buyer_x,
                prices,
                marker="o",
                linewidth=1.8,
                color=buyer_route_colors[(buyer_region, route_group)],
                linestyle="--" if route_group == "cross region" else "-",
                label=f"{buyer_region} {route_label[route_group]} price",
            )
            if show_std and mode == "average":
                ax_price.fill_between(
                    buyer_x,
                    prices - price_errs,
                    prices + price_errs,
                    color=buyer_route_colors[(buyer_region, route_group)],
                    alpha=0.12,
                    linewidth=0,
                )

    for buyer_region, color in [("top", "#1f77b4"), ("bottom", "#ff7f0e")]:
        within = totals_by_region_route.get((buyer_region, "within region"), np.zeros(len(buyer_periods)))
        cross = totals_by_region_route.get((buyer_region, "cross region"), np.zeros(len(buyer_periods)))
        denom = within + cross
        share = np.divide(cross, denom, out=np.full_like(cross, np.nan, dtype=float), where=denom > 0)
        ax_share.plot(
            buyer_x,
            share,
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{buyer_region} cross share",
        )

    ax_cross.set_title("Goods Bought by Buyer Region and Route")
    ax_cross.set_ylabel("Units bought")
    ax_cross.grid(True, axis="y", alpha=0.25)
    ax_cross.legend(loc="upper left", ncol=4, fontsize=8, frameon=True)

    ax_price.set_title("Average Price by Buyer Region and Route\ncross-region includes tariff")
    ax_price.set_ylabel("Avg trade price")
    ax_price.grid(True, axis="y", alpha=0.25)
    ax_price.legend(loc="upper left", ncol=4, fontsize=8, frameon=True)

    ax_share.set_title("Share of Each Region's Purchases That Are Cross-Region")
    ax_share.set_ylabel("Cross / total")
    share_values = []
    for line in ax_share.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        share_values.extend(y[np.isfinite(y)].tolist())
    if share_values:
        share_min = float(np.nanmin(share_values))
        share_max = float(np.nanmax(share_values))
        pad = max(0.03, 0.15 * (share_max - share_min))
        ax_share.set_ylim(max(0.0, share_min - pad), min(1.0, share_max + pad))
    else:
        ax_share.set_ylim(0, 1)
    ax_share.set_xlabel("Tax period")
    ax_share.set_xticks(buyer_x)
    ax_share.set_xticklabels([str(int(k)) for k in buyer_periods], fontsize=8)
    ax_share.grid(True, axis="y", alpha=0.25)
    ax_share.legend(loc="upper left", fontsize=8, frameon=True)

    fig_cross.suptitle(
        f"Within- vs Cross-Region Purchases by Buyer Region ({title_suffix})",
        fontsize=13,
        fontweight="bold",
    )

    fig.redistribution_figure = fig_redist
    fig.cross_region_buyer_figure = fig_cross
    fig.cross_region_buyer_table = buyer_route
    return fig, trade_units, price_summary, redist

def plot_regional_trade_and_planner_redistribution(
    runs,
    short_labels=None,
    mode="single",
    volume_field="price",
    period=100,
    rate_disc=0.05,
    figsize=None,
):
    """
    Plot trade volume distribution and planner redistribution in one figure.

    The trade panel separates within-region trades from cross-region trades.
    The redistribution panel shows total redistributed by each planner region,
    split into the income-tax-funded portion and any remaining redistribution.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def as_runs(obj):
        return obj if isinstance(obj, list) else [obj]

    def extract_episode_logs(run):
        if mode == "single":
            log = run.get("dense_log", None)
            if log is not None:
                return [log]
        dense_logs = run.get("dense_logs", None)
        if isinstance(dense_logs, dict):
            return [v for v in dense_logs.values() if isinstance(v, dict)]
        if isinstance(dense_logs, list):
            return [v for v in dense_logs if isinstance(v, dict)]
        return [run["dense_log"]] if run.get("dense_log", None) is not None else []

    runs = as_runs(runs)
    run_names = [run.get("name", f"Run {i + 1}") for i, run in enumerate(runs)]
    if short_labels is None:
        labels = {name: f"Run {i + 1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        labels = {name: short_labels[i] for i, name in enumerate(run_names)}
    else:
        labels = short_labels

    trade_rows = []
    redistrib_rows = []

    for run, name in zip(runs, run_names):
        episode_logs = extract_episode_logs(run)
        if not episode_logs:
            continue

        per_episode_trade = []
        per_episode_redistrib = []

        for rollout_id, log in enumerate(episode_logs):
            trade_df = extract_trade_region_distribution(log, volume_field=volume_field)
            if not trade_df.empty:
                grouped_trade = (
                    trade_df
                    .groupby(["route_type", "route"], as_index=False)
                    .agg(volume=("volume", "sum"), n_trades=("volume", "size"))
                )
                grouped_trade["rollout_id"] = rollout_id
                per_episode_trade.append(grouped_trade)

            redistrib_df = extract_planner_redistribution_table(
                log,
                period=period,
                rate_disc=rate_disc,
            )
            if not redistrib_df.empty:
                redistrib_df = redistrib_df.copy()
                redistrib_df["rollout_id"] = rollout_id
                per_episode_redistrib.append(redistrib_df)

        label = labels.get(name, name)

        if per_episode_trade:
            df = pd.concat(per_episode_trade, ignore_index=True)
            if mode == "average":
                df = (
                    df
                    .groupby(["route_type", "route"], as_index=False)
                    .agg(volume=("volume", "mean"), n_trades=("n_trades", "mean"))
                )
            else:
                df = df[df["rollout_id"] == df["rollout_id"].min()]
            df["run"] = name
            df["label"] = label
            trade_rows.append(df)

        if per_episode_redistrib:
            df = pd.concat(per_episode_redistrib, ignore_index=True)
            if mode == "average":
                df = (
                    df
                    .groupby("planner_region", as_index=False)
                    .agg(
                        income=("income", "mean"),
                        income_tax_collected=("income_tax_collected", "mean"),
                        tariff_revenue=("tariff_revenue", "mean"),
                        redistributed=("redistributed", "mean"),
                        income_tax_funded_redistribution=("income_tax_funded_redistribution", "mean"),
                        non_income_tax_redistribution=("non_income_tax_redistribution", "mean"),
                        n_agents=("n_agents", "mean"),
                    )
                )
            else:
                df = df[df["rollout_id"] == df["rollout_id"].min()]
            df["run"] = name
            df["label"] = label
            redistrib_rows.append(df)

    trade_out = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    redistrib_out = (
        pd.concat(redistrib_rows, ignore_index=True)
        if redistrib_rows
        else pd.DataFrame()
    )

    n_runs = max(1, len(run_names))
    if figsize is None:
        figsize = (max(10, 3.2 * n_runs), 9)

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    ax_trade, ax_redist = axes

    route_order = ["within top", "within bottom", "top to bottom", "bottom to top"]
    colors_trade = {
        "within top": "#1f77b4",
        "within bottom": "#17becf",
        "top to bottom": "#d62728",
        "bottom to top": "#ff7f0e",
    }

    x = np.arange(len(run_names))
    bottom = np.zeros(len(run_names), dtype=float)
    for route in route_order:
        vals = []
        for name in run_names:
            label = labels.get(name, name)
            if trade_out.empty:
                vals.append(0.0)
            else:
                vals.append(float(trade_out[
                    (trade_out["label"] == label) & (trade_out["route"] == route)
                ]["volume"].sum()))
        vals = np.asarray(vals, dtype=float)
        ax_trade.bar(
            x,
            vals,
            bottom=bottom,
            label=route,
            color=colors_trade.get(route),
            edgecolor="white",
            linewidth=0.8,
        )
        bottom += vals

    ax_trade.set_title("Trade Volume by Region Route")
    ax_trade.set_ylabel(f"Trade volume ({volume_field})")
    ax_trade.set_xticks(x)
    ax_trade.set_xticklabels([labels.get(name, name) for name in run_names])
    ax_trade.legend(ncol=2, frameon=True)
    ax_trade.grid(True, axis="y", alpha=0.3)

    planner_order = ["top", "bottom"]
    bar_labels = []
    income_tax_vals = []
    other_vals = []
    for name in run_names:
        label = labels.get(name, name)
        for planner_region in planner_order:
            bar_labels.append(f"{label}\n{planner_region}")
            dfr = redistrib_out[
                (redistrib_out["label"] == label)
                & (redistrib_out["planner_region"] == planner_region)
            ]
            income_tax_vals.append(
                float(dfr["income_tax_funded_redistribution"].sum())
                if not dfr.empty else 0.0
            )
            other_vals.append(
                float(dfr["non_income_tax_redistribution"].sum())
                if not dfr.empty else 0.0
            )

    x2 = np.arange(len(bar_labels))
    income_tax_vals = np.asarray(income_tax_vals, dtype=float)
    other_vals = np.asarray(other_vals, dtype=float)
    ax_redist.bar(
        x2,
        income_tax_vals,
        label="from income tax",
        color="#2ca02c",
        edgecolor="white",
        linewidth=0.8,
    )
    ax_redist.bar(
        x2,
        other_vals,
        bottom=income_tax_vals,
        label="other redistribution",
        color="#9467bd",
        edgecolor="white",
        linewidth=0.8,
    )
    ax_redist.set_title("Planner Redistribution")
    ax_redist.set_ylabel("Coin redistributed")
    ax_redist.set_xticks(x2)
    ax_redist.set_xticklabels(bar_labels)
    ax_redist.legend(frameon=True)
    ax_redist.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig, trade_out, redistrib_out

def extract_agent_labor_allocation_over_time(
    dense_log,
    agent_ids=None,
    labor_costs=None,
    include_other=True,
):
    import numpy as np
    import pandas as pd

    states = dense_log["states"]

    if agent_ids is None:
        agent_ids = sorted(int(k) for k in states[0].keys() if str(k).isdigit())
    agent_ids = [int(a) for a in agent_ids]

    costs = {
        "Move": 0.21,
        "Gather": 0.21,
        "Build": 2.1,
        "Travel": 4.0,
    }
    if labor_costs is not None:
        costs.update(labor_costs)

    activities = ["Move", "Gather", "Build", "Trade/Order", "Travel"]
    if include_other:
        activities.append("Other")

    n_steps = len(states) - 1
    rows = []
    cumulative_rows = []

    def _events_at(key, t):
        seq = dense_log.get(key, [])
        if isinstance(seq, list) and any(isinstance(event, dict) and "t" in event for event in seq):
            return [
                event
                for event in seq
                if isinstance(event, dict) and int(event.get("t", -1)) == int(t)
            ]
        if t >= len(seq):
            return []

        events = seq[t]

        if isinstance(events, dict):
            for event_key in ["events", "gathers", "builds", "trades", "travels"]:
                if event_key in events:
                    return events.get(event_key, [])
            return []

        return events if isinstance(events, list) else []

    for aid in agent_ids:
        aid_key = str(aid)

        labor = np.array(
            [
                float(state[aid_key].get("endogenous", {}).get("Labor", 0.0))
                for state in states
            ],
            dtype=float,
        )
        labor_delta = np.diff(labor)

        for timestep, value in enumerate(labor):
            cumulative_rows.append(
                {
                    "agent": aid,
                    "timestep": timestep,
                    "cumulative_labor": float(value),
                }
            )

        for t in range(n_steps):
            values = {activity: 0.0 for activity in activities}
            known = 0.0

            loc0 = np.array(states[t][aid_key]["loc"], dtype=float)
            loc1 = np.array(states[t + 1][aid_key]["loc"], dtype=float)
            moved = np.abs(loc1 - loc0).sum() > 0

            travel_events = _events_at("CrossWaterTravel", t)
            did_travel = any(
                int(e.get("agent", -1)) == aid
                for e in travel_events
                if isinstance(e, dict)
            )

            if did_travel:
                values["Travel"] += costs["Travel"]
                known += costs["Travel"]
            elif moved:
                values["Move"] += costs["Move"]
                known += costs["Move"]

            gather_count = sum(
                1
                for event in _events_at("Gather", t)
                if isinstance(event, dict) and int(event.get("agent", -1)) == aid
            )
            if gather_count:
                value = gather_count * costs["Gather"]
                values["Gather"] += value
                known += value

            build_count = sum(
                1
                for event in _events_at("Build", t)
                if isinstance(event, dict) and int(event.get("builder", -1)) == aid
            )
            if build_count:
                value = build_count * costs["Build"]
                values["Build"] += value
                known += value

            residual = max(0.0, float(labor_delta[t]) - known)
            values["Trade/Order"] += residual

            if include_other:
                overshoot = max(0.0, known - float(labor_delta[t]))
                if overshoot > 1e-9:
                    values["Other"] += overshoot

            for activity, value in values.items():
                rows.append(
                    {
                        "agent": aid,
                        "timestep": t + 1,
                        "activity": activity,
                        "labor": float(value),
                    }
                )

    allocation_df = pd.DataFrame(rows)
    cumulative_df = pd.DataFrame(cumulative_rows)

    return allocation_df, cumulative_df

def extract_agent_labor_allocation(
    dense_log,
    agent_ids=None,
    labor_costs=None,
    include_other=True,
):
    import numpy as np
    import pandas as pd

    states = dense_log["states"]

    if agent_ids is None:
        agent_ids = sorted(int(k) for k in states[0].keys() if str(k).isdigit())
    agent_ids = [int(a) for a in agent_ids]

    costs = {
        "Move": 0.21,
        "Gather": 0.21,
        "Build": 2.1,
        "Travel": 4.0,
    }
    if labor_costs is not None:
        costs.update(labor_costs)

    n_actions = max(0, len(states) - 1)

    activities = ["Move", "Gather", "Build", "Trade/Order", "Travel"]
    if include_other:
        activities.append("Other")

    rows = []
    cumulative_rows = []

    def _events_at(key, t):
        seq = dense_log.get(key, [])
        if isinstance(seq, list) and any(isinstance(event, dict) and "t" in event for event in seq):
            return [
                event
                for event in seq
                if isinstance(event, dict) and int(event.get("t", -1)) == int(t)
            ]
        if t >= len(seq):
            return []

        events = seq[t]

        if isinstance(events, dict):
            for event_key in ["events", "gathers", "builds", "trades", "travels"]:
                if event_key in events:
                    return events.get(event_key, [])
            return []

        return events if isinstance(events, list) else []

    for aid in agent_ids:
        aid_key = str(aid)

        labor = np.array(
            [
                float(state[aid_key].get("endogenous", {}).get("Labor", 0.0))
                for state in states
            ],
            dtype=float,
        )
        labor_delta = np.diff(labor)

        totals = {activity: 0.0 for activity in activities}

        for t in range(n_actions):
            known = 0.0

            loc0 = np.array(states[t][aid_key]["loc"], dtype=float)
            loc1 = np.array(states[t + 1][aid_key]["loc"], dtype=float)
            moved = np.abs(loc1 - loc0).sum() > 0

            travel_events = _events_at("CrossWaterTravel", t)
            did_travel = any(
                int(e.get("agent", -1)) == aid
                for e in travel_events
                if isinstance(e, dict)
            )

            if did_travel:
                totals["Travel"] += costs["Travel"]
                known += costs["Travel"]
            elif moved:
                totals["Move"] += costs["Move"]
                known += costs["Move"]

            gather_count = sum(
                1
                for event in _events_at("Gather", t)
                if isinstance(event, dict) and int(event.get("agent", -1)) == aid
            )
            if gather_count:
                value = gather_count * costs["Gather"]
                totals["Gather"] += value
                known += value

            build_count = sum(
                1
                for event in _events_at("Build", t)
                if isinstance(event, dict) and int(event.get("builder", -1)) == aid
            )
            if build_count:
                value = build_count * costs["Build"]
                totals["Build"] += value
                known += value

            residual = max(0.0, float(labor_delta[t]) - known)
            if residual:
                totals["Trade/Order"] += residual

            if include_other:
                overshoot = max(0.0, known - float(labor_delta[t]))
                if overshoot > 1e-9:
                    totals["Other"] += overshoot

        for activity, value in totals.items():
            rows.append(
                {
                    "agent": aid,
                    "activity": activity,
                    "labor": float(value),
                    "final_cumulative_labor": float(labor[-1]),
                }
            )

        for timestep, value in enumerate(labor):
            cumulative_rows.append(
                {
                    "agent": aid,
                    "timestep": timestep,
                    "cumulative_labor": float(value),
                }
            )

    allocation_df = pd.DataFrame(rows)
    cumulative_df = pd.DataFrame(cumulative_rows)

    return allocation_df, cumulative_df

def plot_agent_labor_allocation(
    dense_logs,
    mode="single",
    agent_ids=None,
    n_cols=4,
    bin_size=25,
    labor_costs=None,
    activity_colors=None,
    errorbar="std",
    figsize=(18, 8),
    title=None,
):
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def extract_episode_logs(obj):
        if obj is None:
            return []

        if isinstance(obj, dict) and "states" in obj:
            return [obj]

        if isinstance(obj, (list, tuple)):
            eps = []
            for value in obj:
                eps.extend(extract_episode_logs(value))
            return eps

        if isinstance(obj, dict):
            eps = []
            for key in ["final", "episodes", "dense_logs", "logs", "data"]:
                if key in obj:
                    eps.extend(extract_episode_logs(obj[key]))

            if eps:
                return eps

            for value in obj.values():
                eps.extend(extract_episode_logs(value))

            return eps

        return []

    episode_logs = extract_episode_logs(dense_logs)

    if len(episode_logs) == 0:
        raise ValueError("No dense logs with states were found.")

    if mode == "single":
        episode_logs = episode_logs[:1]
    elif mode != "average":
        raise ValueError("mode must be 'single' or 'average'")

    if agent_ids is None:
        agent_ids = sorted(
            int(k) for k in episode_logs[0]["states"][0].keys() if str(k).isdigit()
        )
    agent_ids = [int(a) for a in agent_ids]

    if activity_colors is None:
        activity_colors = {
            "Move": "#1f77b4",
            "Gather": "#2ca02c",
            "Build": "#ff7f0e",
            "Trade/Order": "#9467bd",
            "Travel": "#d62728",
            "Other": "#7f7f7f",
        }

    activities = list(activity_colors.keys())

    allocation_frames = []
    cumulative_frames = []

    for rollout_id, log in enumerate(episode_logs):
        alloc, cumul = extract_agent_labor_allocation_over_time(
            log,
            agent_ids=agent_ids,
            labor_costs=labor_costs,
        )

        alloc["rollout_id"] = rollout_id
        cumul["rollout_id"] = rollout_id

        allocation_frames.append(alloc)
        cumulative_frames.append(cumul)

    raw_allocation_df = pd.concat(allocation_frames, ignore_index=True)
    raw_cumulative_df = pd.concat(cumulative_frames, ignore_index=True)

    raw_allocation_df["bin"] = (
        ((raw_allocation_df["timestep"] - 1) // bin_size) * bin_size
    ) + (bin_size / 2)

    n_agents = len(agent_ids)
    n_cols = min(n_cols, n_agents)
    n_rows = int(math.ceil(n_agents / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    summary_rows = []
    bar_axes = []

    for panel_idx, aid in enumerate(agent_ids):
        ax = axes[panel_idx // n_cols, panel_idx % n_cols]
        bar_ax = ax.twinx()
        bar_axes.append(bar_ax)

        agent_cumul = raw_cumulative_df[raw_cumulative_df["agent"] == aid]
        agent_alloc = raw_allocation_df[raw_allocation_df["agent"] == aid]

        if mode == "single":
            curve = agent_cumul.sort_values("timestep")

            x = curve["timestep"].to_numpy(dtype=float)
            y = curve["cumulative_labor"].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                color="black",
                linewidth=2.0,
                label="Cumulative labor",
                zorder=5,
            )

            binned = (
                agent_alloc
                .groupby(["bin", "activity"], as_index=False)["labor"]
                .sum()
            )

            bin_values = sorted(binned["bin"].unique())
            bottom = np.zeros(len(bin_values), dtype=float)

            for activity in activities:
                vals = (
                    binned[binned["activity"] == activity]
                    .set_index("bin")
                    .reindex(bin_values)["labor"]
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )

                bar_ax.bar(
                    bin_values,
                    vals,
                    bottom=bottom,
                    width=bin_size * 0.85,
                    color=activity_colors[activity],
                    alpha=0.55,
                    linewidth=0,
                    zorder=1,
                )

                bottom += vals

                for b, value in zip(bin_values, vals):
                    summary_rows.append(
                        {
                            "agent": aid,
                            "bin": float(b),
                            "activity": activity,
                            "labor_mean": float(value),
                            "labor_error": np.nan,
                            "n_dense_logs": 1,
                        }
                    )

        else:
            min_len = int(
                agent_cumul.groupby("rollout_id")["timestep"].max().min() + 1
            )

            curves = []
            for _, group in agent_cumul.groupby("rollout_id"):
                group = group.sort_values("timestep").head(min_len)
                curves.append(group["cumulative_labor"].to_numpy(dtype=float))

            curves = np.vstack(curves)

            x = np.arange(curves.shape[1], dtype=float)
            y = np.mean(curves, axis=0)
            y_sd = np.std(curves, axis=0)

            ax.plot(
                x,
                y,
                color="black",
                linewidth=2.0,
                label="Mean cumulative labor",
                zorder=5,
            )
            ax.fill_between(
                x,
                y - y_sd,
                y + y_sd,
                color="black",
                alpha=0.12,
                linewidth=0,
                zorder=4,
            )

            binned_by_rollout = (
                agent_alloc
                .groupby(["rollout_id", "bin", "activity"], as_index=False)["labor"]
                .sum()
            )

            grouped = (
                binned_by_rollout
                .groupby(["bin", "activity"])["labor"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )

            grouped["std"] = grouped["std"].fillna(0.0)

            if errorbar == "std":
                grouped["err"] = grouped["std"]
            elif errorbar == "sem":
                grouped["err"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
            elif errorbar is None:
                grouped["err"] = 0.0
            else:
                raise ValueError("errorbar must be None, 'std', or 'sem'")

            bin_values = sorted(grouped["bin"].unique())
            bottom = np.zeros(len(bin_values), dtype=float)

            for activity in activities:
                activity_df = (
                    grouped[grouped["activity"] == activity]
                    .set_index("bin")
                    .reindex(bin_values)
                    .fillna(0.0)
                )

                vals = activity_df["mean"].to_numpy(dtype=float)
                errs = activity_df["err"].to_numpy(dtype=float)

                bar_ax.bar(
                    bin_values,
                    vals,
                    bottom=bottom,
                    width=bin_size * 0.85,
                    color=activity_colors[activity],
                    alpha=0.55,
                    linewidth=0,
                    zorder=1,
                )

                if errorbar is not None:
                    bar_ax.errorbar(
                        bin_values,
                        bottom + vals,
                        yerr=errs,
                        fmt="none",
                        ecolor=activity_colors[activity],
                        elinewidth=0.8,
                        capsize=1.5,
                        alpha=0.8,
                        zorder=2,
                    )

                bottom += vals

                for b, value, err in zip(bin_values, vals, errs):
                    summary_rows.append(
                        {
                            "agent": aid,
                            "bin": float(b),
                            "activity": activity,
                            "labor_mean": float(value),
                            "labor_error": float(err),
                            "n_dense_logs": len(episode_logs),
                        }
                    )

        ax.set_title(f"Agent {aid}")
        ax.grid(True, axis="y", alpha=0.28)

        ax.set_ylabel("Cumulative labor" if panel_idx % n_cols == 0 else "")
        bar_ax.set_ylabel("Activity labor" if panel_idx % n_cols == n_cols - 1 else "")

        if panel_idx // n_cols == n_rows - 1:
            ax.set_xlabel("Timestep")

        ax.set_zorder(2)
        bar_ax.set_zorder(1)
        ax.patch.set_visible(False)

    for panel_idx in range(n_agents, n_rows * n_cols):
        axes[panel_idx // n_cols, panel_idx % n_cols].axis("off")

    max_bar_y = 0.0
    for bar_ax in bar_axes:
        _, ymax = bar_ax.get_ylim()
        max_bar_y = max(max_bar_y, ymax)

    for bar_ax in bar_axes:
        bar_ax.set_ylim(0, max_bar_y)

    if title is None:
        if mode == "single":
            title = "Agent Labor Allocation Over Time (Single Dense Log)"
        else:
            marker = (
                "SD"
                if errorbar == "std"
                else "SEM"
                if errorbar == "sem"
                else "No error bars"
            )
            title = f"Agent Labor Allocation Over Time (Mean Across Dense Logs, {marker})"

    fig.suptitle(title, fontsize=14, y=0.98)

    legend_handles = [
        Patch(facecolor=color, label=activity)
        for activity, color in activity_colors.items()
        if activity in activities
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(legend_handles), 6),
        frameon=True,
        fontsize=10,
    )

    fig.subplots_adjust(
        bottom=0.12,
        top=0.91,
        hspace=0.35,
        wspace=0.35,
    )

    summary_df = pd.DataFrame(summary_rows)

    return fig, summary_df, raw_allocation_df, raw_cumulative_df


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

        # single episode / rollout
        if isinstance(obj, dict) and "states" in obj:
            return [obj]

        # list/tuple of episodes, or nested containers of episodes
        if isinstance(obj, (list, tuple)):
            eps = []
            for v in obj:
                eps.extend(extract_episode_logs(v))
            return eps

        # dict keyed by rollout id, or wrappers such as {"final": {...}}
        if isinstance(obj, dict):
            eps = []
            for key in ["final", "episodes", "dense_logs", "logs", "data"]:
                if key in obj:
                    eps.extend(extract_episode_logs(obj[key]))

            if eps:
                return eps

            for v in obj.values():
                eps.extend(extract_episode_logs(v))
            return eps

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


def compare_gini_over_tax_periods(
    runs,
    short_labels=None,
    show_legend=True,
    period=100,
    errorbar="std",      # None, "std", or "sem"
    max_periods=None,
    figsize=(8, 4.5),
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

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

        if isinstance(obj, dict) and "states" in obj:
            return [obj]

        if isinstance(obj, (list, tuple)):
            eps = []
            for v in obj:
                eps.extend(extract_episode_logs(v))
            return eps

        if isinstance(obj, dict):
            eps = []
            for key in ["final", "episodes", "dense_logs", "logs", "data"]:
                if key in obj:
                    eps.extend(extract_episode_logs(obj[key]))

            if eps:
                return eps

            for v in obj.values():
                eps.extend(extract_episode_logs(v))
            return eps

        return []

    def gini_from_state(state):
        coins = []

        for k, s in state.items():
            if str(k).isdigit():
                coins.append(
                    float(s["inventory"].get("Coin", 0.0))
                    + float(s["escrow"].get("Coin", 0.0))
                )

        return gini(np.array(coins, dtype=float))

    run_names = [run["name"] for run in runs]

    if short_labels is None:
        short_labels = {name: f"E{i+1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    colors_list = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]

    rows = []

    for run in runs:
        name = run["name"]
        dense_logs_obj = run.get("dense_logs", None)
        if dense_logs_obj is None:
            dense_logs_obj = run.get("dense_log", None)

        eps = extract_episode_logs(dense_logs_obj)

        for rollout_id, ep in enumerate(eps):
            states = ep.get("states", [])
            tax_days = list(range(period - 1, len(states), period))
            if max_periods is not None:
                tax_days = tax_days[:max_periods]

            for tax_day_number, timestep in enumerate(tax_days, start=1):
                try:
                    gini_value = gini_from_state(states[timestep])
                except Exception:
                    gini_value = np.nan

                rows.append({
                    "run": name,
                    "label": short_labels[name],
                    "rollout_id": rollout_id,
                    "tax_day_number": tax_day_number,
                    "timestep": timestep,
                    "gini": gini_value,
                })

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError("No dense logs with states were found in the supplied runs.")

    raw_df = raw_df[np.isfinite(raw_df["gini"])]
    if raw_df.empty:
        raise ValueError("Dense logs were found, but no finite Gini values could be computed.")

    grouped = raw_df.groupby(["run", "label", "tax_day_number"], sort=False)
    out_df = grouped.agg(
        timestep=("timestep", "median"),
        gini_mean=("gini", "mean"),
        gini_std=("gini", "std"),
        n_dense_logs=("gini", "count"),
    ).reset_index()

    out_df["gini_std"] = out_df["gini_std"].fillna(0.0)
    out_df["gini_sem"] = out_df["gini_std"] / np.sqrt(out_df["n_dense_logs"])

    if errorbar == "std":
        out_df["gini_error"] = out_df["gini_std"]
    elif errorbar == "sem":
        out_df["gini_error"] = out_df["gini_sem"]
    elif errorbar is None:
        out_df["gini_error"] = np.nan
    else:
        raise ValueError("errorbar must be None, 'std', or 'sem'")

    fig, ax = plt.subplots(figsize=figsize)

    for i, name in enumerate(run_names):
        run_df = out_df[out_df["run"] == name].sort_values("tax_day_number")
        if run_df.empty:
            continue

        color = colors_list[i % len(colors_list)]
        label = short_labels[name]

        x = run_df["tax_day_number"].to_numpy()
        mean = run_df["gini_mean"].to_numpy()
        err = run_df["gini_error"].to_numpy()

        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=color,
            label=label,
        )

        if errorbar is not None:
            ax.fill_between(
                x,
                mean - err,
                mean + err,
                color=color,
                alpha=0.18,
                linewidth=0,
            )

    title = "Average Gini Coefficient by Tax Period Across Dense Logs"
    if errorbar == "std":
        title += " ± SD"
    elif errorbar == "sem":
        title += " ± SEM"

    ax.set_title(title)
    ax.set_xlabel("Tax period")
    ax.set_ylabel("Gini")
    ax.set_xticks(sorted(out_df["tax_day_number"].unique()))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, len(run_names)),
            frameon=True,
        )
        fig.subplots_adjust(bottom=0.25)
    else:
        fig.tight_layout()

    return fig, out_df, raw_df


def load_dense_logs_from_result_folder(result_dir):
    import pickle
    from pathlib import Path

    result_dir = Path(result_dir)
    dense_logs_path = result_dir / "dense_logs_final.pkl"

    if not dense_logs_path.exists():
        raise FileNotFoundError(f"No dense_logs_final.pkl found in {result_dir}")

    with open(dense_logs_path, "rb") as f:
        return pickle.load(f)


def get_dense_log_from_result_folder(result_dir, episode_key=0):
    dense_logs = load_dense_logs_from_result_folder(result_dir)

    if isinstance(dense_logs, dict):
        if episode_key in dense_logs:
            return dense_logs[episode_key], dense_logs

        episode_key_str = str(episode_key)
        if episode_key_str in dense_logs:
            return dense_logs[episode_key_str], dense_logs

        for v in dense_logs.values():
            if isinstance(v, dict) and "states" in v:
                return v, dense_logs

    if isinstance(dense_logs, list):
        if len(dense_logs) == 0:
            raise ValueError(f"No dense logs found in {result_dir}")
        return dense_logs[int(episode_key)], dense_logs

    if isinstance(dense_logs, dict) and "states" in dense_logs:
        return dense_logs, dense_logs

    raise ValueError(f"Could not find an episode log with states in {result_dir}")


def _dense_logs_from_source(source):
    """Resolve a result folder path, dense log, dense_logs object, or run dict."""
    import os

    if isinstance(source, (str, os.PathLike)):
        return load_dense_logs_from_result_folder(source)
    return source


def _select_dense_log_from_source(source, episode_key=0):
    dense_logs = _dense_logs_from_source(source)
    log_items = _dense_log_items(dense_logs)
    if not log_items:
        raise ValueError("No dense logs found. Pass a result folder, dense log, dense_logs, or a run dict.")

    for key, dense_log in log_items:
        if key == episode_key or str(key) == str(episode_key):
            return dense_log, dense_logs
    return log_items[int(episode_key)][1], dense_logs


def breakdown_all_agents_from_result_folder(result_dir, episode_key=0, remap_key="build_payment", n_cols=4):
    dense_log, dense_logs = _select_dense_log_from_source(result_dir, episode_key=episode_key)
    breakdown = breakdown_all_agents(dense_log, remap_key=remap_key, n_cols=n_cols)
    return breakdown, dense_log, dense_logs


def breakdown_all_agents_average_from_result_folder(
    result_dir,
    remap_key="build_payment",
    group_by_skill=False,
    trade_count_bin_size=25,
    metrics=None,
    figsize_metrics=(18, 9),
    figsize_trade=(16, 9),
    figsize_skill=(12, 5),
):
    """
    Average-style all-agent breakdown across every dense log in a result folder.

    This intentionally leaves out map and movement panels, where averaging
    trajectories would be misleading. Metric panels use one dense log as one
    sample per agent. Set ``group_by_skill=True`` to pool agents by skill level
    in the metric and trade-value panels.
    """
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dense_logs = _dense_logs_from_source(result_dir)
    log_items = _dense_log_items(dense_logs)
    if not log_items:
        raise ValueError(f"No dense logs found in {result_dir}")

    first_log = log_items[0][1]
    agent_ids = numeric_agent_ids_from_states(first_log["states"][0])
    agent_colors = _make_agent_colors(agent_ids)

    if remap_key is None:
        ordered_agents = agent_ids[:]
    else:
        key_vals = np.array([
            first_log["states"][0][str(aid)].get(remap_key, np.nan)
            for aid in agent_ids
        ], dtype=float)
        ordered_agents = [agent_ids[i] for i in np.argsort(key_vals).tolist()]

    build_payment = {
        aid: float(first_log["states"][0][str(aid)].get("build_payment", np.nan))
        for aid in agent_ids
    }
    finite_skills = sorted({v for v in build_payment.values() if np.isfinite(v)})
    skill_rank = {v: i + 1 for i, v in enumerate(finite_skills)}

    def skill_label(aid):
        build = build_payment.get(aid, np.nan)
        rank = skill_rank.get(build, None)
        if rank is None:
            return "skill ?"
        return f"skill {rank}/{len(finite_skills)}"

    def agent_label(aid):
        build = build_payment.get(aid, np.nan)
        build_text = "?" if not np.isfinite(build) else f"{build:.0f}"
        return f"A{aid}\n{skill_label(aid)}\nbuild {build_text}"

    skill_order = sorted(
        {skill_label(aid) for aid in agent_ids},
        key=lambda x: (999 if x == "skill ?" else int(x.split()[1].split("/")[0])),
    )

    metric_rows = []
    trade_rows = []
    rollout_lengths = {}

    for rollout_id, log in log_items:
        states = log.get("states", [])
        if not states:
            continue
        rollout_lengths[rollout_id] = len(states)

        builds_by_agent = {
            aid: {"build_income": 0.0, "build_count": 0}
            for aid in agent_ids
        }
        for builds in log.get("Build", []):
            builds_ = builds.get("builds", []) if isinstance(builds, dict) else builds
            if not isinstance(builds_, list):
                continue
            for build in builds_:
                if not isinstance(build, dict) or "builder" not in build:
                    continue
                aid = int(build["builder"])
                if aid not in builds_by_agent:
                    continue
                builds_by_agent[aid]["build_income"] += float(build.get("income", 0.0))
                builds_by_agent[aid]["build_count"] += 1

        trade_value = {
            (aid, commodity, side): 0.0
            for aid in agent_ids
            for commodity in ["Wood", "Stone"]
            for side in ["Sell", "Buy"]
        }
        trade_count = {
            (aid, commodity, side): 0
            for aid in agent_ids
            for commodity in ["Wood", "Stone"]
            for side in ["Sell", "Buy"]
        }
        for t, trades in enumerate(log.get("Trade", [])):
            trades_ = trades.get("trades", []) if isinstance(trades, dict) else trades
            if not isinstance(trades_, list):
                continue
            for trade in trades_:
                if not isinstance(trade, dict):
                    continue
                commodity = trade.get("commodity")
                if commodity not in ["Wood", "Stone"]:
                    continue
                seller = trade.get("seller")
                buyer = trade.get("buyer")
                if seller is not None:
                    aid = int(seller)
                    if aid in agent_ids:
                        trade_value[(aid, commodity, "Sell")] += float(
                            trade.get("income", trade.get("price", 0.0))
                        )
                        trade_count[(aid, commodity, "Sell")] += 1
                        trade_rows.append({
                            "rollout_id": rollout_id,
                            "agent": aid,
                            "skill": skill_label(aid),
                            "build_payment": build_payment.get(aid, np.nan),
                            "commodity": commodity,
                            "side": "Sell",
                            "value": float(trade.get("income", trade.get("price", 0.0))),
                            "count": 1,
                            "timestep": t,
                        })
                if buyer is not None:
                    aid = int(buyer)
                    if aid in agent_ids:
                        trade_value[(aid, commodity, "Buy")] += float(
                            trade.get("cost", trade.get("price", 0.0))
                        )
                        trade_count[(aid, commodity, "Buy")] += 1
                        trade_rows.append({
                            "rollout_id": rollout_id,
                            "agent": aid,
                            "skill": skill_label(aid),
                            "build_payment": build_payment.get(aid, np.nan),
                            "commodity": commodity,
                            "side": "Buy",
                            "value": float(trade.get("cost", trade.get("price", 0.0))),
                            "count": 1,
                            "timestep": t,
                        })

        for aid in agent_ids:
            agent_states = [state[str(aid)] for state in states if str(aid) in state]
            if not agent_states:
                continue

            def resource_series(resource):
                return np.array([
                    float(s.get("inventory", {}).get(resource, 0.0))
                    + float(s.get("escrow", {}).get(resource, 0.0))
                    for s in agent_states
                ], dtype=float)

            coin = resource_series("Coin")
            wood = resource_series("Wood")
            stone = resource_series("Stone")
            labor = np.array([
                float(s.get("endogenous", {}).get("Labor", np.nan))
                for s in agent_states
            ], dtype=float)
            utility = np.array([
                float(s.get("utility", np.nan))
                for s in agent_states
            ], dtype=float)
            sell_total = sum(trade_value[(aid, c, "Sell")] for c in ["Wood", "Stone"])
            buy_total = sum(trade_value[(aid, c, "Buy")] for c in ["Wood", "Stone"])

            metric_rows.append({
                "rollout_id": rollout_id,
                "agent": aid,
                "skill": skill_label(aid),
                "build_payment": build_payment.get(aid, np.nan),
                "final_coin": float(coin[-1]) if len(coin) else np.nan,
                "mean_coin": float(np.nanmean(coin)) if len(coin) else np.nan,
                "final_wood": float(wood[-1]) if len(wood) else np.nan,
                "final_stone": float(stone[-1]) if len(stone) else np.nan,
                "final_labor": float(labor[-1]) if len(labor) else np.nan,
                "mean_labor": float(np.nanmean(labor)) if len(labor) else np.nan,
                "mean_utility": float(np.nanmean(utility)) if np.any(np.isfinite(utility)) else np.nan,
                "final_utility": float(utility[-1]) if len(utility) and np.isfinite(utility[-1]) else np.nan,
                "build_income": builds_by_agent[aid]["build_income"],
                "build_count": builds_by_agent[aid]["build_count"],
                "sell_income": sell_total,
                "buy_cost": buy_total,
                "net_market": sell_total - buy_total,
                "wood_sell_value": trade_value[(aid, "Wood", "Sell")],
                "wood_buy_value": trade_value[(aid, "Wood", "Buy")],
                "stone_sell_value": trade_value[(aid, "Stone", "Sell")],
                "stone_buy_value": trade_value[(aid, "Stone", "Buy")],
                "wood_sell_count": trade_count[(aid, "Wood", "Sell")],
                "wood_buy_count": trade_count[(aid, "Wood", "Buy")],
                "stone_sell_count": trade_count[(aid, "Stone", "Sell")],
                "stone_buy_count": trade_count[(aid, "Stone", "Buy")],
            })

    metric_df = pd.DataFrame(metric_rows)
    trade_event_df = pd.DataFrame(trade_rows)
    if metric_df.empty:
        raise ValueError("Dense logs were found, but no agent metric rows could be constructed.")

    if metrics is None:
        metrics = [
            "build_count",
            "mean_coin",
            "mean_labor",
            "mean_utility",
            "build_income",
            "net_market",
        ]
    metrics = [m for m in metrics if m in metric_df.columns and metric_df[m].notna().any()]

    n_cols = min(3, max(1, len(metrics)))
    n_rows = int(math.ceil(len(metrics) / n_cols))
    fig_metrics, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize_metrics,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()

    if group_by_skill:
        metric_groups = skill_order
        metric_labels = metric_groups
        metric_data_for = lambda group, metric: metric_df.loc[metric_df["skill"] == group, metric]
        metric_colors = {
            group: plt.get_cmap("tab10")(i % 10)
            for i, group in enumerate(metric_groups)
        }
    else:
        metric_groups = ordered_agents
        metric_labels = [agent_label(aid) for aid in ordered_agents]
        metric_data_for = lambda aid, metric: metric_df.loc[metric_df["agent"] == aid, metric]
        metric_colors = {aid: agent_colors.get(aid, "0.5") for aid in ordered_agents}

    for ax, metric in zip(axes_flat, metrics):
        data = [
            metric_data_for(group, metric).dropna().to_numpy(dtype=float)
            for group in metric_groups
        ]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="black", markeredgecolor="black", markersize=4),
            medianprops=dict(color="black", linewidth=1.2),
        )
        for patch, group in zip(bp["boxes"], metric_groups):
            patch.set_facecolor(metric_colors.get(group, "0.5"))
            patch.set_alpha(0.55)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(np.arange(1, len(metric_groups) + 1))
        ax.set_xticklabels(metric_labels, rotation=0, fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)
    metric_title_suffix = "Grouped by Skill" if group_by_skill else "By Agent"
    fig_metrics.suptitle(f"Agent Metrics Across Dense Logs ({metric_title_suffix})", fontsize=14, fontweight="bold")

    trade_summary = (
        metric_df
        .groupby(["agent", "skill", "build_payment"], as_index=False)
        .agg(
            wood_sell_value=("wood_sell_value", "mean"),
            wood_sell_std=("wood_sell_value", "std"),
            wood_buy_value=("wood_buy_value", "mean"),
            wood_buy_std=("wood_buy_value", "std"),
            stone_sell_value=("stone_sell_value", "mean"),
            stone_sell_std=("stone_sell_value", "std"),
            stone_buy_value=("stone_buy_value", "mean"),
            stone_buy_std=("stone_buy_value", "std"),
            wood_sell_count=("wood_sell_count", "mean"),
            wood_buy_count=("wood_buy_count", "mean"),
            stone_sell_count=("stone_sell_count", "mean"),
            stone_buy_count=("stone_buy_count", "mean"),
        )
    ).fillna(0.0)

    fig_trade, trade_axes = plt.subplots(2, 2, figsize=figsize_trade, sharex=True, constrained_layout=True)
    trade_specs = [
        ("Wood", "Sell", "wood_sell_value", "wood_sell_std", "#4c78a8"),
        ("Wood", "Buy", "wood_buy_value", "wood_buy_std", "#72b7b2"),
        ("Stone", "Sell", "stone_sell_value", "stone_sell_std", "#f58518"),
        ("Stone", "Buy", "stone_buy_value", "stone_buy_std", "#eeca3b"),
    ]
    if group_by_skill:
        trade_groups = skill_order
        trade_labels = trade_groups

        def trade_values_for(group, value_col):
            return metric_df.loc[metric_df["skill"] == group, value_col].dropna().to_numpy(dtype=float)
    else:
        trade_groups = ordered_agents
        trade_labels = [agent_label(aid) for aid in ordered_agents]

        def trade_values_for(group, value_col):
            return metric_df.loc[metric_df["agent"] == group, value_col].dropna().to_numpy(dtype=float)

    x = np.arange(len(trade_groups))
    for ax, (commodity, side, value_col, std_col, color) in zip(trade_axes.ravel(), trade_specs):
        values = []
        errs = []
        for group in trade_groups:
            raw_vals = trade_values_for(group, value_col)
            values.append(float(np.nanmean(raw_vals)) if len(raw_vals) else 0.0)
            errs.append(float(np.nanstd(raw_vals, ddof=1)) if len(raw_vals) > 1 else 0.0)
        ax.bar(x, values, yerr=errs, color=color, alpha=0.82, capsize=3, edgecolor="white")
        ax.set_title(f"{side} {commodity}: Mean Value per Dense Log")
        ax.set_ylabel("coin")
        ax.set_xticks(x)
        ax.set_xticklabels(trade_labels, fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    trade_title_suffix = "Grouped by Skill" if group_by_skill else "By Agent"
    fig_trade.suptitle(
        f"Trading Values by Commodity and Direction ({trade_title_suffix})",
        fontsize=14,
        fontweight="bold",
    )

    if group_by_skill:
        count_groups = skill_order
        count_title_suffix = "Grouped by Skill"
    else:
        count_groups = ordered_agents
        count_title_suffix = "By Agent"

    count_labels = [
        group if group_by_skill else agent_label(group)
        for group in count_groups
    ]
    bin_size = max(1, int(trade_count_bin_size))
    max_steps = max(rollout_lengths.values()) if rollout_lengths else 0
    n_bins = max(1, int(math.ceil(max_steps / bin_size)))
    bin_ids = list(range(n_bins))

    if trade_event_df.empty:
        raw_count_df = pd.DataFrame(columns=[
            "rollout_id", "time_bin", "agent", "skill", "commodity", "side", "count"
        ])
    else:
        raw_count_df = trade_event_df.copy()
        raw_count_df["time_bin"] = (raw_count_df["timestep"].astype(int) // bin_size).clip(lower=0)
        raw_count_df = (
            raw_count_df
            .groupby(["rollout_id", "time_bin", "agent", "skill", "commodity", "side"], as_index=False)
            .agg(count=("count", "sum"))
        )

    count_grid_rows = []
    for rollout_id, _log in log_items:
        for time_bin in bin_ids:
            for aid in agent_ids:
                for commodity in ["Wood", "Stone"]:
                    for side in ["Sell", "Buy"]:
                        count_grid_rows.append({
                            "rollout_id": rollout_id,
                            "time_bin": time_bin,
                            "agent": aid,
                            "skill": skill_label(aid),
                            "commodity": commodity,
                            "side": side,
                        })
    count_grid = pd.DataFrame(count_grid_rows)
    count_plot_raw = count_grid.merge(
        raw_count_df,
        on=["rollout_id", "time_bin", "agent", "skill", "commodity", "side"],
        how="left",
    )
    count_plot_raw["count"] = count_plot_raw["count"].fillna(0.0)

    if group_by_skill:
        per_rollout_count = (
            count_plot_raw
            .groupby(["rollout_id", "time_bin", "skill", "commodity", "side"], as_index=False)
            .agg(count=("count", "mean"))
        )
        count_plot_df = (
            per_rollout_count
            .groupby(["time_bin", "skill", "commodity", "side"], as_index=False)
            .agg(mean_count=("count", "mean"), std_count=("count", "std"))
        )
        count_group_col = "skill"
    else:
        count_plot_df = (
            count_plot_raw
            .groupby(["time_bin", "agent", "commodity", "side"], as_index=False)
            .agg(mean_count=("count", "mean"), std_count=("count", "std"))
        )
        count_group_col = "agent"
    count_plot_df["std_count"] = count_plot_df["std_count"].fillna(0.0)

    n_count_cols = min(4, max(1, len(count_groups)))
    n_count_rows = int(math.ceil(len(count_groups) / n_count_cols))
    fig_skill, count_axes = plt.subplots(
        n_count_rows,
        n_count_cols,
        figsize=(
            max(figsize_skill[0], 3.4 * n_count_cols),
            max(figsize_skill[1], 2.7 * n_count_rows),
        ),
        squeeze=False,
        constrained_layout=True,
    )
    count_axes_flat = count_axes.ravel()
    resource_colors = {"Wood": "#8fcf7b", "Stone": "#f2ead4"}
    x_vals = np.array([(b * bin_size) + (bin_size / 2.0) for b in bin_ids], dtype=float)

    max_abs_count = 1.0
    for _, row in count_plot_df.iterrows():
        max_abs_count = max(max_abs_count, abs(float(row["mean_count"])))

    for ax, group, label in zip(count_axes_flat, count_groups, count_labels):
        ax.set_facecolor([0.30, 0.30, 0.30])
        ax.axhline(0, color="white", linewidth=1.0, alpha=0.8)
        for commodity in ["Wood", "Stone"]:
            sell_match = count_plot_df[
                (count_plot_df[count_group_col] == group)
                & (count_plot_df["commodity"] == commodity)
                & (count_plot_df["side"] == "Sell")
            ].set_index("time_bin").reindex(bin_ids)
            buy_match = count_plot_df[
                (count_plot_df[count_group_col] == group)
                & (count_plot_df["commodity"] == commodity)
                & (count_plot_df["side"] == "Buy")
            ].set_index("time_bin").reindex(bin_ids)
            sell_counts = sell_match["mean_count"].fillna(0.0).to_numpy(dtype=float)
            buy_counts = buy_match["mean_count"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x_vals,
                sell_counts,
                color=resource_colors[commodity],
                linewidth=2.0,
                marker=".",
                markersize=5,
                alpha=0.95,
            )
            ax.plot(
                x_vals,
                -buy_counts,
                color=resource_colors[commodity],
                linewidth=2.0,
                marker=".",
                markersize=5,
                alpha=0.95,
            )

        ax.set_title(label, fontsize=9, color="black")
        ax.set_xlim(0, max_steps if max_steps > 0 else bin_size)
        ax.set_ylim(-max_abs_count * 1.25, max_abs_count * 1.25)
        ax.set_xlabel("timestep", fontsize=8)
        ax.tick_params(axis="y", colors="black", labelsize=8)
        ax.grid(True, axis="y", color="white", alpha=0.18)

    for ax in count_axes_flat[len(count_groups):]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color=resource_colors["Wood"], linewidth=2.5, label="Wood"),
        Line2D([0], [0], color=resource_colors["Stone"], linewidth=2.5, label="Stone"),
    ]
    fig_skill.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=True,
        fontsize=9,
    )
    fig_skill.subplots_adjust(bottom=0.16)
    fig_skill.suptitle(
        f"Average Trade Counts by {bin_size}-Step Time Bin ({count_title_suffix})",
        fontsize=14,
        fontweight="bold",
    )

    figures = {
        "metrics": fig_metrics,
        "trade_values_by_agent": fig_trade,
        "trade_counts_by_skill": fig_skill,
    }
    tables = {
        "agent_metrics": metric_df,
        "trade_events": trade_event_df,
        "trade_summary": trade_summary,
        "trade_counts": count_plot_df,
    }
    return figures, tables, dense_logs


def extract_periodic_tax_streams(log, tax_keys=("PeriodicTax-p_top", "PeriodicTax-p_bottom")):
    """Summarize sparse PeriodicTax component logs into one row per tax event."""
    import numpy as np
    import pandas as pd

    rows = []
    for tax_key in tax_keys:
        stream = log.get(tax_key, [])
        if stream is None:
            continue

        planner = tax_key.replace("PeriodicTax-", "")
        tax_period = 0
        for timestep, event in enumerate(stream):
            if not isinstance(event, dict) or not event:
                continue

            agent_rows = [
                value for key, value in event.items()
                if str(key).isdigit() and isinstance(value, dict)
            ]
            tax_period += 1
            schedule = np.asarray(event.get("schedule", []), dtype=float)

            rows.append({
                "planner": planner,
                "tax_key": tax_key,
                "tax_period": tax_period,
                "timestep": timestep,
                "total_income": float(np.nansum([r.get("income", 0.0) for r in agent_rows])),
                "total_tax_paid": float(np.nansum([r.get("tax_paid", 0.0) for r in agent_rows])),
                "mean_lump_sum": float(np.nanmean([r.get("lump_sum", np.nan) for r in agent_rows])) if agent_rows else np.nan,
                "top_marginal_rate": float(schedule[-1]) if len(schedule) else np.nan,
                "mean_marginal_rate": float(np.nanmean(schedule)) if len(schedule) else np.nan,
            })

    return pd.DataFrame(rows)


def plot_periodic_tax_streams_from_result_folder(result_dir, episode_key=0, figsize=(10, 8)):
    """
    Simple plot for dense_log["PeriodicTax-p_top"] and
    dense_log["PeriodicTax-p_bottom"] from one result folder.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    dense_log, dense_logs = get_dense_log_from_result_folder(result_dir, episode_key=episode_key)
    df = extract_periodic_tax_streams(dense_log)
    if df.empty:
        raise ValueError(
            "No PeriodicTax-p_top or PeriodicTax-p_bottom events found in the selected dense log."
        )

    colors = {"p_top": "#1f77b4", "p_bottom": "#d62728"}
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25])
    ax_tax_paid = fig.add_subplot(gs[0, :])
    ax_lump_sum = fig.add_subplot(gs[1, :], sharex=ax_tax_paid)
    ax_top_schedule = fig.add_subplot(gs[2, 0])
    ax_bottom_schedule = fig.add_subplot(gs[2, 1], sharey=ax_top_schedule)
    axes = [ax_tax_paid, ax_lump_sum, ax_top_schedule, ax_bottom_schedule]

    for planner, dfr in df.groupby("planner"):
        dfr = dfr.sort_values("tax_period")
        color = colors.get(planner, None)
        ax_tax_paid.plot(dfr["tax_period"], dfr["total_tax_paid"], marker="o", color=color, label=planner)
        ax_lump_sum.plot(dfr["tax_period"], dfr["mean_lump_sum"], marker="o", color=color, label=planner)

    def last_tax_event(tax_key):
        stream = dense_log.get(tax_key, []) or []
        for event in reversed(stream):
            if isinstance(event, dict) and event:
                return event
        return None

    def plot_schedule(ax, tax_key, title, color):
        event = last_tax_event(tax_key)
        if event is None:
            ax.text(0.5, 0.5, "no tax events", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(title)
            return

        rates = np.asarray(event.get("schedule", []), dtype=float)
        cutoffs = np.asarray(event.get("cutoffs", np.arange(len(rates))), dtype=float)
        if len(rates) == 0:
            ax.text(0.5, 0.5, "no schedule", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(title)
            return

        if len(cutoffs) != len(rates):
            cutoffs = np.arange(len(rates), dtype=float)
        last_bin_width = cutoffs[-1] - cutoffs[-2] if len(cutoffs) > 1 else 1.0
        right_edge = cutoffs[-1] + last_bin_width
        step_x = np.append(cutoffs, right_edge)
        step_y = np.append(rates, rates[-1])

        ax.step(step_x, step_y, where="post", color=color, linewidth=2)
        ax.fill_between(step_x, step_y, step="post", alpha=0.22, color=color)
        ax.set_title(title)
        ax.set_xlabel("Income (k USD)")
        ax.set_ylabel("Marginal rate")
        ax.set_xlim(cutoffs[0], right_edge)
        ax.set_ylim(0, max(1.0, float(np.nanmax(rates)) * 1.05))

    plot_schedule(ax_top_schedule, "PeriodicTax-p_top", "p_top (Top Region)", colors["p_top"])
    plot_schedule(ax_bottom_schedule, "PeriodicTax-p_bottom", "p_bottom (Bottom Region)", colors["p_bottom"])

    ax_tax_paid.set_title("Total Income Tax Paid")
    ax_tax_paid.set_ylabel("coin")
    ax_lump_sum.set_title("Average Lump-Sum Redistribution")
    ax_lump_sum.set_ylabel("coin / agent")
    ax_lump_sum.set_xlabel("Tax period")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    ax_tax_paid.legend(frameon=True)
    ax_lump_sum.legend(frameon=True)

    return fig, df, dense_log, dense_logs




try:
    from simulation import get_disc_rates
except ModuleNotFoundError:
    def get_disc_rates(env_obj=None):
        return np.arange(0.0, 1.0 + 1e-9, 0.05)

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from simulation import get_disc_rates
except ModuleNotFoundError:
    def get_disc_rates(env_obj=None):
        return np.arange(0.0, 1.0 + 1e-9, 0.05)


def _extract_logs_from_run(run):
    if "dense_logs" in run:
        obj = run["dense_logs"]
    elif "dense_log" in run:
        obj = {0: run["dense_log"]}
    else:
        return {}

    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {i: log for i, log in enumerate(obj)}
    return {}


def _numeric_agent_ids(log):
    return sorted(
        int(k) for k, v in log["states"][0].items()
        if str(k).isdigit() and isinstance(v, dict) and "loc" in v
    )


def _infer_waterline(log):
    for world_state in log.get("world", []):
        if not world_state:
            continue
        for arr in world_state.values():
            try:
                shape = np.asarray(arr).shape
            except Exception:
                continue
            if len(shape) >= 2 and shape[0] > 0:
                return int(shape[0] // 2)
    return 25


def _planner_region_from_initial_state(log, aid, waterline=None):
    if waterline is None:
        waterline = _infer_waterline(log)

    s0 = log["states"][0][str(aid)]
    row0 = int(s0["loc"][0])

    # Matches regional_two_planner.py:
    # p_top if initial row <= waterline, else p_bottom
    return "top" if row0 <= waterline else "bottom"


def _location_region_from_state(s, waterline):
    row = int(s["loc"][0])
    return "top" if row <= waterline else "bottom"


def _coin(s):
    return float(s["inventory"].get("Coin", 0.0)) + float(s["escrow"].get("Coin", 0.0))


def _resource_total(s, resource):
    return float(s["inventory"].get(resource, 0.0)) + float(s["escrow"].get(resource, 0.0))


def _movement_distance(log, aid):
    locs = np.array([st[str(aid)]["loc"] for st in log["states"]], dtype=float)
    if len(locs) <= 1:
        return 0.0
    return float(np.abs(np.diff(locs, axis=0)).sum())


def _region_switches(log, aid, waterline=None):
    if waterline is None:
        waterline = _infer_waterline(log)

    regions = [
        _location_region_from_state(st[str(aid)], waterline=waterline)
        for st in log["states"]
    ]
    return int(sum(regions[i] != regions[i - 1] for i in range(1, len(regions))))


def _build_income_by_agent(log):
    out = {}
    for builds in log.get("Build", []):
        builds_ = builds.get("builds", []) if isinstance(builds, dict) else builds
        for b in builds_:
            aid = int(b["builder"])
            out[aid] = out.get(aid, 0.0) + float(b.get("income", 0.0))
    return out


def _trade_income_by_agent(log):
    sell_income = {}
    buy_cost = {}

    for trades in log.get("Trade", []):
        trades_ = trades.get("trades", []) if isinstance(trades, dict) else trades
        for tr in trades_:
            seller = int(tr["seller"])
            buyer = int(tr["buyer"])

            sell_income[seller] = sell_income.get(seller, 0.0) + float(
                tr.get("income", tr.get("price", 0.0))
            )
            buy_cost[buyer] = buy_cost.get(buyer, 0.0) + float(
                tr.get("cost", tr.get("price", 0.0))
            )

    return sell_income, buy_cost


def _agent_behavior_table(log):
    aids = _numeric_agent_ids(log)
    waterline = _infer_waterline(log)

    build_income = _build_income_by_agent(log)
    sell_income, buy_cost = _trade_income_by_agent(log)

    rows = []
    for aid in aids:
        s0 = log["states"][0][str(aid)]
        s1 = log["states"][-1][str(aid)]

        rows.append({
            "agent": aid,
            "skill_build_payment": float(s0.get("build_payment", np.nan)),
            "gather_bonus": float(s0.get("bonus_gather_prob", np.nan)),

            # Fixed tax/planner assignment. This is the important grouping.
            "planner_region": _planner_region_from_initial_state(log, aid, waterline=waterline),

            # Physical location regions, useful for travel diagnostics.
            "start_location_region": _location_region_from_state(s0, waterline=waterline),
            "final_location_region": _location_region_from_state(s1, waterline=waterline),

            "final_coin": _coin(s1),
            "coin_change": _coin(s1) - _coin(s0),
            "final_wood": _resource_total(s1, "Wood"),
            "final_stone": _resource_total(s1, "Stone"),
            "final_labor": float(s1.get("endogenous", {}).get("Labor", np.nan)),
            "final_utility": float(s1.get("utility", np.nan)),
            "build_income": build_income.get(aid, 0.0),
            "sell_income": sell_income.get(aid, 0.0),
            "buy_cost": buy_cost.get(aid, 0.0),
            "total_income": build_income.get(aid, 0.0) + sell_income.get(aid, 0.0),
            "net_market": sell_income.get(aid, 0.0) - buy_cost.get(aid, 0.0),
            "move_distance": _movement_distance(log, aid),
            "region_switches": _region_switches(log, aid, waterline=waterline),
        })

    return pd.DataFrame(rows)


def _final_tax_rates(log, env_obj, top_first=True):
    disc_rates = get_disc_rates(env_obj)

    def split_last(arr):
        last = np.asarray(arr)[-1]
        half = last.shape[0] // 2
        if top_first:
            return last[:half], last[half:]
        else:
            return last[half:], last[:half]

    planner_actions = log["planner_actions"]

    top_actions, _ = split_last(planner_actions["p_top"])
    _, bottom_actions = split_last(planner_actions["p_bottom"])

    top_rates = disc_rates[np.clip(top_actions.astype(int), 0, len(disc_rates) - 1)]
    bottom_rates = disc_rates[np.clip(bottom_actions.astype(int), 0, len(disc_rates) - 1)]

    return top_rates, bottom_rates

def plot_tax_and_agent_behavior_for_logs(
    runs,
    env_obj,
    brackets,
    selected=None,          # None = first few logs; or [(run_idx, log_key), ...]
    top_first=True,
    short_labels=None,
    last_bin_width=100,
    max_logs=6,
    behavior_metric="total_income",
    figsize_per_log=(13, 3.2),
):
    brackets = np.asarray(brackets, dtype=float)
    step_x = np.append(brackets, brackets[-1] + last_bin_width)

    available = []
    selected_set = set(selected) if selected is not None else None

    for run_idx, run in enumerate(runs):
        logs = _extract_logs_from_run(run)
        for log_key, log in logs.items():
            if selected_set is None or (run_idx, log_key) in selected_set:
                available.append((run_idx, log_key, run, log))

    if selected is None:
        available = available[:max_logs]

    n = len(available)
    if n == 0:
        raise ValueError("No dense logs selected/found.")

    fig, axes = plt.subplots(
        n,
        4,
        figsize=(figsize_per_log[0], figsize_per_log[1] * n),
        squeeze=False,
    )

    region_colors = {"top": "#1f77b4", "bottom": "#ff7f0e"}

    for row, (run_idx, log_key, run, log) in enumerate(available):
        run_label = (
            short_labels[run_idx]
            if short_labels is not None
            else run.get("name", f"run {run_idx}")
        )

        top_rates, bottom_rates = _final_tax_rates(log, env_obj, top_first=top_first)
        top_step_y = np.append(top_rates, top_rates[-1])
        bottom_step_y = np.append(bottom_rates, bottom_rates[-1])

        df_agents = _agent_behavior_table(log)
        df_agents = df_agents.sort_values(["planner_region", "skill_build_payment", "agent"])

        # 1. Top tax schedule
        ax = axes[row, 0]
        ax.step(step_x, top_step_y, where="post", color="#1f77b4", linewidth=2)
        ax.set_title(f"{run_label}, log {log_key}\np_top tax")
        ax.set_ylim(0, max(1.0, top_step_y.max() * 1.05))
        ax.grid(True, alpha=0.3)

        # 2. Bottom tax schedule
        ax = axes[row, 1]
        ax.step(step_x, bottom_step_y, where="post", color="#ff7f0e", linewidth=2)
        ax.set_title("p_bottom tax")
        ax.set_ylim(0, max(1.0, bottom_step_y.max() * 1.05))
        ax.grid(True, alpha=0.3)

        # 3. Agent behavior bars, colored by fixed planner/tax region
        ax = axes[row, 2]
        colors = [region_colors.get(r, "gray") for r in df_agents["planner_region"]]
        labels = [
            f"{int(a)} ({pr[0]}->{lr[0]})"
            for a, pr, lr in zip(
                df_agents["agent"],
                df_agents["planner_region"],
                df_agents["final_location_region"],
            )
        ]

        ax.bar(labels, df_agents[behavior_metric], color=colors, alpha=0.85)
        ax.set_title(f"Agent {behavior_metric}\nlabel = agent (planner->final loc)")
        ax.set_xlabel("agent")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

        # 4. Coin/labor scatter, colored by fixed planner/tax region
        ax = axes[row, 3]
        colors = [region_colors.get(r, "gray") for r in df_agents["planner_region"]]

        skill = df_agents["skill_build_payment"]
        if skill.notna().any() and skill.max() > skill.min():
            sizes = 40 + 100 * ((skill - skill.min()) / (skill.max() - skill.min()))
        else:
            sizes = np.full(len(df_agents), 70)

        ax.scatter(
            df_agents["final_labor"],
            df_agents["final_coin"],
            c=colors,
            s=sizes,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        for _, r in df_agents.iterrows():
            ax.annotate(
                str(int(r["agent"])),
                (r["final_labor"], r["final_coin"]),
                fontsize=8,
            )

        ax.set_title("Final coin vs labor\ncolor = fixed planner region")
        ax.set_xlabel("final labor")
        ax.set_ylabel("final coin")
        ax.grid(True, alpha=0.3)

        # Print counts so assignment issues are visible immediately
        print(f"{run_label}, log {log_key}")
        print("planner_region counts:")
        print(df_agents["planner_region"].value_counts(dropna=False).to_string())
        print("final_location_region counts:")
        print(df_agents["final_location_region"].value_counts(dropna=False).to_string())
        print()

    fig.tight_layout()
    return fig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
try:
    from simulation import get_disc_rates
except ModuleNotFoundError:
    def get_disc_rates(env_obj=None):
        return np.arange(0.0, 1.0 + 1e-9, 0.05)


def _extract_logs_from_run(run):
    if "dense_logs" in run:
        obj = run["dense_logs"]
    elif "dense_log" in run:
        obj = {0: run["dense_log"]}
    else:
        return {}

    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {i: log for i, log in enumerate(obj)}
    return {}


def _numeric_agent_ids(log):
    return sorted(
        int(k) for k, v in log["states"][0].items()
        if str(k).isdigit() and isinstance(v, dict) and "loc" in v
    )


def _infer_waterline(log):
    for world_state in log.get("world", []):
        if not world_state:
            continue
        for arr in world_state.values():
            try:
                shape = np.asarray(arr).shape
            except Exception:
                continue
            if len(shape) >= 2 and shape[0] > 0:
                return int(shape[0] // 2)
    return 25


def _location_region_from_state(s, waterline):
    row = int(s["loc"][0])
    return "top" if row <= waterline else "bottom"


def _planner_region_from_initial_state(log, aid, waterline=None):
    if waterline is None:
        waterline = _infer_waterline(log)
    s0 = log["states"][0][str(aid)]
    row0 = int(s0["loc"][0])
    return "top" if row0 <= waterline else "bottom"


def _coin(s):
    return float(s["inventory"].get("Coin", 0.0)) + float(s["escrow"].get("Coin", 0.0))


def _resource_total(s, resource):
    return float(s["inventory"].get(resource, 0.0)) + float(s["escrow"].get(resource, 0.0))


def _movement_distance(log, aid):
    locs = np.array([st[str(aid)]["loc"] for st in log["states"]], dtype=float)
    if len(locs) <= 1:
        return 0.0
    return float(np.abs(np.diff(locs, axis=0)).sum())


def _iter_travel_events(log):
    for event in log.get("CrossWaterTravel", []):
        if isinstance(event, dict):
            if "agent" in event:
                yield event
            else:
                for value in event.values():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                yield item
        elif isinstance(event, list):
            for item in event:
                if isinstance(item, dict):
                    yield item


def _travel_events_by_agent(log):
    out = {}
    for event in _iter_travel_events(log):
        if "agent" not in event:
            continue
        aid = int(event["agent"])
        out.setdefault(aid, []).append(event)
    return out


def _region_sequence(log, aid, waterline=None):
    if waterline is None:
        waterline = _infer_waterline(log)
    return [
        _location_region_from_state(st[str(aid)], waterline=waterline)
        for st in log["states"]
    ]


def _region_shares(log, aid, waterline=None):
    regions = _region_sequence(log, aid, waterline=waterline)
    n = max(1, len(regions))
    share_top = sum(r == "top" for r in regions) / n
    share_bottom = sum(r == "bottom" for r in regions) / n
    switches = sum(regions[i] != regions[i - 1] for i in range(1, len(regions)))
    majority = "top" if share_top >= share_bottom else "bottom"
    return share_top, share_bottom, majority, int(switches)


def _build_income_by_agent(log):
    out = {}
    for builds in log.get("Build", []):
        builds_ = builds.get("builds", []) if isinstance(builds, dict) else builds
        for b in builds_:
            aid = int(b["builder"])
            out[aid] = out.get(aid, 0.0) + float(b.get("income", 0.0))
    return out


def _trade_income_by_agent(log):
    sell_income = {}
    buy_cost = {}

    for trades in log.get("Trade", []):
        trades_ = trades.get("trades", []) if isinstance(trades, dict) else trades
        for tr in trades_:
            seller = int(tr["seller"])
            buyer = int(tr["buyer"])

            sell_income[seller] = sell_income.get(seller, 0.0) + float(
                tr.get("income", tr.get("price", 0.0))
            )
            buy_cost[buyer] = buy_cost.get(buyer, 0.0) + float(
                tr.get("cost", tr.get("price", 0.0))
            )

    return sell_income, buy_cost


def _agent_behavior_table(log):
    aids = _numeric_agent_ids(log)
    waterline = _infer_waterline(log)

    build_income = _build_income_by_agent(log)
    sell_income, buy_cost = _trade_income_by_agent(log)
    travel_events = _travel_events_by_agent(log)

    rows = []
    for aid in aids:
        s0 = log["states"][0][str(aid)]
        s1 = log["states"][-1][str(aid)]

        share_top, share_bottom, majority_region, switches = _region_shares(
            log, aid, waterline=waterline
        )

        rows.append({
            "agent": aid,
            "skill_build_payment": float(s0.get("build_payment", np.nan)),
            "gather_bonus": float(s0.get("bonus_gather_prob", np.nan)),

            "planner_region": _planner_region_from_initial_state(log, aid, waterline=waterline),
            "start_location_region": _location_region_from_state(s0, waterline=waterline),
            "final_location_region": _location_region_from_state(s1, waterline=waterline),
            "majority_location_region": majority_region,
            "share_top": share_top,
            "share_bottom": share_bottom,
            "region_switches": switches,

            "final_coin": _coin(s1),
            "coin_change": _coin(s1) - _coin(s0),
            "final_wood": _resource_total(s1, "Wood"),
            "final_stone": _resource_total(s1, "Stone"),
            "final_labor": float(s1.get("endogenous", {}).get("Labor", np.nan)),
            "final_utility": float(s1.get("utility", np.nan)),
            "build_income": build_income.get(aid, 0.0),
            "sell_income": sell_income.get(aid, 0.0),
            "buy_cost": buy_cost.get(aid, 0.0),
            "total_income": build_income.get(aid, 0.0) + sell_income.get(aid, 0.0),
            "net_market": sell_income.get(aid, 0.0) - buy_cost.get(aid, 0.0),
            "move_distance": _movement_distance(log, aid),
            "travel_events": len(travel_events.get(aid, [])),
        })

    return pd.DataFrame(rows)


def _split_last_tax_action(arr, top_first=True):
    last = np.asarray(arr)[-1]
    half = last.shape[0] // 2
    if top_first:
        return last[:half], last[half:]
    return last[half:], last[:half]


def _split_tax_action_matrix(arr, top_first=True):
    arr = np.asarray(arr)
    half = arr.shape[1] // 2
    if top_first:
        return arr[:, :half], arr[:, half:]
    return arr[:, half:], arr[:, :half]


def _final_tax_rates(log, env_obj, top_first=True):
    disc_rates = get_disc_rates(env_obj)
    planner_actions = log["planner_actions"]

    top_actions, _ = _split_last_tax_action(planner_actions["p_top"], top_first=top_first)
    _, bottom_actions = _split_last_tax_action(planner_actions["p_bottom"], top_first=top_first)

    top_rates = disc_rates[np.clip(top_actions.astype(int), 0, len(disc_rates) - 1)]
    bottom_rates = disc_rates[np.clip(bottom_actions.astype(int), 0, len(disc_rates) - 1)]

    return top_rates, bottom_rates


def _tax_rate_matrices(log, env_obj, top_first=True):
    disc_rates = get_disc_rates(env_obj)

    ptop_actions = np.asarray(log["planner_actions"]["p_top"])
    pbot_actions = np.asarray(log["planner_actions"]["p_bottom"])

    ptop_top, _ = _split_tax_action_matrix(ptop_actions, top_first=top_first)
    _, pbot_bottom = _split_tax_action_matrix(pbot_actions, top_first=top_first)

    ptop_rates = disc_rates[np.clip(ptop_top.astype(int), 0, len(disc_rates) - 1)]
    pbot_rates = disc_rates[np.clip(pbot_bottom.astype(int), 0, len(disc_rates) - 1)]

    return ptop_rates, pbot_rates


def _period_income_table(log, period=100):
    states = log["states"]
    aids = _numeric_agent_ids(log)
    waterline = _infer_waterline(log)

    tax_days = list(range(period - 1, len(states), period))
    prev_idx = 0

    rows = []
    for tax_day_number, t in enumerate(tax_days, start=1):
        for aid in aids:
            s_prev = states[prev_idx][str(aid)]
            s_now = states[t][str(aid)]
            date = _date_from_state(s_now)

            rows.append({
                "tax_day_number": tax_day_number,
                "timestep": t,
                "date": date,
                "agent": aid,
                "planner_region": _planner_region_from_initial_state(log, aid, waterline=waterline),
                "location_region": _location_region_from_state(s_now, waterline=waterline),
                "income": _coin(s_now) - _coin(s_prev),
                "coin_end": _coin(s_now),
                "labor_end": float(s_now.get("endogenous", {}).get("Labor", np.nan)),
            })

        prev_idx = t

    return pd.DataFrame(rows)


def _date_from_state(state):
    for value in state.values():
        if isinstance(value, dict) and "Date" in value and value["Date"] is not None:
            return value["Date"]
    return None


def _x_values_and_label(df, fallback_col="tax_day_number"):
    if "date" in df.columns and df["date"].notna().any():
        return pd.to_datetime(df["date"]), "date"
    return df[fallback_col], "tax period (dates unavailable in dense log)"


def _bracket_labels_from_cutoffs(cutoffs):
    labels = []
    for lo, hi in zip(cutoffs[:-1], cutoffs[1:]):
        lo_s = "-inf" if np.isneginf(lo) else f"{lo:g}"
        hi_s = "inf" if np.isposinf(hi) else f"{hi:g}"
        labels.append(f"[{lo_s}, {hi_s})")
    return labels


def _income_bracket_counts(log, brackets, period=100):
    df = _period_income_table(log, period=period)

    brackets = np.asarray(brackets, dtype=float)
    cutoffs = np.r_[-np.inf, brackets[1:], np.inf]
    labels = _bracket_labels_from_cutoffs(cutoffs)

    df["tax_bracket"] = pd.cut(
        df["income"],
        bins=cutoffs,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    counts = (
        df.groupby(["tax_day_number", "planner_region", "tax_bracket"], observed=False)
        .size()
        .reset_index(name="n_agents")
    )

    return df, counts, labels


def _gini_coefficient(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    values = np.maximum(values, 0.0)
    total = np.sum(values)

    if total <= 0:
        return np.nan

    values = np.sort(values)
    n = len(values)
    return float(
        (2.0 * np.sum(np.arange(1, n + 1) * values)) / (n * total)
        - (n + 1.0) / n
    )


def _extract_episode_logs(obj):
    if obj is None:
        return []

    if isinstance(obj, dict) and "states" in obj:
        return [obj]

    if isinstance(obj, (list, tuple)):
        eps = []
        for v in obj:
            eps.extend(_extract_episode_logs(v))
        return eps

    if isinstance(obj, dict):
        eps = []
        for key in ["final", "episodes", "dense_logs", "logs", "data"]:
            if key in obj:
                eps.extend(_extract_episode_logs(obj[key]))

        if eps:
            return eps

        for v in obj.values():
            eps.extend(_extract_episode_logs(v))
        return eps

    return []


def plot_agent_income_and_gini(
    log,
    period=100,
    ax=None,
    income_metric="income",
    title="Agent Income and Gini Coefficient by Tax Period",
    agent_cmap="tab10",
    gini_color="#8c564b",
    gini_alpha=0.28,
):
    """
    Plot each agent's income as lines and the per-period Gini coefficient as
    bars on a secondary y-axis.
    """
    df_income = _period_income_table(log, period=period)

    if income_metric not in df_income.columns:
        raise ValueError(
            f"income_metric must be one of {sorted(df_income.columns)}; "
            f"got {income_metric!r}"
        )

    gini_df = (
        df_income.groupby("tax_day_number")[income_metric]
        .apply(_gini_coefficient)
        .reset_index(name="gini")
    )

    if ax is None:
        fig, ax_income = plt.subplots(figsize=(12, 5))
    else:
        ax_income = ax
        fig = ax_income.figure

    ax_gini = ax_income.twinx()
    ax_income.set_zorder(ax_gini.get_zorder() + 1)
    ax_income.patch.set_visible(False)

    x_gini = gini_df["tax_day_number"].to_numpy()
    ax_gini.bar(
        x_gini,
        gini_df["gini"].to_numpy(),
        width=0.72,
        color=gini_color,
        alpha=gini_alpha,
        label="Gini coefficient",
        zorder=1,
    )
    ax_gini.set_ylabel("Gini coefficient", color=gini_color)
    ax_gini.tick_params(axis="y", labelcolor=gini_color)
    finite_gini = gini_df["gini"].to_numpy(dtype=float)
    finite_gini = finite_gini[np.isfinite(finite_gini)]
    gini_top = max(1.0, float(finite_gini.max()) * 1.12) if len(finite_gini) else 1.0
    ax_gini.set_ylim(0, gini_top)

    agents = sorted(df_income["agent"].unique())
    cmap = plt.get_cmap(agent_cmap, max(len(agents), 1))

    for idx, aid in enumerate(agents):
        dfa = df_income[df_income["agent"] == aid].sort_values("tax_day_number")
        ax_income.plot(
            dfa["tax_day_number"],
            dfa[income_metric],
            marker="o",
            linewidth=1.8,
            color=cmap(idx),
            label=f"agent {int(aid)}",
            zorder=3,
        )

    ax_income.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax_income.set_title(title)
    ax_income.set_xlabel("tax period")
    ax_income.set_ylabel(income_metric.replace("_", " "))
    ax_income.grid(True, axis="y", alpha=0.3)

    line_handles, line_labels = ax_income.get_legend_handles_labels()
    bar_handles, bar_labels = ax_gini.get_legend_handles_labels()
    ax_income.legend(
        line_handles + bar_handles,
        line_labels + bar_labels,
        loc="upper left",
        ncol=min(4, len(line_labels) + len(bar_labels)),
        fontsize=9,
        frameon=True,
    )

    fig.tight_layout()
    return fig, ax_income, ax_gini, df_income, gini_df


def plot_agent_income_and_gini_for_log(
    log,
    period=100,
    max_periods=None,
    ax=None,
    income_metric="income",
    title="Agent Income and Gini Coefficient by Tax Period",
    agent_cmap="tab10",
    gini_color="#8c564b",
    gini_alpha=0.28,
):
    """
    Plot one raw dense log. This does not average across multiple dense logs.
    """
    if max_periods is None:
        return plot_agent_income_and_gini(
            log,
            period=period,
            ax=ax,
            income_metric=income_metric,
            title=title,
            agent_cmap=agent_cmap,
            gini_color=gini_color,
            gini_alpha=gini_alpha,
        )

    df_income = _period_income_table(log, period=period)
    df_income = df_income[df_income["tax_day_number"] <= max_periods].copy()

    if income_metric not in df_income.columns:
        raise ValueError(
            f"income_metric must be one of {sorted(df_income.columns)}; "
            f"got {income_metric!r}"
        )

    gini_df = (
        df_income.groupby("tax_day_number")[income_metric]
        .apply(_gini_coefficient)
        .reset_index(name="gini")
    )

    if ax is None:
        fig, ax_income = plt.subplots(figsize=(12, 5))
    else:
        ax_income = ax
        fig = ax_income.figure

    ax_gini = ax_income.twinx()
    ax_income.set_zorder(ax_gini.get_zorder() + 1)
    ax_income.patch.set_visible(False)

    x_gini = gini_df["tax_day_number"].to_numpy()
    ax_gini.bar(
        x_gini,
        gini_df["gini"].to_numpy(),
        width=0.72,
        color=gini_color,
        alpha=gini_alpha,
        label="Gini coefficient",
        zorder=1,
    )
    ax_gini.set_ylabel("Gini coefficient", color=gini_color)
    ax_gini.tick_params(axis="y", labelcolor=gini_color)
    finite_gini = gini_df["gini"].to_numpy(dtype=float)
    finite_gini = finite_gini[np.isfinite(finite_gini)]
    gini_top = max(1.0, float(finite_gini.max()) * 1.12) if len(finite_gini) else 1.0
    ax_gini.set_ylim(0, gini_top)

    agents = sorted(df_income["agent"].unique())
    cmap = plt.get_cmap(agent_cmap, max(len(agents), 1))

    for idx, aid in enumerate(agents):
        dfa = df_income[df_income["agent"] == aid].sort_values("tax_day_number")
        ax_income.plot(
            dfa["tax_day_number"],
            dfa[income_metric],
            marker="o",
            linewidth=1.8,
            color=cmap(idx),
            label=f"agent {int(aid)}",
            zorder=3,
        )

    ax_income.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax_income.set_title(title)
    ax_income.set_xlabel("tax period")
    ax_income.set_ylabel(income_metric.replace("_", " "))
    ax_income.grid(True, axis="y", alpha=0.3)

    line_handles, line_labels = ax_income.get_legend_handles_labels()
    bar_handles, bar_labels = ax_gini.get_legend_handles_labels()
    ax_income.legend(
        line_handles + bar_handles,
        line_labels + bar_labels,
        loc="upper left",
        ncol=min(4, len(line_labels) + len(bar_labels)),
        fontsize=9,
        frameon=True,
    )

    fig.tight_layout()
    return fig, ax_income, ax_gini, df_income, gini_df


def _get_log_from_dense_logs(dense_logs, episode_key=0):
    if isinstance(dense_logs, dict) and "states" in dense_logs:
        return dense_logs

    if isinstance(dense_logs, dict):
        if episode_key in dense_logs:
            return dense_logs[episode_key]

        episode_key_str = str(episode_key)
        if episode_key_str in dense_logs:
            return dense_logs[episode_key_str]

        for v in dense_logs.values():
            if isinstance(v, dict) and "states" in v:
                return v

    if isinstance(dense_logs, (list, tuple)):
        if len(dense_logs) == 0:
            raise ValueError("No dense logs found.")
        return dense_logs[int(episode_key)]

    raise ValueError("Could not find an episode log with states.")


def plot_agent_income_and_gini_for_dense_logs(
    dense_logs,
    episode_key=0,
    period=100,
    max_periods=None,
    **kwargs,
):
    log = _get_log_from_dense_logs(dense_logs, episode_key=episode_key)
    return plot_agent_income_and_gini_for_log(
        log,
        period=period,
        max_periods=max_periods,
        **kwargs,
    )


def plot_agent_income_and_gini_single_log_from_result_folder(
    run_dir,
    episode_key=0,
    period=100,
    max_periods=None,
    **kwargs,
):
    dense_logs = load_dense_logs_from_result_folder(run_dir)
    return plot_agent_income_and_gini_for_dense_logs(
        dense_logs,
        episode_key=episode_key,
        period=period,
        max_periods=max_periods,
        **kwargs,
    )


def plot_agent_income_and_gini_for_runs(
    runs,
    short_labels=None,
    period=100,
    max_periods=None,
    errorbar="std",
    figsize_per_run=(12, 4.5),
    agent_cmap="tab10",
    gini_color="#8c564b",
    gini_alpha=0.28,
):
    """
    Plot agent income lines and Gini bars for experiment runs loaded with
    load_experiment_runs(run_dirs). Uses all dense logs in each run.
    """
    run_names = [run["name"] for run in runs]

    if short_labels is None:
        short_labels = {name: f"E{i + 1}" for i, name in enumerate(run_names)}
    elif isinstance(short_labels, list):
        short_labels = {name: short_labels[i] for i, name in enumerate(run_names)}

    raw_rows = []

    for run in runs:
        name = run["name"]
        dense_logs_obj = run.get("dense_logs", None)
        if dense_logs_obj is None:
            dense_logs_obj = run.get("dense_log", None)

        eps = _extract_episode_logs(dense_logs_obj)

        for rollout_id, log in enumerate(eps):
            df = _period_income_table(log, period=period)
            if max_periods is not None:
                df = df[df["tax_day_number"] <= max_periods]

            for _, row in df.iterrows():
                raw_rows.append({
                    "run": name,
                    "label": short_labels[name],
                    "rollout_id": rollout_id,
                    "tax_day_number": int(row["tax_day_number"]),
                    "timestep": int(row["timestep"]),
                    "agent": int(row["agent"]),
                    "income": float(row["income"]),
                    "planner_region": row["planner_region"],
                    "location_region": row["location_region"],
                })

    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        raise ValueError("No dense logs with period income were found in the supplied runs.")

    income_summary = (
        raw_df.groupby(["run", "label", "tax_day_number", "agent"], sort=False)
        .agg(
            income_mean=("income", "mean"),
            income_std=("income", "std"),
            n_dense_logs=("income", "count"),
        )
        .reset_index()
    )
    income_summary["income_std"] = income_summary["income_std"].fillna(0.0)
    income_summary["income_sem"] = (
        income_summary["income_std"] / np.sqrt(income_summary["n_dense_logs"])
    )

    rollout_gini = (
        raw_df.groupby(["run", "label", "rollout_id", "tax_day_number"], sort=False)["income"]
        .apply(_gini_coefficient)
        .reset_index(name="gini")
    )

    gini_summary = (
        rollout_gini.groupby(["run", "label", "tax_day_number"], sort=False)
        .agg(
            gini_mean=("gini", "mean"),
            gini_std=("gini", "std"),
            n_dense_logs=("gini", "count"),
        )
        .reset_index()
    )
    gini_summary["gini_std"] = gini_summary["gini_std"].fillna(0.0)
    gini_summary["gini_sem"] = (
        gini_summary["gini_std"] / np.sqrt(gini_summary["n_dense_logs"])
    )

    if errorbar == "std":
        income_err_col = "income_std"
        gini_err_col = "gini_std"
    elif errorbar == "sem":
        income_err_col = "income_sem"
        gini_err_col = "gini_sem"
    elif errorbar is None:
        income_err_col = None
        gini_err_col = None
    else:
        raise ValueError("errorbar must be None, 'std', or 'sem'")

    fig, axes = plt.subplots(
        len(run_names),
        1,
        figsize=(figsize_per_run[0], figsize_per_run[1] * len(run_names)),
        squeeze=False,
    )

    twin_axes = []

    for row_idx, name in enumerate(run_names):
        ax_income = axes[row_idx, 0]
        ax_gini = ax_income.twinx()
        twin_axes.append(ax_gini)

        ax_income.set_zorder(ax_gini.get_zorder() + 1)
        ax_income.patch.set_visible(False)

        run_income = income_summary[income_summary["run"] == name]
        run_gini = gini_summary[gini_summary["run"] == name]

        x_gini = run_gini["tax_day_number"].to_numpy()
        y_gini = run_gini["gini_mean"].to_numpy()
        yerr_gini = None if gini_err_col is None else run_gini[gini_err_col].to_numpy()

        ax_gini.bar(
            x_gini,
            y_gini,
            width=0.72,
            color=gini_color,
            alpha=gini_alpha,
            label="Gini coefficient",
            yerr=yerr_gini,
            capsize=4 if gini_err_col is not None else 0,
            ecolor=gini_color,
            zorder=1,
        )

        finite_gini = y_gini[np.isfinite(y_gini)]
        gini_top = max(1.0, float(finite_gini.max()) * 1.12) if len(finite_gini) else 1.0
        ax_gini.set_ylim(0, gini_top)
        ax_gini.set_ylabel("Gini coefficient", color=gini_color)
        ax_gini.tick_params(axis="y", labelcolor=gini_color)

        agents = sorted(run_income["agent"].unique())
        cmap = plt.get_cmap(agent_cmap, max(len(agents), 1))

        for idx, aid in enumerate(agents):
            dfa = run_income[run_income["agent"] == aid].sort_values("tax_day_number")
            x = dfa["tax_day_number"].to_numpy()
            y = dfa["income_mean"].to_numpy()

            color = cmap(idx)
            ax_income.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                color=color,
                label=f"agent {int(aid)}",
                zorder=3,
            )

            if income_err_col is not None:
                err = dfa[income_err_col].to_numpy()
                ax_income.fill_between(
                    x,
                    y - err,
                    y + err,
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                    zorder=2,
                )

        ax_income.axhline(0, color="black", linewidth=0.8, alpha=0.7)
        ax_income.set_title(f"{short_labels[name]}: Agent Income and Gini by Tax Period")
        ax_income.set_xlabel("Tax period")
        ax_income.set_ylabel("Income")
        ax_income.grid(True, axis="y", alpha=0.3)

        line_handles, line_labels = ax_income.get_legend_handles_labels()
        bar_handles, bar_labels = ax_gini.get_legend_handles_labels()
        ax_income.legend(
            line_handles + bar_handles,
            line_labels + bar_labels,
            loc="upper left",
            ncol=min(4, len(line_labels) + len(bar_labels)),
            fontsize=9,
            frameon=True,
        )

    fig.tight_layout()
    return fig, axes[:, 0].tolist(), twin_axes, income_summary, gini_summary, raw_df


def plot_agent_income_and_gini_for_run(
    run,
    label=None,
    period=100,
    max_periods=None,
    errorbar="std",
    figsize=(12, 4.5),
    agent_cmap="tab10",
    gini_color="#8c564b",
    gini_alpha=0.28,
):
    """
    Plot one experiment run loaded with load_experiment_run(run_dir). Uses all
    dense logs in that run.
    """
    run_name = run["name"]
    short_labels = [label] if label is not None else None

    fig, ax_income_list, ax_gini_list, income_summary, gini_summary, raw_df = (
        plot_agent_income_and_gini_for_runs(
            [run],
            short_labels=short_labels,
            period=period,
            max_periods=max_periods,
            errorbar=errorbar,
            figsize_per_run=figsize,
            agent_cmap=agent_cmap,
            gini_color=gini_color,
            gini_alpha=gini_alpha,
        )
    )

    return (
        fig,
        ax_income_list[0],
        ax_gini_list[0],
        income_summary[income_summary["run"] == run_name].reset_index(drop=True),
        gini_summary[gini_summary["run"] == run_name].reset_index(drop=True),
        raw_df[raw_df["run"] == run_name].reset_index(drop=True),
    )


def plot_agent_income_and_gini_from_result_folder(
    run_dir,
    label=None,
    period=100,
    max_periods=None,
    errorbar="std",
    figsize=(12, 4.5),
):
    run = load_experiment_run(run_dir)
    return plot_agent_income_and_gini_for_run(
        run,
        label=label,
        period=period,
        max_periods=max_periods,
        errorbar=errorbar,
        figsize=figsize,
    )


def plot_agent_income_and_gini_from_result_folders(
    run_dirs,
    short_labels=None,
    period=100,
    max_periods=None,
    errorbar="std",
    figsize_per_run=(12, 4.5),
):
    runs = load_experiment_runs(run_dirs)
    return plot_agent_income_and_gini_for_runs(
        runs,
        short_labels=short_labels,
        period=period,
        max_periods=max_periods,
        errorbar=errorbar,
        figsize_per_run=figsize_per_run,
    )


def plot_tax_and_agent_behavior_for_logs_v2(
    runs,
    env_obj,
    brackets,
    selected=None,
    top_first=True,
    short_labels=None,
    last_bin_width=100,
    max_logs=4,
    behavior_metric="total_income",
    figsize_per_log=(16, 3.6),
):
    brackets = np.asarray(brackets, dtype=float)
    step_x = np.append(brackets, brackets[-1] + last_bin_width)

    available = []
    selected_set = set(selected) if selected is not None else None

    for run_idx, run in enumerate(runs):
        logs = _extract_logs_from_run(run)
        for log_key, log in logs.items():
            if selected_set is None or (run_idx, log_key) in selected_set:
                available.append((run_idx, log_key, run, log))

    if selected is None:
        available = available[:max_logs]

    if len(available) == 0:
        available_keys = {
            run_idx: list(_extract_logs_from_run(run).keys())
            for run_idx, run in enumerate(runs)
        }
        raise ValueError(
            "No dense logs selected/found. "
            f"Available log keys by run are: {available_keys}"
        )

    region_colors = {"top": "#1f77b4", "bottom": "#ff7f0e"}
    edge_colors = {"top": "#08306b", "bottom": "#7f2704"}

    fig, axes = plt.subplots(
        len(available),
        5,
        figsize=(figsize_per_log[0], figsize_per_log[1] * len(available)),
        squeeze=False,
    )

    for row, (run_idx, log_key, run, log) in enumerate(available):
        run_label = (
            short_labels[run_idx]
            if short_labels is not None
            else run.get("name", f"run {run_idx}")
        )

        top_rates, bottom_rates = _final_tax_rates(log, env_obj, top_first=top_first)
        top_step_y = np.append(top_rates, top_rates[-1])
        bottom_step_y = np.append(bottom_rates, bottom_rates[-1])

        df_agents = _agent_behavior_table(log)
        df_agents = df_agents.sort_values(["planner_region", "skill_build_payment", "agent"])
        if behavior_metric not in df_agents.columns:
            raise ValueError(
                f"Unknown behavior_metric {behavior_metric!r}. "
                f"Available metrics include: {sorted(df_agents.columns)}"
            )

        # 1. p_top final tax
        ax = axes[row, 0]
        ax.step(step_x, top_step_y, where="post", color=region_colors["top"], linewidth=2)
        ax.set_title(f"{run_label}, log {log_key}\np_top final tax")
        ax.set_ylim(0, max(1.0, top_step_y.max() * 1.05))
        ax.grid(True, alpha=0.3)

        # 2. p_bottom final tax
        ax = axes[row, 1]
        ax.step(step_x, bottom_step_y, where="post", color=region_colors["bottom"], linewidth=2)
        ax.set_title("p_bottom final tax")
        ax.set_ylim(0, max(1.0, bottom_step_y.max() * 1.05))
        ax.grid(True, alpha=0.3)

        # 3. behavior bars
        ax = axes[row, 2]
        fill = [region_colors.get(r, "gray") for r in df_agents["majority_location_region"]]
        edge = [edge_colors.get(r, "black") for r in df_agents["planner_region"]]

        labels = [
            f"{int(a)}\n{st:.0%}T\nsw={int(sw)}"
            for a, st, sw in zip(
                df_agents["agent"],
                df_agents["share_top"],
                df_agents["region_switches"],
            )
        ]

        bars = ax.bar(labels, df_agents[behavior_metric], color=fill, alpha=0.85)
        for bar, ec in zip(bars, edge):
            bar.set_edgecolor(ec)
            bar.set_linewidth(2.5)

        ax.set_title(f"{behavior_metric}\nfill=majority loc, edge=planner")
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # 4. coin/labor scatter
        ax = axes[row, 3]
        fill = [region_colors.get(r, "gray") for r in df_agents["majority_location_region"]]
        edge = [edge_colors.get(r, "black") for r in df_agents["planner_region"]]

        skill = df_agents["skill_build_payment"]
        if skill.notna().any() and skill.max() > skill.min():
            sizes = 45 + 110 * ((skill - skill.min()) / (skill.max() - skill.min()))
        else:
            sizes = np.full(len(df_agents), 75)

        ax.scatter(
            df_agents["final_labor"],
            df_agents["final_coin"],
            c=fill,
            edgecolor=edge,
            s=sizes,
            linewidth=2,
            alpha=0.85,
        )

        for _, r in df_agents.iterrows():
            ax.annotate(str(int(r["agent"])), (r["final_labor"], r["final_coin"]), fontsize=8)

        ax.set_title("Final coin vs labor")
        ax.set_xlabel("final labor")
        ax.set_ylabel("final coin")
        ax.grid(True, alpha=0.3)

        # 5. region timeline heatmap
        ax = axes[row, 4]
        aids = df_agents["agent"].astype(int).tolist()
        waterline = _infer_waterline(log)

        region_mat = []
        for aid in aids:
            seq = _region_sequence(log, aid, waterline=waterline)
            region_mat.append([0 if r == "top" else 1 for r in seq])

        region_mat = np.asarray(region_mat)
        ax.imshow(region_mat, aspect="auto", interpolation="nearest", cmap=plt.cm.get_cmap("tab10", 2))
        ax.set_yticks(np.arange(len(aids)))
        ax.set_yticklabels([str(a) for a in aids], fontsize=8)
        ax.set_title("Location over time")
        travel_by_agent = _travel_events_by_agent(log)
        for y, aid in enumerate(aids):
            ts = [
                int(event["t"])
                for event in travel_by_agent.get(aid, [])
                if "t" in event
            ]
            if ts:
                ax.scatter(
                    ts,
                    np.full(len(ts), y),
                    marker="|",
                    s=90,
                    color="crimson",
                    linewidths=1.6,
                    alpha=0.95,
                )

        states = log.get("states", [])
        date_ticks = [
            (i, _date_from_state(state))
            for i, state in enumerate(states)
            if _date_from_state(state) is not None
        ]
        if date_ticks:
            tick_idx = np.linspace(0, len(date_ticks) - 1, min(5, len(date_ticks))).round().astype(int)
            ax.set_xticks([date_ticks[i][0] for i in tick_idx])
            ax.set_xticklabels([str(date_ticks[i][1]) for i in tick_idx], rotation=30, ha="right")
            ax.set_xlabel("date")
        else:
            ax.set_xlabel("timestep (dates unavailable)")
        ax.set_ylabel("agent")

        print(f"{run_label}, log {log_key}")
        print("planner_region counts:")
        print(df_agents["planner_region"].value_counts(dropna=False).to_string())
        print("majority_location_region counts:")
        print(df_agents["majority_location_region"].value_counts(dropna=False).to_string())
        print("final_location_region counts:")
        print(df_agents["final_location_region"].value_counts(dropna=False).to_string())
        print("travel_events by agent:")
        print(df_agents.set_index("agent")["travel_events"].to_string())
        print()

    legend_handles = [
        Patch(facecolor=region_colors["top"], label="fill: mostly top"),
        Patch(facecolor=region_colors["bottom"], label="fill: mostly bottom"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=edge_colors["top"], markeredgewidth=2.5, label="edge: p_top assigned"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=edge_colors["bottom"], markeredgewidth=2.5, label="edge: p_bottom assigned"),
        Line2D([0], [0], marker="|", color="crimson", linestyle="None",
               markersize=12, markeredgewidth=1.8, label="travel event"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=True)
    fig.subplots_adjust(bottom=0.12)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    return fig


def plot_bracket_counts_for_log(
    log,
    brackets,
    period=100,
    figsize=(13, 5),
):
    import numpy as np
    import pandas as pd

    log_items = _dense_log_items(log)
    if not log_items:
        raise ValueError("No dense logs found. Pass a dense log, dense_logs, or a run dict.")

    income_frames = []
    count_frames = []
    labels = None
    for rollout_id, dense_log in log_items:
        df_one, counts_one, labels_one = _income_bracket_counts(
            dense_log,
            brackets,
            period=period,
        )
        if labels is None:
            labels = labels_one
        if not df_one.empty:
            df_one = df_one.copy()
            df_one["rollout_id"] = rollout_id
            income_frames.append(df_one)
        if not counts_one.empty:
            counts_one = counts_one.copy()
            counts_one["rollout_id"] = rollout_id
            count_frames.append(counts_one)

    df_income = pd.concat(income_frames, ignore_index=True) if income_frames else pd.DataFrame()
    raw_counts = pd.concat(count_frames, ignore_index=True) if count_frames else pd.DataFrame()
    if raw_counts.empty:
        raise ValueError("No income bracket counts could be constructed.")

    if len(log_items) > 1:
        full_index = pd.MultiIndex.from_product(
            [
                raw_counts["rollout_id"].unique(),
                sorted(raw_counts["tax_day_number"].unique()),
                ["top", "bottom"],
                labels,
            ],
            names=["rollout_id", "tax_day_number", "planner_region", "tax_bracket"],
        )
        counts_complete = (
            raw_counts
            .set_index(["rollout_id", "tax_day_number", "planner_region", "tax_bracket"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )
        counts = (
            counts_complete
            .groupby(["tax_day_number", "planner_region", "tax_bracket"], as_index=False, observed=False)
            .agg(
                n_agents=("n_agents", "mean"),
                n_agents_std=("n_agents", "std"),
                n_dense_logs=("n_agents", "count"),
            )
        )
        counts["n_agents_std"] = counts["n_agents_std"].fillna(0.0)
    else:
        counts = raw_counts.copy()
        counts["n_agents_std"] = 0.0
        counts["n_dense_logs"] = 1

    date_by_period = (
        df_income.dropna(subset=["date"])
        .drop_duplicates("tax_day_number")
        .set_index("tax_day_number")["date"]
        if "date" in df_income.columns
        else pd.Series(dtype=object)
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
    )

    im = None

    for ax, region in zip(axes, ["top", "bottom"]):
        pivot = (
            counts[counts["planner_region"] == region]
            .pivot(index="tax_bracket", columns="tax_day_number", values="n_agents")
            .reindex(labels)
            .fillna(0)
        )

        im = ax.imshow(
            pivot.values.astype(float),
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
        )

        title_suffix = "average" if len(log_items) > 1 else "agents"
        ax.set_title(f"{region} planner: {title_suffix} per income bracket")
        ax.set_xlabel("tax day")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xticks(np.arange(len(pivot.columns)))
        if len(date_by_period):
            ax.set_xticklabels([date_by_period.get(c, c) for c in pivot.columns], rotation=30, ha="right")
        else:
            ax.set_xticklabels(pivot.columns)

    axes[0].set_ylabel("income bracket")

    # Colorbar placed outside the right edge
    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.9, pad=0.02)
    cbar.set_label("n agents")

    if len(log_items) > 1:
        cbar.set_label("mean n agents")
    return fig, df_income, counts



def plot_income_and_tax_over_time(
    log,
    env_obj,
    brackets,
    period=100,
    top_first=True,
    figsize=(14, 10),
):
    df_income = _period_income_table(log, period=period)
    ptop_rates, pbot_rates = _tax_rate_matrices(log, env_obj, top_first=top_first)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=False)

    # 1. Agent incomes over tax periods
    ax = axes[0]
    for aid, dfa in df_income.groupby("agent"):
        planner_region = dfa["planner_region"].iloc[0]
        color = "#1f77b4" if planner_region == "top" else "#ff7f0e"
        x, x_label = _x_values_and_label(dfa)
        ax.plot(x, dfa["income"], marker="o", linewidth=1.5, color=color, alpha=0.8, label=f"agent {aid}")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Agent income by tax period")
    ax.set_xlabel(x_label)
    ax.set_ylabel("income")
    ax.grid(True, alpha=0.3)

    # 2. Mean income by planner assignment
    ax = axes[1]
    mean_income = (
        df_income.groupby(["tax_day_number", "planner_region"])["income"]
        .mean()
        .reset_index()
    )
    date_by_period = (
        df_income.dropna(subset=["date"])
        .drop_duplicates("tax_day_number")
        .set_index("tax_day_number")["date"]
        if "date" in df_income.columns
        else pd.Series(dtype=object)
    )

    for region, color in [("top", "#1f77b4"), ("bottom", "#ff7f0e")]:
        dfr = mean_income[mean_income["planner_region"] == region]
        if len(date_by_period):
            x = pd.to_datetime(dfr["tax_day_number"].map(date_by_period))
            x_label = "date"
        else:
            x = dfr["tax_day_number"]
            x_label = "tax period (dates unavailable in dense log)"
        ax.plot(x, dfr["income"], marker="o", linewidth=2.5, color=color, label=region)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Mean income by fixed planner assignment")
    ax.set_xlabel(x_label)
    ax.set_ylabel("mean income")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. p_top tax rates over planner decisions
    ax = axes[2]
    x_top = np.arange(1, ptop_rates.shape[0] + 1)
    for b in range(ptop_rates.shape[1]):
        label = f"b{b}: {brackets[b]:g}+"
        ax.plot(x_top, ptop_rates[:, b], linewidth=1.8, label=label)

    ax.set_title("p_top tax rates over planner decisions")
    ax.set_ylabel("tax rate")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)

    # 4. p_bottom tax rates over planner decisions
    ax = axes[3]
    x_bot = np.arange(1, pbot_rates.shape[0] + 1)
    for b in range(pbot_rates.shape[1]):
        label = f"b{b}: {brackets[b]:g}+"
        ax.plot(x_bot, pbot_rates[:, b], linewidth=1.8, label=label)

    ax.set_title("p_bottom tax rates over planner decisions")
    ax.set_xlabel("planner decision index")
    ax.set_ylabel("tax rate")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)

    fig.tight_layout()
    return fig, df_income

def _equality_from_values(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) <= 1:
        return 1.0

    values = np.maximum(values, 0.0)
    mean = np.mean(values)

    if mean <= 0:
        return 1.0

    diffs = np.abs(values[:, None] - values[None, :])
    gini = np.mean(diffs) / (2.0 * mean)
    return float(1.0 - gini)


def _period_swf_table(log, period=100):
    states = log["states"]
    aids = _numeric_agent_ids(log)
    waterline = _infer_waterline(log)

    planner_rewards = log.get("planner_rewards", {})
    rewards_top = np.asarray(planner_rewards.get("p_top", []), dtype=float)
    rewards_bottom = np.asarray(planner_rewards.get("p_bottom", []), dtype=float)

    tax_days = list(range(period - 1, len(states), period))
    rows = []

    for tax_day_number, t in enumerate(tax_days, start=1):
        t_start = 0 if tax_day_number == 1 else tax_days[tax_day_number - 2] + 1
        t_end = t + 1

        for region, planner_id, reward_arr in [
            ("top", "p_top", rewards_top),
            ("bottom", "p_bottom", rewards_bottom),
        ]:
            assigned = [
                aid for aid in aids
                if _planner_region_from_initial_state(log, aid, waterline=waterline) == region
            ]

            state = states[t]
            coins = np.array([_coin(state[str(aid)]) for aid in assigned], dtype=float)

            production = float(np.sum(coins)) if len(coins) else np.nan
            equality = _equality_from_values(coins) if len(coins) else np.nan
            swf_proxy = production * equality if np.isfinite(production) and np.isfinite(equality) else np.nan

            reward_slice = reward_arr[t_start:min(t_end, len(reward_arr))]
            reward_slice = reward_slice[np.isfinite(reward_slice)]

            rows.append({
                "tax_day_number": tax_day_number,
                "planner_region": region,
                "planner_id": planner_id,
                "production": production,
                "equality": equality,
                "swf_proxy": swf_proxy,
                "planner_reward_sum": float(np.sum(reward_slice)) if len(reward_slice) else np.nan,
                "planner_reward_mean": float(np.mean(reward_slice)) if len(reward_slice) else np.nan,
                "n_agents": len(assigned),
            })

    return pd.DataFrame(rows)



def plot_tax_bracket_snapshots_compact(
    log,
    env_obj,
    brackets,
    period=100,
    n_snapshots=10,
    top_first=True,
    figsize=None,
):
    df_income, counts, labels = _income_bracket_counts(log, brackets, period=period)
    df_swf = _period_swf_table(log, period=period)

    ptop_rates, pbot_rates = _tax_rate_matrices(log, env_obj, top_first=top_first)

    planner_rewards = log.get("planner_rewards", {})
    rewards_top = np.asarray(planner_rewards.get("p_top", []), dtype=float)
    rewards_bottom = np.asarray(planner_rewards.get("p_bottom", []), dtype=float)

    tax_days = sorted(df_income["tax_day_number"].unique())

    if len(tax_days) <= n_snapshots:
        chosen_days = tax_days
    else:
        idx = np.linspace(0, len(tax_days) - 1, n_snapshots).round().astype(int)
        chosen_days = [tax_days[i] for i in idx]

    n_cols = 5
    n_rows = 4

    if figsize is None:
        figsize = (2.8 * n_cols, 10.8)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.asarray(axes).reshape(n_rows, n_cols)

    bracket_x = np.arange(len(labels))
    max_count = max(1, int(counts["n_agents"].max()))

    configs = [
        ("top", "p_top", ptop_rates, rewards_top, "#1f77b4", 0),
        ("bottom", "p_bottom", pbot_rates, rewards_bottom, "#ff7f0e", 2),
    ]

    def tax_day_to_decision_idx(tax_day, rate_matrix):
        if rate_matrix.shape[0] == len(tax_days):
            return min(tax_day - 1, rate_matrix.shape[0] - 1)

        if len(tax_days) == 1 or rate_matrix.shape[0] == 1:
            return 0

        frac = (tax_day - tax_days[0]) / (tax_days[-1] - tax_days[0])
        return int(np.clip(round(frac * (rate_matrix.shape[0] - 1)), 0, rate_matrix.shape[0] - 1))

    def reward_at_decision(decision_idx, reward_arr):
        if len(reward_arr) == 0:
            return np.nan
        if len(reward_arr) == ptop_rates.shape[0] or len(reward_arr) == pbot_rates.shape[0]:
            return reward_arr[min(decision_idx, len(reward_arr) - 1)]
        if len(reward_arr) == len(tax_days):
            return reward_arr[min(decision_idx, len(reward_arr) - 1)]
        if len(reward_arr) == 1:
            return reward_arr[0]

        frac = decision_idx / max(1, ptop_rates.shape[0] - 1)
        ridx = int(np.clip(round(frac * (len(reward_arr) - 1)), 0, len(reward_arr) - 1))
        return reward_arr[ridx]

    for ax in axes.flat:
        ax.set_visible(False)

    for region, planner_id, rate_matrix, reward_arr, color, base_row in configs:
        tax_ymax = max(1.0, float(np.nanmax(rate_matrix)) * 1.05)

        for snapshot_idx, tax_day in enumerate(chosen_days[: n_rows // 2 * n_cols]):
            row = base_row + snapshot_idx // n_cols
            col = snapshot_idx % n_cols
            ax = axes[row, col]
            ax.set_visible(True)

            day_counts = (
                counts[
                    (counts["planner_region"] == region)
                    & (counts["tax_day_number"] == tax_day)
                ]
                .set_index("tax_bracket")["n_agents"]
                .reindex(labels)
                .fillna(0)
            )

            decision_idx = tax_day_to_decision_idx(tax_day, rate_matrix)
            rates = rate_matrix[decision_idx]

            scaled_rates = rates / tax_ymax * max_count

            ax.bar(
                bracket_x,
                day_counts.values,
                color=color,
                alpha=0.35,
                edgecolor=color,
                linewidth=1.1,
            )

            ax.plot(
                bracket_x,
                scaled_rates,
                color=color,
                marker="o",
                linewidth=2.2,
            )

            swf_row = df_swf[
                (df_swf["tax_day_number"] == tax_day)
                & (df_swf["planner_region"] == region)
            ]

            if len(swf_row):
                sr = swf_row.iloc[0]
                production = sr["production"]
                equality = sr["equality"]
            else:
                production = np.nan
                equality = np.nan
                planner_reward = np.nan

            if len(swf_row):
                planner_reward = sr["planner_reward_sum"]


            txt = (
                f"{planner_id} R sum: {planner_reward:.3g}\n"
                f"prod: {production:.3g}\n"
                f"eq: {equality:.3f}"
            )


            ax.text(
                0.03,
                0.95,
                txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
            )

            ax.set_ylim(0, max_count + 0.5)
            ax.set_yticks(range(0, max_count + 1))

            if col == 0:
                ax.set_ylabel("n agents")
            else:
                ax.tick_params(axis="y", labelleft=False)

            is_last_visible_col = col == min(n_cols, len(chosen_days) - (row - base_row) * n_cols) - 1
            if is_last_visible_col:
                axr = ax.secondary_yaxis(
                    "right",
                    functions=(
                        lambda y: y / max_count * tax_ymax,
                        lambda y: y / tax_ymax * max_count,
                    ),
                )
                axr.set_ylabel("tax rate")

            ax.set_title(f"{planner_id}\ntax day {tax_day}", fontsize=10)
            ax.set_xticks(bracket_x)
            ax.set_xticklabels([f"b{i}" for i in range(len(labels))], fontsize=8)
            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Tax-Day Snapshots: Bracket Counts, Tax Rates, Planner Reward, Production, Equality",
        fontsize=14,
    )

    return fig, df_income, counts, df_swf


def plot_tax_bracket_snapshots_compact_average(
    run,
    env_obj,
    brackets,
    period=100,
    n_snapshots=10,
    top_first=True,
    show_std=True,
    figsize=None,
):
    """Average compact tax bracket snapshots across all dense logs in a run."""
    log_items = _dense_log_items(run)
    if not log_items:
        raise ValueError("No dense logs found. Pass a run dict, dense_logs, or a dense log.")

    income_frames = []
    count_frames = []
    swf_frames = []
    rate_frames = []
    labels = None

    for rollout_id, dense_log in log_items:
        df_income_one, counts_one, labels_one = _income_bracket_counts(
            dense_log,
            brackets,
            period=period,
        )
        if labels is None:
            labels = labels_one

        df_swf_one = _period_swf_table(dense_log, period=period)
        ptop_rates, pbot_rates = _tax_rate_matrices(dense_log, env_obj, top_first=top_first)
        tax_days = sorted(df_income_one["tax_day_number"].unique())

        if not df_income_one.empty:
            df_income_one = df_income_one.copy()
            df_income_one["rollout_id"] = rollout_id
            income_frames.append(df_income_one)

        if not counts_one.empty:
            counts_one = counts_one.copy()
            counts_one["rollout_id"] = rollout_id
            count_frames.append(counts_one)

        if not df_swf_one.empty:
            df_swf_one = df_swf_one.copy()
            df_swf_one["rollout_id"] = rollout_id
            swf_frames.append(df_swf_one)

        def tax_day_to_decision_idx(tax_day, rate_matrix):
            if rate_matrix.shape[0] == len(tax_days):
                return min(int(tax_day) - 1, rate_matrix.shape[0] - 1)
            if len(tax_days) == 1 or rate_matrix.shape[0] == 1:
                return 0
            frac = (tax_day - tax_days[0]) / (tax_days[-1] - tax_days[0])
            return int(np.clip(round(frac * (rate_matrix.shape[0] - 1)), 0, rate_matrix.shape[0] - 1))

        for region, planner_id, rate_matrix in [
            ("top", "p_top", ptop_rates),
            ("bottom", "p_bottom", pbot_rates),
        ]:
            for tax_day in tax_days:
                decision_idx = tax_day_to_decision_idx(tax_day, rate_matrix)
                for bracket_idx, rate in enumerate(rate_matrix[decision_idx]):
                    rate_frames.append({
                        "rollout_id": rollout_id,
                        "tax_day_number": tax_day,
                        "planner_region": region,
                        "planner_id": planner_id,
                        "bracket_idx": bracket_idx,
                        "rate": float(rate),
                    })

    if labels is None:
        raise ValueError("No tax bracket labels could be constructed.")

    df_income = pd.concat(income_frames, ignore_index=True) if income_frames else pd.DataFrame()
    raw_counts = pd.concat(count_frames, ignore_index=True) if count_frames else pd.DataFrame()
    raw_swf = pd.concat(swf_frames, ignore_index=True) if swf_frames else pd.DataFrame()
    raw_rates = pd.DataFrame(rate_frames)

    if raw_counts.empty or raw_rates.empty:
        raise ValueError("No bracket counts or tax rates could be constructed.")

    rollout_ids = [rollout_id for rollout_id, _ in log_items]
    tax_days = sorted(raw_counts["tax_day_number"].dropna().unique())
    full_count_index = pd.MultiIndex.from_product(
        [rollout_ids, tax_days, ["top", "bottom"], labels],
        names=["rollout_id", "tax_day_number", "planner_region", "tax_bracket"],
    )
    counts_complete = (
        raw_counts
        .set_index(["rollout_id", "tax_day_number", "planner_region", "tax_bracket"])
        .reindex(full_count_index, fill_value=0)
        .reset_index()
    )
    counts = (
        counts_complete
        .groupby(["tax_day_number", "planner_region", "tax_bracket"], as_index=False, observed=False)
        .agg(
            n_agents=("n_agents", "mean"),
            n_agents_std=("n_agents", "std"),
            n_dense_logs=("n_agents", "count"),
        )
    )
    counts["n_agents_std"] = counts["n_agents_std"].fillna(0.0)

    rate_summary = (
        raw_rates
        .groupby(["tax_day_number", "planner_region", "planner_id", "bracket_idx"], as_index=False)
        .agg(
            rate=("rate", "mean"),
            rate_std=("rate", "std"),
            n_dense_logs=("rate", "count"),
        )
    )
    rate_summary["rate_std"] = rate_summary["rate_std"].fillna(0.0)

    if raw_swf.empty:
        df_swf = pd.DataFrame()
    else:
        df_swf = (
            raw_swf
            .groupby(["tax_day_number", "planner_region", "planner_id"], as_index=False)
            .agg(
                production=("production", "mean"),
                production_std=("production", "std"),
                equality=("equality", "mean"),
                equality_std=("equality", "std"),
                swf_proxy=("swf_proxy", "mean"),
                swf_proxy_std=("swf_proxy", "std"),
                planner_reward_sum=("planner_reward_sum", "mean"),
                planner_reward_sum_std=("planner_reward_sum", "std"),
                planner_reward_mean=("planner_reward_mean", "mean"),
                planner_reward_mean_std=("planner_reward_mean", "std"),
                n_dense_logs=("planner_reward_sum", "count"),
            )
        )
        for col in [c for c in df_swf.columns if c.endswith("_std")]:
            df_swf[col] = df_swf[col].fillna(0.0)

    if len(tax_days) <= n_snapshots:
        chosen_days = tax_days
    else:
        idx = np.linspace(0, len(tax_days) - 1, n_snapshots).round().astype(int)
        chosen_days = [tax_days[i] for i in idx]

    n_cols = 5
    n_rows = 4
    if figsize is None:
        figsize = (2.8 * n_cols, 10.8)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    bracket_x = np.arange(len(labels))
    max_count = max(1.0, float((counts["n_agents"] + counts["n_agents_std"]).max()))

    configs = [
        ("top", "p_top", "#1f77b4", 0),
        ("bottom", "p_bottom", "#d62728", 2),
    ]

    for ax in axes.flat:
        ax.set_visible(False)

    for region, planner_id, color, base_row in configs:
        region_rates = rate_summary[rate_summary["planner_region"] == region]
        tax_ymax = max(1.0, float((region_rates["rate"] + region_rates["rate_std"]).max()) * 1.05)

        for snapshot_idx, tax_day in enumerate(chosen_days[: n_rows // 2 * n_cols]):
            row = base_row + snapshot_idx // n_cols
            col = snapshot_idx % n_cols
            ax = axes[row, col]
            ax.set_visible(True)

            day_counts = (
                counts[
                    (counts["planner_region"] == region)
                    & (counts["tax_day_number"] == tax_day)
                ]
                .set_index("tax_bracket")
                .reindex(labels)
            )
            count_vals = day_counts["n_agents"].fillna(0.0).to_numpy(dtype=float)
            count_err = day_counts["n_agents_std"].fillna(0.0).to_numpy(dtype=float)

            day_rates = (
                rate_summary[
                    (rate_summary["planner_region"] == region)
                    & (rate_summary["tax_day_number"] == tax_day)
                ]
                .set_index("bracket_idx")
                .reindex(range(len(labels)))
            )
            rates = day_rates["rate"].fillna(0.0).to_numpy(dtype=float)
            rate_err = day_rates["rate_std"].fillna(0.0).to_numpy(dtype=float)
            scaled_rates = rates / tax_ymax * max_count
            scaled_rate_err = rate_err / tax_ymax * max_count

            ax.bar(
                bracket_x,
                count_vals,
                yerr=count_err if show_std else None,
                color=color,
                alpha=0.35,
                edgecolor=color,
                linewidth=1.1,
                error_kw=dict(ecolor=color, elinewidth=0.8, capsize=2, alpha=0.75),
            )

            ax.plot(
                bracket_x,
                scaled_rates,
                color=color,
                marker="o",
                linewidth=2.2,
            )
            if show_std:
                ax.fill_between(
                    bracket_x,
                    np.maximum(0.0, scaled_rates - scaled_rate_err),
                    scaled_rates + scaled_rate_err,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )

            swf_row = df_swf[
                (df_swf["tax_day_number"] == tax_day)
                & (df_swf["planner_region"] == region)
            ] if not df_swf.empty else df_swf

            if len(swf_row):
                sr = swf_row.iloc[0]
                reward_text = f"{sr['planner_reward_sum']:.3g}"
                prod_text = f"{sr['production']:.3g}"
                eq_text = f"{sr['equality']:.3f}"
                if show_std:
                    reward_text += f" +/- {sr['planner_reward_sum_std']:.2g}"
                    prod_text += f" +/- {sr['production_std']:.2g}"
                    eq_text += f" +/- {sr['equality_std']:.2g}"
            else:
                reward_text = prod_text = eq_text = "n/a"

            txt = (
                f"{planner_id} R sum: {reward_text}\n"
                f"prod: {prod_text}\n"
                f"eq: {eq_text}"
            )
            ax.text(
                0.03,
                0.95,
                txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
            )

            ax.set_ylim(0, max_count + 0.5)
            ax.set_yticks(range(0, int(np.ceil(max_count)) + 1))

            if col == 0:
                ax.set_ylabel("mean n agents")
            else:
                ax.tick_params(axis="y", labelleft=False)

            is_last_visible_col = col == min(n_cols, len(chosen_days) - (row - base_row) * n_cols) - 1
            if is_last_visible_col:
                axr = ax.secondary_yaxis(
                    "right",
                    functions=(
                        lambda y: y / max_count * tax_ymax,
                        lambda y: y / tax_ymax * max_count,
                    ),
                )
                axr.set_ylabel("tax rate")

            ax.set_title(f"{planner_id}\ntax day {int(tax_day)}", fontsize=10)
            ax.set_xticks(bracket_x)
            ax.set_xticklabels([f"b{i}" for i in range(len(labels))], fontsize=8)
            ax.grid(True, axis="y", alpha=0.25)

    suffix = "mean +/- SD" if show_std else "mean"
    fig.suptitle(
        f"Average Tax-Day Snapshots Across Dense Logs ({suffix})",
        fontsize=14,
    )

    return fig, df_income, counts, df_swf


def tax_bracket_correlation_outcome_table_average(
    run,
    env_obj,
    brackets,
    period=100,
    top_first=True,
    exclude_highest_bracket=True,
    min_tax_period=1,
):
    """
    Measure how tax rates align with occupied income brackets by tax period.

    The correlation is Pearson corr(tax rate by bracket, number of agents in
    bracket), computed separately for each rollout, tax period, and planner
    region. Positive values mean higher rates are assigned to more populated
    brackets; negative values mean higher rates are assigned to less populated
    brackets. Outcomes are the same production/equality values used by the
    compact tax-bracket snapshot plots.
    """
    import numpy as np
    import pandas as pd

    log_items = _dense_log_items(run)
    if not log_items:
        raise ValueError("No dense logs found. Pass a run dict, dense_logs, or a dense log.")

    labels = None
    rows = []

    def pearson_corr(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if len(x) < 2:
            return np.nan
        if float(np.nanstd(x)) <= 1e-12 or float(np.nanstd(y)) <= 1e-12:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    for rollout_id, dense_log in log_items:
        _, counts_one, labels_one = _income_bracket_counts(
            dense_log,
            brackets,
            period=period,
        )
        if labels is None:
            labels = labels_one

        df_swf_one = _period_swf_table(dense_log, period=period)
        ptop_rates, pbot_rates = _tax_rate_matrices(dense_log, env_obj, top_first=top_first)
        tax_days = sorted(counts_one["tax_day_number"].dropna().unique())

        if not tax_days:
            continue

        def tax_day_to_decision_idx(tax_day, rate_matrix):
            if rate_matrix.shape[0] == len(tax_days):
                return min(int(tax_day) - 1, rate_matrix.shape[0] - 1)
            if len(tax_days) == 1 or rate_matrix.shape[0] == 1:
                return 0
            frac = (tax_day - tax_days[0]) / (tax_days[-1] - tax_days[0])
            return int(np.clip(round(frac * (rate_matrix.shape[0] - 1)), 0, rate_matrix.shape[0] - 1))

        for region, planner_id, rate_matrix in [
            ("top", "p_top", ptop_rates),
            ("bottom", "p_bottom", pbot_rates),
        ]:
            for tax_day in tax_days:
                if int(tax_day) < int(min_tax_period):
                    continue
                decision_idx = tax_day_to_decision_idx(tax_day, rate_matrix)
                rates = np.asarray(rate_matrix[decision_idx], dtype=float)
                n_brackets = min(len(labels), len(rates))
                include_idx = np.arange(n_brackets)
                if exclude_highest_bracket and len(include_idx) > 1:
                    include_idx = include_idx[:-1]

                day_counts = (
                    counts_one[
                        (counts_one["planner_region"] == region)
                        & (counts_one["tax_day_number"] == tax_day)
                    ]
                    .set_index("tax_bracket")["n_agents"]
                    .reindex(labels[:n_brackets])
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )

                corr = pearson_corr(rates[include_idx], day_counts[include_idx])
                swf_row = df_swf_one[
                    (df_swf_one["tax_day_number"] == tax_day)
                    & (df_swf_one["planner_region"] == region)
                ]
                if len(swf_row):
                    sr = swf_row.iloc[0]
                    production = float(sr["production"])
                    equality = float(sr["equality"])
                    planner_reward_sum = float(sr["planner_reward_sum"])
                else:
                    production = equality = planner_reward_sum = np.nan

                rows.append({
                    "rollout_id": rollout_id,
                    "tax_day_number": int(tax_day),
                    "planner_region": region,
                    "planner_id": planner_id,
                    "tax_count_corr": corr,
                    "mean_tax_rate_included": float(np.nanmean(rates[include_idx])) if len(include_idx) else np.nan,
                    "mean_agents_per_bracket_included": float(np.nanmean(day_counts[include_idx])) if len(include_idx) else np.nan,
                    "total_agents_included": float(np.nansum(day_counts[include_idx])) if len(include_idx) else np.nan,
                    "production": production,
                    "equality": equality,
                    "planner_reward_sum": planner_reward_sum,
                    "n_brackets_included": int(len(include_idx)),
                    "highest_bracket_excluded": bool(exclude_highest_bracket),
                })

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError("No tax-bracket correlation rows could be constructed.")

    summary = (
        raw_df
        .groupby(["tax_day_number", "planner_region", "planner_id"], as_index=False)
        .agg(
            tax_count_corr=("tax_count_corr", "mean"),
            tax_count_corr_std=("tax_count_corr", "std"),
            production=("production", "mean"),
            production_std=("production", "std"),
            equality=("equality", "mean"),
            equality_std=("equality", "std"),
            planner_reward_sum=("planner_reward_sum", "mean"),
            planner_reward_sum_std=("planner_reward_sum", "std"),
            n_dense_logs=("tax_count_corr", "count"),
            n_brackets_included=("n_brackets_included", "max"),
        )
    )
    for col in [c for c in summary.columns if c.endswith("_std")]:
        summary[col] = summary[col].fillna(0.0)

    return summary, raw_df


def plot_tax_bracket_correlation_outcomes_average(
    run,
    env_obj,
    brackets,
    period=100,
    top_first=True,
    exclude_highest_bracket=True,
    min_tax_period=1,
    show_std=True,
    figsize=(14, 10),
):
    """
    Plot whether tax-bracket alignment correlates with equality/production.

    Returns
    -------
    fig, summary_df, raw_df
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    summary, raw_df = tax_bracket_correlation_outcome_table_average(
        run,
        env_obj,
        brackets,
        period=period,
        top_first=top_first,
        exclude_highest_bracket=exclude_highest_bracket,
        min_tax_period=min_tax_period,
    )

    colors = {"top": "#1f77b4", "bottom": "#d62728"}
    if figsize == (14, 10):
        figsize = (18, 5.5)
    fig, (ax_corr, ax_eq, ax_prod) = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    for region in ["top", "bottom"]:
        dfr = summary[summary["planner_region"] == region].sort_values("tax_day_number")
        if dfr.empty:
            continue
        color = colors[region]
        x = dfr["tax_day_number"].to_numpy(dtype=float)
        corr = dfr["tax_count_corr"].to_numpy(dtype=float)
        corr_std = dfr["tax_count_corr_std"].to_numpy(dtype=float)

        ax_corr.plot(x, corr, marker="o", linewidth=2.1, color=color, label=region)
        if show_std:
            ax_corr.fill_between(
                x,
                corr - corr_std,
                corr + corr_std,
                color=color,
                alpha=0.14,
                linewidth=0,
            )

        ax_eq.scatter(
            corr,
            dfr["equality"],
            s=52,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            label=region,
        )
        ax_prod.scatter(
            corr,
            dfr["production"],
            s=52,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            label=region,
        )
        for _, row in dfr.iterrows():
            ax_eq.annotate(
                str(int(row["tax_day_number"])),
                (row["tax_count_corr"], row["equality"]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
                color=color,
            )
            ax_prod.annotate(
                str(int(row["tax_day_number"])),
                (row["tax_count_corr"], row["production"]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
                color=color,
            )

    for ax, metric in [(ax_eq, "equality"), (ax_prod, "production")]:
        trend_df = summary[["tax_count_corr", metric]].dropna()
        if len(trend_df) >= 2 and trend_df["tax_count_corr"].nunique() >= 2:
            xfit = trend_df["tax_count_corr"].to_numpy(dtype=float)
            yfit = trend_df[metric].to_numpy(dtype=float)
            slope, intercept = np.polyfit(xfit, yfit, 1)
            xs = np.linspace(float(np.nanmin(xfit)), float(np.nanmax(xfit)), 100)
            ax.plot(xs, slope * xs + intercept, color="0.2", linewidth=1.8, linestyle="--")

    ax_corr.axhline(0, color="0.25", linewidth=1.0, linestyle="--")
    ax_corr.set_title("Tax-Count Correlation by Tax Period")
    ax_corr.set_xlabel("tax period")
    ax_corr.set_ylabel("corr(tax rate, n agents)")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.grid(True, alpha=0.25)
    ax_corr.legend(frameon=True)

    ax_eq.axvline(0, color="0.25", linewidth=1.0, linestyle="--")
    ax_eq.set_title("Equality vs Tax-Count Correlation")
    ax_eq.set_xlabel("corr(tax rate, n agents)")
    ax_eq.set_ylabel("equality")
    ax_eq.grid(True, alpha=0.25)
    ax_eq.legend(frameon=True)

    ax_prod.axvline(0, color="0.25", linewidth=1.0, linestyle="--")
    ax_prod.set_title("Production vs Tax-Count Correlation")
    ax_prod.set_xlabel("corr(tax rate, n agents)")
    ax_prod.set_ylabel("production")
    ax_prod.grid(True, alpha=0.25)
    ax_prod.legend(frameon=True)

    bracket_note = "excluding highest bracket" if exclude_highest_bracket else "including all brackets"
    period_note = (
        f", tax periods >= {int(min_tax_period)}"
        if int(min_tax_period) > 1
        else ""
    )
    fig.suptitle(
        f"Tax Bracket Alignment vs Equality and Production ({bracket_note}{period_note})",
        fontsize=14,
        fontweight="bold",
    )
    return fig, summary, raw_df


def _extract_tax_policy_from_actions(log, period=100, rate_disc=0.05):
    """
    Return one regional tax schedule per tax period from dense_log["actions"].

    The rollout logs full planner actions at every timestep. At tax-period starts
    those action dicts contain the active regional bracket keys for p_top and
    p_bottom. Planner action 0 is no-op; positive actions are one-based indices
    into the discrete rate grid.
    """
    actions = log.get("actions", [])
    n_periods = int(np.ceil((len(log.get("states", [])) - 1) / float(period)))
    rows = []

    for tax_period in range(1, n_periods + 1):
        t = min((tax_period - 1) * period, max(0, len(actions) - 1))
        action_t = actions[t] if t < len(actions) else {}

        for region, planner_id in [("top", "p_top"), ("bottom", "p_bottom")]:
            planner_action = action_t.get(planner_id, {}) if isinstance(action_t, dict) else {}
            pairs = []
            for key, value in planner_action.items():
                if "TaxIndexBracket" not in str(key):
                    continue
                try:
                    cutoff = float(str(key).split("_")[-1])
                except ValueError:
                    cutoff = float(len(pairs))
                rate = max(0.0, (float(value) - 1.0) * float(rate_disc))
                pairs.append((cutoff, min(1.0, rate)))

            if not pairs:
                continue

            pairs = sorted(pairs, key=lambda x: x[0])
            rates = np.asarray([rate for _, rate in pairs], dtype=float)
            rows.append({
                "tax_period": tax_period,
                "decision_timestep": t,
                "planner_region": region,
                "planner_id": planner_id,
                "tax_schedule": rates,
                "top_marginal_rate": float(rates[-1]),
                "avg_marginal_rate": float(np.mean(rates)),
                "progressivity": float(rates[-1] - rates[0]),
            })

    if rows:
        return pd.DataFrame(rows)

    # Fallback for older logs that only have planner_actions. In the current
    # two-planner rollout format, the active regional half is stored in the last
    # half of each planner vector.
    planner_actions = log.get("planner_actions", {})
    for region, planner_id in [("top", "p_top"), ("bottom", "p_bottom")]:
        arr = np.asarray(planner_actions.get(planner_id, []))
        if arr.ndim != 2 or arr.shape[1] == 0:
            continue
        half = arr.shape[1] // 2
        active = arr[:, half:]
        for idx, row in enumerate(active, start=1):
            rates = np.clip((row.astype(float) - 1.0) * float(rate_disc), 0.0, 1.0)
            rows.append({
                "tax_period": idx,
                "decision_timestep": (idx - 1) * period,
                "planner_region": region,
                "planner_id": planner_id,
                "tax_schedule": rates,
                "top_marginal_rate": float(rates[-1]),
                "avg_marginal_rate": float(np.mean(rates)),
                "progressivity": float(rates[-1] - rates[0]),
            })

    return pd.DataFrame(rows)


def _region_from_loc(loc, waterline):
    row = int(loc[0])
    return "top" if row <= waterline else "bottom"


def _tax_policy_table(log, period=100, rate_disc=0.05):
    tax_df = _extract_tax_policy_from_actions(log, period=period, rate_disc=rate_disc)
    if tax_df.empty:
        raise ValueError("No regional tax policy could be extracted from the dense log.")
    return tax_df


def _period_region_income_table(log, brackets, period=100):
    states = log["states"]
    aids = _numeric_agent_ids(log)
    waterline = _infer_waterline(log)
    brackets = np.asarray(brackets, dtype=float)
    cutoffs = np.r_[-np.inf, brackets[1:], np.inf]
    labels = _bracket_labels_from_cutoffs(cutoffs)

    rows = []
    tax_days = list(range(period - 1, len(states), period))
    prev_idx = 0

    for tax_period, t in enumerate(tax_days, start=1):
        for aid in aids:
            s_prev = states[prev_idx][str(aid)]
            s_now = states[t][str(aid)]
            income = _coin(s_now) - _coin(s_prev)
            physical_region = _location_region_from_state(s_now, waterline=waterline)

            rows.append({
                "tax_period": tax_period,
                "tax_day_number": tax_period,
                "timestep": t,
                "agent": aid,
                "region": physical_region,
                "income": income,
                "coin_end": _coin(s_now),
                "labor_used": float(s_now.get("endogenous", {}).get("Labor", np.nan))
                - float(s_prev.get("endogenous", {}).get("Labor", np.nan)),
                "build_payment": float(states[0][str(aid)].get("build_payment", np.nan)),
            })

        prev_idx = t

    df = pd.DataFrame(rows)
    if not df.empty:
        df["tax_bracket"] = pd.cut(
            df["income"],
            bins=cutoffs,
            labels=labels,
            right=False,
            include_lowest=True,
        )

    counts = (
        df.groupby(["tax_period", "region", "tax_bracket"], observed=False)
        .size()
        .reset_index(name="n_agents")
        if not df.empty
        else pd.DataFrame(columns=["tax_period", "region", "tax_bracket", "n_agents"])
    )

    return df, counts, labels


def _regional_period_outcomes_from_income(df_income):
    rows = []
    for (tax_period, region), dfr in df_income.groupby(["tax_period", "region"]):
        incomes = dfr["income"].to_numpy(dtype=float)
        nonnegative_income = np.maximum(incomes, 0.0)
        production = float(np.sum(nonnegative_income))
        equality = _equality_from_values(nonnegative_income)

        rows.append({
            "tax_period": tax_period,
            "region": region,
            "production": production,
            "equality": equality,
            "n_agents": int(len(dfr)),
            "avg_skill_level": float(np.nanmean(dfr["build_payment"])) if len(dfr) else np.nan,
            "avg_labor_used": float(np.nanmean(dfr["labor_used"])) if len(dfr) else np.nan,
        })

    return pd.DataFrame(rows)


def regional_environment_metrics_by_tax_period(log, period=100):
    """
    Environmental metrics measured over each tax period and physical region.

    Prices are observed trade prices within the trade's logged region. Build
    counts use the build event location. Skill and labor averages use agents'
    physical region at the tax-period boundary.
    """
    states = log["states"]
    aids = _numeric_agent_ids(log)
    waterline = _infer_waterline(log)
    tax_days = list(range(period - 1, len(states), period))

    skill_by_agent = {
        aid: float(states[0][str(aid)].get("build_payment", np.nan))
        for aid in aids
    }

    rows = []
    prev_idx = 0
    for tax_period, t in enumerate(tax_days, start=1):
        start_t = 0 if tax_period == 1 else prev_idx + 1
        end_t = t + 1
        state_t = states[t]
        state_prev = states[prev_idx]

        trades = _events_in_period(log.get("Trade", []), start_t, end_t, implicit_timeline=True)
        builds = _events_in_period(log.get("Build", []), start_t, end_t, implicit_timeline=True)

        for region in ["top", "bottom"]:
            region_aids = [
                aid for aid in aids
                if _location_region_from_state(state_t[str(aid)], waterline=waterline) == region
            ]

            region_trades = [tr for tr in trades if tr.get("region") == region]
            wood_prices = [
                float(tr["price"]) for tr in region_trades
                if tr.get("commodity") == "Wood" and "price" in tr
            ]
            stone_prices = [
                float(tr["price"]) for tr in region_trades
                if tr.get("commodity") == "Stone" and "price" in tr
            ]

            region_builds = [
                b for b in builds
                if "loc" in b and _region_from_loc(b["loc"], waterline) == region
            ]

            labor_used = []
            for aid in region_aids:
                labor_now = float(state_t[str(aid)].get("endogenous", {}).get("Labor", np.nan))
                labor_prev = float(state_prev[str(aid)].get("endogenous", {}).get("Labor", np.nan))
                if np.isfinite(labor_now) and np.isfinite(labor_prev):
                    labor_used.append(labor_now - labor_prev)

            rows.append({
                "tax_period": tax_period,
                "region": region,
                "avg_wood_price": float(np.mean(wood_prices)) if wood_prices else np.nan,
                "avg_stone_price": float(np.mean(stone_prices)) if stone_prices else np.nan,
                "avg_skill_level": float(np.nanmean([skill_by_agent[aid] for aid in region_aids])) if region_aids else np.nan,
                "n_builds": int(len(region_builds)),
                "avg_builds_per_agent": len(region_builds) / max(1, len(region_aids)),
                "avg_labor_used": float(np.mean(labor_used)) if labor_used else np.nan,
                "n_agents": int(len(region_aids)),
                "n_trades": int(len(region_trades)),
            })

        prev_idx = t

    return pd.DataFrame(rows)


def plot_tax_bracket_snapshots_with_environment_table(
    log,
    brackets,
    period=100,
    n_snapshots=10,
    rate_disc=0.05,
    figsize=None,
):
    """
    Mobility-safe tax snapshot figure.

    Each panel shows, for a physical region and tax period:
    - bars: number of agents in each income tax bracket for that period
    - line: current marginal tax schedule chosen at the start of that period
    - text: production and equality computed from period incomes in that region

    The top table gives period-level environmental metrics for the same periods.
    """
    df_income, counts, labels = _period_region_income_table(log, brackets, period=period)
    df_outcomes = _regional_period_outcomes_from_income(df_income)
    df_env = regional_environment_metrics_by_tax_period(log, period=period)
    df_tax = _tax_policy_table(log, period=period, rate_disc=rate_disc)

    tax_periods = sorted(df_income["tax_period"].unique())
    if len(tax_periods) <= n_snapshots:
        chosen_periods = tax_periods
    else:
        idx = np.linspace(0, len(tax_periods) - 1, n_snapshots).round().astype(int)
        chosen_periods = [tax_periods[i] for i in idx]

    if figsize is None:
        figsize = (max(16, 2.4 * len(chosen_periods)), 11.0)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, len(chosen_periods), height_ratios=[1.65, 2.2, 2.2])
    ax_table = fig.add_subplot(gs[0, :])
    axes = np.asarray([
        [fig.add_subplot(gs[1, col]) for col in range(len(chosen_periods))],
        [fig.add_subplot(gs[2, col]) for col in range(len(chosen_periods))],
    ])

    ax_table.axis("off")
    metric_rows = [
        ("top wood price", "top", "avg_wood_price", "{:.2f}"),
        ("bottom wood price", "bottom", "avg_wood_price", "{:.2f}"),
        ("top stone price", "top", "avg_stone_price", "{:.2f}"),
        ("bottom stone price", "bottom", "avg_stone_price", "{:.2f}"),
        ("top avg skill", "top", "avg_skill_level", "{:.2f}"),
        ("bottom avg skill", "bottom", "avg_skill_level", "{:.2f}"),
        ("top builds", "top", "n_builds", "{:.0f}"),
        ("bottom builds", "bottom", "n_builds", "{:.0f}"),
        ("top avg labor", "top", "avg_labor_used", "{:.2f}"),
        ("bottom avg labor", "bottom", "avg_labor_used", "{:.2f}"),
    ]
    table_values = []
    row_labels = []
    for row_label, region, metric, fmt in metric_rows:
        row_vals = []
        for tax_period in chosen_periods:
            match = df_env[
                (df_env["tax_period"] == tax_period)
                & (df_env["region"] == region)
            ]
            value = match.iloc[0][metric] if len(match) else np.nan
            row_vals.append("" if pd.isna(value) else fmt.format(value))
        row_labels.append(row_label)
        table_values.append(row_vals)

    table = ax_table.table(
        cellText=table_values,
        rowLabels=row_labels,
        colLabels=[f"k={int(k)}" for k in chosen_periods],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.05)
    ax_table.text(
        0.5,
        1.03,
        "Environmental Metrics by Tax Period and Physical Region",
        transform=ax_table.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )

    bracket_x = np.arange(len(labels))
    max_count = max(1, int(counts["n_agents"].max())) if len(counts) else 1
    colors = {"top": "#1f77b4", "bottom": "#ff7f0e"}

    for row, region in enumerate(["top", "bottom"]):
        for col, tax_period in enumerate(chosen_periods):
            ax = axes[row, col]
            color = colors[region]

            day_counts = (
                counts[
                    (counts["region"] == region)
                    & (counts["tax_period"] == tax_period)
                ]
                .set_index("tax_bracket")["n_agents"]
                .reindex(labels)
                .fillna(0)
            )

            tax_row = df_tax[
                (df_tax["planner_region"] == region)
                & (df_tax["tax_period"] == tax_period)
            ]
            if len(tax_row):
                rates = np.asarray(tax_row.iloc[0]["tax_schedule"], dtype=float)
            else:
                rates = np.full(len(labels), np.nan)

            if len(rates) != len(labels):
                padded = np.full(len(labels), np.nan)
                padded[:min(len(labels), len(rates))] = rates[:len(labels)]
                rates = padded

            scaled_rates = rates * max_count
            ax.bar(
                bracket_x,
                day_counts.values,
                color=color,
                alpha=0.32,
                edgecolor=color,
                linewidth=1.0,
            )
            ax.plot(
                bracket_x,
                scaled_rates,
                color=color,
                marker="o",
                linewidth=2.0,
            )

            outcome = df_outcomes[
                (df_outcomes["region"] == region)
                & (df_outcomes["tax_period"] == tax_period)
            ]
            if len(outcome):
                production = outcome.iloc[0]["production"]
                equality = outcome.iloc[0]["equality"]
                n_agents = int(outcome.iloc[0]["n_agents"])
            else:
                production = np.nan
                equality = np.nan
                n_agents = 0

            ax.text(
                0.03,
                0.95,
                f"prod: {production:.1f}\neq: {equality:.3f}\nagents: {n_agents}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82),
            )

            ax.set_ylim(0, max_count + 0.6)
            ax.set_yticks(range(0, max_count + 1))
            if col == 0:
                ax.set_ylabel("agents")
                ax.text(
                    0.97,
                    0.95,
                    region,
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=10,
                    color=color,
                    fontweight="bold",
                )
            else:
                ax.tick_params(axis="y", labelleft=False)

            if row == 0:
                ax.set_title(f"k={tax_period}", fontsize=10)

            if row == 1:
                ax.set_xticks(bracket_x)
                ax.set_xticklabels([f"b{i}" for i in range(len(labels))], fontsize=8)
            else:
                ax.set_xticks(bracket_x)
                ax.set_xticklabels([])

            if col == len(chosen_periods) - 1:
                axr = ax.secondary_yaxis(
                    "right",
                    functions=(lambda y: y / max_count, lambda y: y * max_count),
                )
                axr.set_ylabel("tax rate")

            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Tax Bracket Snapshots with Regional Environment Metrics",
        fontsize=14,
    )

    return fig, df_income, counts, df_outcomes, df_env, df_tax


def plot_tax_period_rows_with_environment_metrics(
    log,
    brackets,
    period=100,
    n_snapshots=10,
    rate_disc=0.05,
    figsize=None,
):
    """
    Alternate layout for the same snapshot data.

    Rows are tax periods. Columns show environmental metrics and regional
    tax-bracket snapshots, which is less cramped when many periods are shown.
    """
    df_income, counts, labels = _period_region_income_table(log, brackets, period=period)
    df_outcomes = _regional_period_outcomes_from_income(df_income)
    df_env = regional_environment_metrics_by_tax_period(log, period=period)
    df_tax = _tax_policy_table(log, period=period, rate_disc=rate_disc)

    tax_periods = sorted(df_income["tax_period"].unique())
    if len(tax_periods) <= n_snapshots:
        chosen_periods = tax_periods
    else:
        idx = np.linspace(0, len(tax_periods) - 1, n_snapshots).round().astype(int)
        chosen_periods = [tax_periods[i] for i in idx]

    n_rows = len(chosen_periods)
    if figsize is None:
        figsize = (14, max(2.4 * n_rows, 10))

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 3, width_ratios=[1.45, 1.0, 1.0])

    colors = {"top": "#1f77b4", "bottom": "#ff7f0e"}
    bracket_x = np.arange(len(labels))
    max_count = max(1, int(counts["n_agents"].max())) if len(counts) else 1

    metric_labels = [
        ("wood", "avg_wood_price", "{:.2f}"),
        ("stone", "avg_stone_price", "{:.2f}"),
        ("skill", "avg_skill_level", "{:.2f}"),
        ("builds", "n_builds", "{:.0f}"),
        ("labor", "avg_labor_used", "{:.2f}"),
        ("agents", "n_agents", "{:.0f}"),
    ]

    for row, tax_period in enumerate(chosen_periods):
        ax_metrics = fig.add_subplot(gs[row, 0])
        ax_top = fig.add_subplot(gs[row, 1])
        ax_bottom = fig.add_subplot(gs[row, 2])

        ax_metrics.axis("off")
        metric_rows = []
        for label, metric, fmt in metric_labels:
            row_vals = [label]
            for region in ["top", "bottom"]:
                match = df_env[
                    (df_env["tax_period"] == tax_period)
                    & (df_env["region"] == region)
                ]
                value = match.iloc[0][metric] if len(match) else np.nan
                row_vals.append("" if pd.isna(value) else fmt.format(value))
            metric_rows.append(row_vals)

        table = ax_metrics.table(
            cellText=metric_rows,
            colLabels=["metric", "top", "bottom"],
            cellLoc="center",
            loc="center",
            colWidths=[0.42, 0.29, 0.29],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.12)
        ax_metrics.set_title(f"k={int(tax_period)}", loc="left", fontsize=11, fontweight="bold")

        for region, ax in [("top", ax_top), ("bottom", ax_bottom)]:
            color = colors[region]
            day_counts = (
                counts[
                    (counts["region"] == region)
                    & (counts["tax_period"] == tax_period)
                ]
                .set_index("tax_bracket")["n_agents"]
                .reindex(labels)
                .fillna(0)
            )

            tax_row = df_tax[
                (df_tax["planner_region"] == region)
                & (df_tax["tax_period"] == tax_period)
            ]
            if len(tax_row):
                rates = np.asarray(tax_row.iloc[0]["tax_schedule"], dtype=float)
            else:
                rates = np.full(len(labels), np.nan)

            if len(rates) != len(labels):
                padded = np.full(len(labels), np.nan)
                padded[:min(len(labels), len(rates))] = rates[:len(labels)]
                rates = padded

            ax.bar(
                bracket_x,
                day_counts.values,
                color=color,
                alpha=0.32,
                edgecolor=color,
                linewidth=1.0,
            )
            ax.plot(
                bracket_x,
                rates * max_count,
                color=color,
                marker="o",
                linewidth=1.8,
            )

            outcome = df_outcomes[
                (df_outcomes["region"] == region)
                & (df_outcomes["tax_period"] == tax_period)
            ]
            if len(outcome):
                production = outcome.iloc[0]["production"]
                equality = outcome.iloc[0]["equality"]
                n_agents = int(outcome.iloc[0]["n_agents"])
            else:
                production = np.nan
                equality = np.nan
                n_agents = 0

            ax.text(
                0.03,
                0.95,
                f"prod {production:.1f}\neq {equality:.3f}\nagents {n_agents}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82),
            )

            ax.set_ylim(0, max_count + 0.6)
            ax.set_yticks(range(0, max_count + 1))
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_xticks(bracket_x)
            ax.set_xticklabels([f"b{i}" for i in range(len(labels))], fontsize=8)

            if row == 0:
                ax.set_title(f"{region}: bars agents, line tax schedule", color=color, fontsize=10)
            if region == "top":
                ax.set_ylabel("agents")
            else:
                ax.tick_params(axis="y", labelleft=False)
                axr = ax.secondary_yaxis(
                    "right",
                    functions=(lambda y: y / max_count, lambda y: y * max_count),
                )
                axr.set_ylabel("tax rate")

    fig.suptitle(
        "Tax Period Rows: Environment Metrics, Tax Groups, Tax Schedule, Production, Equality",
        fontsize=14,
    )

    return fig, df_income, counts, df_outcomes, df_env, df_tax


def _tax_due_for_schedule(income, schedule, cutoffs):
    income = float(income)
    schedule = np.asarray(schedule, dtype=float)
    cutoffs = np.asarray(cutoffs, dtype=float)
    if len(schedule) != len(cutoffs):
        n = min(len(schedule), len(cutoffs))
        schedule = schedule[:n]
        cutoffs = cutoffs[:n]

    bracket_edges = np.concatenate([cutoffs, [np.inf]])
    bracket_sizes = bracket_edges[1:] - bracket_edges[:-1]
    past_cutoff = np.maximum(0.0, income - cutoffs)
    bin_income = np.minimum(bracket_sizes, past_cutoff)
    return float(np.sum(schedule * bin_income))


def _planner_schedules_from_actions_at_period(log, tax_period, period=100, rate_disc=0.05):
    actions = log.get("actions", [])
    t = min(max(0, (int(tax_period) - 1) * period), max(0, len(actions) - 1))
    action_t = actions[t] if t < len(actions) else {}
    schedules = {}

    for planner_id in ["p_top", "p_bottom"]:
        planner_action = action_t.get(planner_id, {}) if isinstance(action_t, dict) else {}
        pairs = []
        for key, value in planner_action.items():
            if "TaxIndexBracket" not in str(key):
                continue
            try:
                cutoff = float(str(key).split("_")[-1])
            except ValueError:
                cutoff = float(len(pairs))
            rate = np.clip((float(value) - 1.0) * float(rate_disc), 0.0, 1.0)
            pairs.append((cutoff, float(rate)))

        if pairs:
            pairs = sorted(pairs, key=lambda x: x[0])
            schedules[planner_id] = np.asarray([rate for _, rate in pairs], dtype=float)

    return schedules


def _all_current_planner_schedules_from_actions(log, period=100, rate_disc=0.05, cutoffs=None):
    if cutoffs is None:
        for tax_event in log.get("PeriodicTax", []):
            if isinstance(tax_event, dict) and tax_event and "cutoffs" in tax_event:
                cutoffs = np.asarray(tax_event["cutoffs"], dtype=float)
                break
    if cutoffs is None:
        cutoffs = np.asarray([0.0, 9.7, 39.475, 84.2, 160.725, 204.1, 510.3], dtype=float)

    cutoffs = np.asarray(cutoffs, dtype=float)
    current = {
        "p_top": np.zeros(len(cutoffs), dtype=float),
        "p_bottom": np.zeros(len(cutoffs), dtype=float),
    }
    schedules = {}
    n_periods = int(np.ceil((len(log.get("states", [])) - 1) / float(period)))

    for tax_period in range(1, n_periods + 1):
        t = min(max(0, (tax_period - 1) * period), max(0, len(log.get("actions", [])) - 1))
        action_t = log.get("actions", [])[t] if log.get("actions", []) else {}
        period_schedules = {}

        for planner_id in ["p_top", "p_bottom"]:
            planner_action = action_t.get(planner_id, {}) if isinstance(action_t, dict) else {}

            for key, value in planner_action.items():
                if "TaxIndexBracket" not in str(key):
                    continue
                try:
                    cutoff = float(str(key).split("_")[-1])
                except ValueError:
                    continue

                bracket_idx = int(np.argmin(np.abs(cutoffs - cutoff)))
                # Planner action 0 is no-op, so the previous current rate remains active.
                if float(value) > 0:
                    current[planner_id][bracket_idx] = np.clip(
                        (float(value) - 1.0) * float(rate_disc),
                        0.0,
                        1.0,
                    )

            period_schedules[planner_id] = current[planner_id].copy()

        schedules[tax_period] = period_schedules

    return schedules


def _closest_schedule_id(schedule, candidates):
    if not candidates:
        return None
    schedule = np.asarray(schedule, dtype=float)
    best_key = None
    best_dist = np.inf
    for key, candidate in candidates.items():
        candidate = np.asarray(candidate, dtype=float)
        n = min(len(schedule), len(candidate))
        if n == 0:
            continue
        dist = float(np.nanmean(np.abs(schedule[:n] - candidate[:n])))
        if dist < best_dist:
            best_dist = dist
            best_key = key
    return best_key


def agent_tax_mobility_counterfactual_table(log, period=100, rate_disc=0.05, travel_cost_coin=10.0):
    """
    Reconstruct full regional taxes/redistribution and counterfactual taxes.

    This is intentionally not limited to dense_log["PeriodicTax"], because both
    regional tax components share that dense-log key in the current simulator.
    Instead, per-period taxable income is reconstructed from dense event logs:
    build income + trade sales - trade purchases - travel fees. Then each agent's
    actual physical region at the tax boundary determines which current regional
    planner schedule is applied.

    Returns
    -------
    df_agent, df_period
        df_agent has one row per agent, aggregated over the full episode.
        df_period has one row per agent per tax event, useful for auditing.
    """
    states = log["states"]
    aids = _numeric_agent_ids(log)
    waterline = _infer_waterline(log)
    skill_by_agent = {
        aid: float(states[0][str(aid)].get("build_payment", np.nan))
        for aid in aids
    }
    all_current_schedules = _all_current_planner_schedules_from_actions(
        log,
        period=period,
        rate_disc=rate_disc,
    )

    cutoffs = None
    for tax_event in log.get("PeriodicTax", []):
        if isinstance(tax_event, dict) and tax_event and "cutoffs" in tax_event:
            cutoffs = np.asarray(tax_event["cutoffs"], dtype=float)
            break
    if cutoffs is None:
        cutoffs = np.asarray([0.0, 9.7, 39.475, 84.2, 160.725, 204.1, 510.3], dtype=float)

    tax_days = list(range(period - 1, len(states), period))
    period_rows = []

    for tax_period, tax_t in enumerate(tax_days, start=1):
        start_t = (tax_period - 1) * period
        end_t = min(tax_period * period, len(states) - 1)
        state_idx = min(tax_t, len(states) - 1)
        schedules = all_current_schedules.get(tax_period, {})

        travel_counts = {aid: 0 for aid in aids}
        travel_revenue = {"top": 0.0, "bottom": 0.0}
        for event in log.get("CrossWaterTravel", []):
            if not isinstance(event, dict) or "t" not in event:
                continue
            if start_t <= int(event["t"]) < end_t:
                aid = int(event.get("agent", -1))
                if aid in travel_counts:
                    travel_counts[aid] += 1
                if "from" in event:
                    origin_region = _region_from_loc(event["from"], waterline)
                    travel_revenue[origin_region] += float(travel_cost_coin)

        agent_period = []
        region_tax_total = {"top": 0.0, "bottom": 0.0}
        region_agents = {"top": [], "bottom": []}

        for aid in aids:
            builds = _events_in_period(log.get("Build", []), start_t, end_t, implicit_timeline=True)
            trades = _events_in_period(log.get("Trade", []), start_t, end_t, implicit_timeline=True)
            build_income = sum(float(b.get("income", 0.0)) for b in builds if int(b.get("builder", -1)) == aid)
            sell_income = sum(float(tr.get("income", 0.0)) for tr in trades if int(tr.get("seller", -1)) == aid)
            buy_cost = sum(float(tr.get("cost", 0.0)) for tr in trades if int(tr.get("buyer", -1)) == aid)
            travel_cost = travel_counts[aid] * float(travel_cost_coin)

            income = build_income + sell_income - buy_cost - travel_cost
            agent_region = _location_region_from_state(states[state_idx][str(aid)], waterline=waterline)
            actual_source = "p_top" if agent_region == "top" else "p_bottom"
            other_region = "bottom" if agent_region == "top" else "top"
            counterfactual_source = "p_bottom" if actual_source == "p_top" else "p_top"

            actual_schedule = schedules.get(actual_source, np.zeros(len(cutoffs)))
            counterfactual_schedule = schedules.get(counterfactual_source, np.zeros(len(cutoffs)))
            actual_tax_paid = _tax_due_for_schedule(income, actual_schedule, cutoffs)
            counterfactual_tax_due = _tax_due_for_schedule(income, counterfactual_schedule, cutoffs)

            region_tax_total[agent_region] += actual_tax_paid
            region_agents[agent_region].append(aid)

            agent_period.append({
                "tax_period": tax_period,
                "tax_timestep": tax_t,
                "agent": aid,
                "skill_level": skill_by_agent.get(aid, np.nan),
                "agent_physical_region": agent_region,
                "taxed_region": agent_region,
                "other_region": other_region,
                "income": income,
                "build_income": build_income,
                "sell_income": sell_income,
                "buy_cost": buy_cost,
                "travel_cost": travel_cost,
                "actual_tax_paid": actual_tax_paid,
                "counterfactual_tax_due_if_other_region": counterfactual_tax_due,
                "actual_minus_counterfactual_tax": actual_tax_paid - counterfactual_tax_due,
                "actual_schedule_source": actual_source,
                "counterfactual_schedule_source": counterfactual_source,
            })

        lump_sum = {
            region: (
                (region_tax_total[region] + travel_revenue[region])
                / max(1, len(region_agents[region]))
            )
            for region in ["top", "bottom"]
        }

        logged_event = log.get("PeriodicTax", [None] * (tax_t + 1))[tax_t]
        logged_schedule_source = None
        if isinstance(logged_event, dict) and logged_event and "schedule" in logged_event:
            logged_schedule_source = _closest_schedule_id(
                logged_event["schedule"],
                schedules,
            )

        for row in agent_period:
            reconstructed_redistribution = lump_sum[row["agent_physical_region"]]
            row["redistribution_received"] = reconstructed_redistribution
            row["region_travel_revenue"] = travel_revenue[row["agent_physical_region"]]
            row["logged_tax_paid"] = np.nan
            row["logged_redistribution_received"] = np.nan
            row["logged_schedule_source"] = logged_schedule_source

            if isinstance(logged_event, dict) and str(row["agent"]) in logged_event:
                entry = logged_event[str(row["agent"])]
                if isinstance(entry, dict):
                    row["logged_tax_paid"] = float(entry.get("tax_paid", np.nan))
                    row["logged_redistribution_received"] = float(entry.get("lump_sum", np.nan))
                    if row["actual_schedule_source"] == logged_schedule_source:
                        row["actual_tax_paid"] = row["logged_tax_paid"]
                        row["redistribution_received"] = row["logged_redistribution_received"]
                        row["actual_minus_counterfactual_tax"] = (
                            row["actual_tax_paid"]
                            - row["counterfactual_tax_due_if_other_region"]
                        )
            row["reconstructed_redistribution_received"] = reconstructed_redistribution

            period_rows.append(row)

    df_period = pd.DataFrame(period_rows)
    if df_period.empty:
        empty_agent = pd.DataFrame({"agent": aids})
        return empty_agent, df_period

    rows = []
    for aid, dfa in df_period.groupby("agent"):
        top_mask = dfa["taxed_region"] == "top"
        bottom_mask = dfa["taxed_region"] == "bottom"

        rows.append({
            "agent": int(aid),
            "skill_level": skill_by_agent.get(int(aid), np.nan),
            "total_tax_paid_top_region": float(dfa.loc[top_mask, "actual_tax_paid"].sum()),
            "total_tax_paid_bottom_region": float(dfa.loc[bottom_mask, "actual_tax_paid"].sum()),
            "total_redistribution_top_region": float(dfa.loc[top_mask, "redistribution_received"].sum()),
            "total_redistribution_bottom_region": float(dfa.loc[bottom_mask, "redistribution_received"].sum()),
            "total_actual_tax_paid": float(dfa["actual_tax_paid"].sum()),
            "total_counterfactual_tax_due_if_other_region": float(
                dfa["counterfactual_tax_due_if_other_region"].sum()
            ),
            "total_actual_minus_counterfactual_tax": float(
                dfa["actual_minus_counterfactual_tax"].sum()
            ),
            "net_tax_after_redistribution": float(
                dfa["actual_tax_paid"].sum() - dfa["redistribution_received"].sum()
            ),
            "n_tax_events_with_top_region": int(top_mask.sum()),
            "n_tax_events_with_bottom_region": int(bottom_mask.sum()),
        })

    return pd.DataFrame(rows).sort_values("agent").reset_index(drop=True), df_period


def _events_in_period(events, start_t, end_t, implicit_timeline=False):
    out = []
    for idx, item in enumerate(events):
        if not item:
            continue
        step_items = item if isinstance(item, list) else item.get("events", item.get("trades", item.get("builds", []))) if isinstance(item, dict) else []
        if isinstance(step_items, dict):
            step_items = [step_items]
        if not isinstance(step_items, list):
            continue
        for event in step_items:
            if not isinstance(event, dict):
                continue
            t = int(event.get("t", idx if implicit_timeline else -1))
            if start_t <= t < end_t:
                out.append(event)
    return out


def _regional_feedback_table(log, period=100, rate_disc=0.05, high_skill_quantile=0.5):
    tax_df = _extract_tax_policy_from_actions(log, period=period, rate_disc=rate_disc)
    if tax_df.empty:
        raise ValueError("No regional tax policy could be extracted from the dense log.")

    states = log["states"]
    aids = _numeric_agent_ids(log)
    waterline = _infer_waterline(log)
    skills = {
        aid: float(states[0][str(aid)].get("build_payment", np.nan))
        for aid in aids
    }
    finite_skills = np.asarray([v for v in skills.values() if np.isfinite(v)], dtype=float)
    high_skill_cutoff = (
        float(np.nanquantile(finite_skills, high_skill_quantile))
        if len(finite_skills)
        else np.nan
    )

    rewards = {
        "top": np.asarray(log.get("planner_rewards", {}).get("p_top", []), dtype=float),
        "bottom": np.asarray(log.get("planner_rewards", {}).get("p_bottom", []), dtype=float),
    }

    rows = []
    n_periods = int(tax_df["tax_period"].max())
    for tax_period in range(1, n_periods + 1):
        start_t = (tax_period - 1) * period
        end_t = min(tax_period * period, len(states) - 1)
        state_t = states[end_t]

        builds = _events_in_period(log.get("Build", []), start_t, end_t, implicit_timeline=True)
        trades = _events_in_period(log.get("Trade", []), start_t, end_t, implicit_timeline=True)
        travels = _events_in_period(log.get("CrossWaterTravel", []), start_t, end_t, implicit_timeline=False)

        for region, planner_id in [("top", "p_top"), ("bottom", "p_bottom")]:
            region_aids = [
                aid for aid in aids
                if _location_region_from_state(state_t[str(aid)], waterline=waterline) == region
            ]
            assigned_aids = [
                aid for aid in aids
                if _planner_region_from_initial_state(log, aid, waterline=waterline) == region
            ]

            region_builds = [
                b for b in builds
                if int(b.get("builder", -1)) in region_aids
            ]
            region_trades = [
                tr for tr in trades
                if tr.get("region") == region
                or int(tr.get("buyer", -1)) in region_aids
                or int(tr.get("seller", -1)) in region_aids
            ]
            region_travels = [
                ev for ev in travels
                if int(ev.get("agent", -1)) in region_aids
            ]

            coins = np.asarray([_coin(state_t[str(aid)]) for aid in assigned_aids], dtype=float)
            utilities = np.asarray(
                [float(state_t[str(aid)].get("utility", np.nan)) for aid in assigned_aids],
                dtype=float,
            )
            skill_vals = [skills[aid] for aid in region_aids]
            high_skill = [
                v for v in skill_vals
                if np.isfinite(v) and np.isfinite(high_skill_cutoff) and v >= high_skill_cutoff
            ]

            reward_arr = rewards[region]
            reward_slice = reward_arr[start_t:min(end_t, len(reward_arr))]
            reward_slice = reward_slice[np.isfinite(reward_slice)]

            rows.append({
                "tax_period": tax_period,
                "start_timestep": start_t,
                "end_timestep": end_t,
                "planner_region": region,
                "planner_id": planner_id,
                "population": len(region_aids),
                "population_share": len(region_aids) / max(1, len(aids)),
                "share_high_skill": len(high_skill) / max(1, len(region_aids)),
                "build_count": len(region_builds),
                "build_income": float(np.sum([b.get("income", 0.0) for b in region_builds])),
                "trade_count": len(region_trades),
                "trade_volume": float(np.sum([tr.get("price", tr.get("income", 0.0)) for tr in region_trades])),
                "travel_count": len(region_travels),
                "mean_labor": float(np.nanmean([
                    state_t[str(aid)].get("endogenous", {}).get("Labor", np.nan)
                    for aid in region_aids
                ])) if region_aids else np.nan,
                "production": float(np.nansum(coins)) if len(coins) else np.nan,
                "equality": _equality_from_values(coins) if len(coins) else np.nan,
                "swf_proxy": float(np.nansum(coins) * _equality_from_values(coins)) if len(coins) else np.nan,
                "mean_utility": float(np.nanmean(utilities)) if len(utilities) else np.nan,
                "planner_reward_sum": float(np.sum(reward_slice)) if len(reward_slice) else np.nan,
                "planner_reward_mean": float(np.mean(reward_slice)) if len(reward_slice) else np.nan,
            })

    behavior_df = pd.DataFrame(rows)
    df = behavior_df.merge(
        tax_df.drop(columns=["planner_id"]),
        on=["tax_period", "planner_region"],
        how="left",
    )

    next_tax = tax_df[["tax_period", "planner_region", "top_marginal_rate", "avg_marginal_rate", "progressivity"]].copy()
    next_tax["tax_period"] = next_tax["tax_period"] - 1
    next_tax = next_tax.rename(columns={
        "top_marginal_rate": "next_top_marginal_rate",
        "avg_marginal_rate": "next_avg_marginal_rate",
        "progressivity": "next_progressivity",
    })

    df = df.merge(next_tax, on=["tax_period", "planner_region"], how="left")
    return df


def plot_lagged_agent_planner_response_panel(
    log,
    period=100,
    rate_disc=0.05,
    policy_metric="top_marginal_rate",
    behavior_metric="build_count",
    composition_metric="share_high_skill",
    welfare_metric="planner_reward_sum",
    high_skill_quantile=0.5,
    figsize=(12, 9),
):
    """
    Plot tax-period timing: tax policy_k, behavior_k, composition_k, welfare_k.

    Returns
    -------
    fig, df
        df includes the lagged next-period tax columns used by the phase plot.
    """
    df = _regional_feedback_table(
        log,
        period=period,
        rate_disc=rate_disc,
        high_skill_quantile=high_skill_quantile,
    )

    labels = {
        "top_marginal_rate": "top marginal tax rate",
        "avg_marginal_rate": "average marginal tax rate",
        "progressivity": "tax progressivity",
        "build_count": "builds",
        "build_income": "build income",
        "trade_count": "trades",
        "trade_volume": "trade volume",
        "travel_count": "travel events",
        "mean_labor": "mean labor",
        "share_high_skill": "high-skill share",
        "population_share": "population share",
        "planner_reward_sum": "planner reward, period sum",
        "planner_reward_mean": "planner reward, period mean",
        "swf_proxy": "welfare proxy",
        "production": "production",
        "equality": "equality",
    }

    metrics = [
        (policy_metric, f"Planner policy at start of k: {labels.get(policy_metric, policy_metric)}"),
        (behavior_metric, f"Agent behavior during k: {labels.get(behavior_metric, behavior_metric)}"),
        (composition_metric, f"Regional composition at end of k: {labels.get(composition_metric, composition_metric)}"),
        (welfare_metric, f"Planner outcome over k: {labels.get(welfare_metric, welfare_metric)}"),
    ]

    colors = {"top": "#1f77b4", "bottom": "#ff7f0e"}
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)

    for ax, (metric, title) in zip(axes, metrics):
        if metric not in df.columns:
            raise ValueError(f"Unknown metric {metric!r}. Available columns: {sorted(df.columns)}")

        for region in ["top", "bottom"]:
            dfr = df[df["planner_region"] == region].sort_values("tax_period")
            ax.plot(
                dfr["tax_period"],
                dfr[metric],
                marker="o",
                linewidth=2,
                color=colors[region],
                label=region,
            )
        ax.set_title(title)
        ax.set_ylabel(labels.get(metric, metric))
        ax.grid(True, alpha=0.3)

    axes[0].legend(title="region", ncol=2, loc="best")
    axes[-1].set_xlabel("tax period k")
    fig.suptitle("Lagged Agent-Planner Response Panel", fontsize=14)
    return fig, df


def extract_travel_context_table(
    log,
    visible_radius=5,
    income_window=100,
    include_all_agent_steps=True,
):
    """
    Build agent-timestep rows describing the context around travel decisions.

    Travel events are evaluated at the state immediately before the logged travel
    timestep when possible. Non-travel rows are included so travel contexts can be
    compared against ordinary agent-timesteps from the same dense log.
    """
    import numpy as np
    import pandas as pd

    states = log.get("states", [])
    worlds = log.get("world", [])
    if not states:
        raise ValueError("Dense log has no states.")

    aids = _numeric_agent_ids(log)
    travel_events = list(_iter_travel_events(log))
    travel_lookup = {}
    for event in travel_events:
        if "agent" not in event:
            continue
        t = int(event.get("t", 0))
        aid = int(event["agent"])
        context_t = max(0, min(t - 1, len(states) - 1))
        travel_lookup.setdefault((aid, context_t), []).append(event)

    if include_all_agent_steps:
        pairs = [(aid, t) for t in range(len(states)) for aid in aids]
    else:
        pairs = sorted(travel_lookup.keys(), key=lambda x: (x[1], x[0]))

    def world_at(t):
        if not worlds:
            return {}
        t = min(max(0, int(t)), len(worlds) - 1)
        if worlds[t]:
            return worlds[t]
        for j in range(t, -1, -1):
            if worlds[j]:
                return worlds[j]
        for item in worlds:
            if item:
                return item
        return {}

    def map_array(world, key):
        value = world.get(key, None) if isinstance(world, dict) else None
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        return np.asarray(value)

    def visible_bounds(loc, shape):
        r, c = int(loc[0]), int(loc[1])
        r0 = max(0, r - visible_radius)
        r1 = min(shape[0], r + visible_radius + 1)
        c0 = max(0, c - visible_radius)
        c1 = min(shape[1], c + visible_radius + 1)
        return r0, r1, c0, c1

    def visible_sum(world, key, loc):
        arr = map_array(world, key)
        if arr is None or isinstance(arr, dict):
            return np.nan
        r0, r1, c0, c1 = visible_bounds(loc, arr.shape)
        return float(np.nansum(arr[r0:r1, c0:c1]))

    def visible_houses(world, loc, aid):
        house = map_array(world, "House")
        if not isinstance(house, dict):
            return np.nan, np.nan
        owner = np.asarray(house.get("owner", []))
        health = np.asarray(house.get("health", []), dtype=float)
        if owner.size == 0 or health.size == 0:
            return np.nan, np.nan
        r0, r1, c0, c1 = visible_bounds(loc, health.shape)
        owner_v = owner[r0:r1, c0:c1]
        health_v = health[r0:r1, c0:c1]
        active = health_v > 0
        own = float(np.sum(active & (owner_v == int(aid))))
        other = float(np.sum(active & (owner_v >= 0) & (owner_v != int(aid))))
        return own, other

    def total_own_houses(world, aid):
        house = map_array(world, "House")
        if not isinstance(house, dict):
            return np.nan
        owner = np.asarray(house.get("owner", []))
        health = np.asarray(house.get("health", []), dtype=float)
        if owner.size == 0 or health.size == 0:
            return np.nan
        return float(np.sum((health > 0) & (owner == int(aid))))

    waterline = _infer_waterline(log)
    rows = []
    for aid, t in pairs:
        if str(aid) not in states[t]:
            continue
        state = states[t][str(aid)]
        world = world_at(t)
        loc = state["loc"]
        event_list = travel_lookup.get((aid, t), [])
        did_travel = len(event_list) > 0
        event = event_list[0] if did_travel else {}

        prev_t = max(0, t - int(income_window))
        prev_state = states[prev_t][str(aid)]
        coin_now = _coin(state)
        recent_income = coin_now - _coin(prev_state)
        own_houses, other_houses = visible_houses(world, loc, aid)
        visible_wood = visible_sum(world, "Wood", loc)
        visible_stone = visible_sum(world, "Stone", loc)
        visible_wood_source = visible_sum(world, "WoodSourceBlock", loc)
        visible_stone_source = visible_sum(world, "StoneSourceBlock", loc)

        rows.append({
            "agent": int(aid),
            "timestep": int(t),
            "event_timestep": int(event.get("t", t)) if did_travel else np.nan,
            "did_travel": bool(did_travel),
            "location_region": _location_region_from_state(state, waterline=waterline),
            "planner_region": _planner_region_from_initial_state(log, aid, waterline=waterline),
            "row": int(loc[0]),
            "col": int(loc[1]),
            "to_row": event.get("to", [np.nan, np.nan])[0] if did_travel else np.nan,
            "to_col": event.get("to", [np.nan, np.nan])[1] if did_travel else np.nan,
            "current_coin": coin_now,
            "recent_income": float(recent_income),
            "skill_build_payment": float(states[0][str(aid)].get("build_payment", np.nan)),
            "final_travel_count": len(_travel_events_by_agent(log).get(aid, [])),
            "visible_wood": visible_wood,
            "visible_stone": visible_stone,
            "visible_resources": float(np.nansum([visible_wood, visible_stone])),
            "visible_wood_source": visible_wood_source,
            "visible_stone_source": visible_stone_source,
            "visible_resource_sources": float(np.nansum([visible_wood_source, visible_stone_source])),
            "visible_own_houses": own_houses,
            "visible_other_houses": other_houses,
            "own_houses_total": total_own_houses(world, aid),
            "labor": float(state.get("endogenous", {}).get("Labor", np.nan)),
            "utility": float(state.get("utility", np.nan)),
        })

    return pd.DataFrame(rows)


def plot_travel_context_dashboard(
    log,
    comparison_log=None,
    visible_radius=5,
    income_window=100,
    nontravel_sample=1200,
    random_state=0,
    figsize=(18, 14),
):
    """
    Large diagnostic dashboard for who travels and what their local situation
    looked like before travel.

    Usage:
        log = plotting._extract_logs_from_run(runs[2])[0]
        baseline_log = plotting._extract_logs_from_run(runs[0])[0]
        fig, context_df, agent_df = plotting.plot_travel_context_dashboard(
            log, comparison_log=baseline_log
        )
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    focal_context_df = extract_travel_context_table(
        log,
        visible_radius=visible_radius,
        income_window=income_window,
        include_all_agent_steps=True,
    )
    if focal_context_df.empty:
        raise ValueError("No agent context rows could be constructed.")

    if comparison_log is None:
        baseline_context_df = focal_context_df.copy()
        baseline_label = "same log non-travel"
    else:
        baseline_context_df = extract_travel_context_table(
            comparison_log,
            visible_radius=visible_radius,
            income_window=income_window,
            include_all_agent_steps=True,
        )
        baseline_label = "comparison log non-travel"

    agent_df = _agent_behavior_table(log)
    travel_df = focal_context_df[focal_context_df["did_travel"]].copy()
    nontravel_df = baseline_context_df[~baseline_context_df["did_travel"]].copy()
    if len(nontravel_df) > nontravel_sample:
        nontravel_plot = nontravel_df.sample(nontravel_sample, random_state=random_state)
    else:
        nontravel_plot = nontravel_df

    focal_context_df = focal_context_df.copy()
    baseline_context_df = baseline_context_df.copy()
    focal_context_df["context_source"] = "travel_log"
    baseline_context_df["context_source"] = "comparison_log"
    context_df = pd.concat([focal_context_df, baseline_context_df], ignore_index=True)

    features = [
        "current_coin",
        "recent_income",
        "skill_build_payment",
        "visible_resources",
        "visible_resource_sources",
        "visible_other_houses",
        "visible_own_houses",
        "labor",
    ]
    feature_labels = {
        "current_coin": "coin",
        "recent_income": f"income, last {income_window}",
        "skill_build_payment": "skill",
        "visible_resources": "visible wood+stone",
        "visible_resource_sources": "visible source tiles",
        "visible_other_houses": "visible other houses",
        "visible_own_houses": "visible own houses",
        "labor": "labor",
    }

    def standardized_difference(feature):
        a = travel_df[feature].dropna().to_numpy(dtype=float)
        b = nontravel_df[feature].dropna().to_numpy(dtype=float)
        if len(a) == 0 or len(b) == 0:
            return np.nan
        pooled = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2.0)
        if pooled <= 1e-12:
            return 0.0
        return float((np.nanmean(a) - np.nanmean(b)) / pooled)

    smd = pd.DataFrame({
        "feature": features,
        "label": [feature_labels[f] for f in features],
        "standardized_difference": [standardized_difference(f) for f in features],
    }).sort_values("standardized_difference")

    agent_colors = _make_agent_colors(_numeric_agent_ids(log))
    region_colors = {"top": "#1f77b4", "bottom": "#ff7f0e"}
    travel_color = "#c92a2a"
    nontravel_color = "0.75"

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[1.0, 1.15, 1.15, 1.2],
        width_ratios=[1.1, 1.1, 1.0],
    )

    ax_timeline = fig.add_subplot(gs[0, :2])
    ax_counts = fig.add_subplot(gs[0, 2])
    ax_smd = fig.add_subplot(gs[1, 0])
    ax_coin = fig.add_subplot(gs[1, 1])
    ax_resources = fig.add_subplot(gs[1, 2])
    ax_scatter = fig.add_subplot(gs[2, 0])
    ax_houses = fig.add_subplot(gs[2, 1])
    ax_other_houses = fig.add_subplot(gs[2, 2])
    ax_heat = fig.add_subplot(gs[3, :2])
    ax_table = fig.add_subplot(gs[3, 2])

    # 1. Timeline heatmap: where travel happens in time by agent.
    aids = sorted(context_df["agent"].unique())
    aid_to_y = {aid: i for i, aid in enumerate(aids)}
    for aid in aids:
        dfa = focal_context_df[focal_context_df["agent"] == aid]
        color = agent_colors.get(int(aid), "0.5")
        ax_timeline.hlines(
            aid_to_y[aid],
            dfa["timestep"].min(),
            dfa["timestep"].max(),
            color=color,
            alpha=0.35,
            linewidth=5,
        )
    if not travel_df.empty:
        ax_timeline.scatter(
            travel_df["event_timestep"],
            [aid_to_y[a] for a in travel_df["agent"]],
            marker="|",
            s=220,
            linewidths=2.2,
            color=travel_color,
            label="travel",
            zorder=3,
        )
    ax_timeline.set_yticks(range(len(aids)))
    ax_timeline.set_yticklabels([str(a) for a in aids])
    ax_timeline.set_title("Who Travels, and When")
    ax_timeline.set_xlabel("timestep")
    ax_timeline.set_ylabel("agent")
    ax_timeline.grid(True, axis="x", alpha=0.25)

    # 2. Counts by agent.
    counts = agent_df.sort_values(["travel_events", "agent"], ascending=[False, True]).copy()
    bar_colors = [agent_colors.get(int(a), "0.5") for a in counts["agent"]]
    ax_counts.bar(counts["agent"].astype(str), counts["travel_events"], color=bar_colors, edgecolor="white")
    ax_counts.set_title("Travel Events by Agent")
    ax_counts.set_xlabel("agent")
    ax_counts.set_ylabel("events")
    ax_counts.grid(True, axis="y", alpha=0.25)

    # 3. Common-denominator effect sizes.
    ax_smd.barh(
        smd["label"],
        smd["standardized_difference"],
        color=[travel_color if v > 0 else "#2b8a3e" for v in smd["standardized_difference"]],
        alpha=0.85,
    )
    ax_smd.axvline(0, color="0.2", linewidth=1)
    ax_smd.set_title(f"Travel Context vs {baseline_label}\nstandardized mean difference")
    ax_smd.set_xlabel("higher at travel  ->")
    ax_smd.grid(True, axis="x", alpha=0.25)

    def grouped_box(ax, feature, title, ylabel):
        data = [
            nontravel_df[feature].dropna().to_numpy(dtype=float),
            travel_df[feature].dropna().to_numpy(dtype=float),
        ]
        bp = ax.boxplot(data, labels=["non-travel", "travel"], patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], [nontravel_color, travel_color]):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

    grouped_box(ax_coin, "current_coin", "Coin Holdings Before Decision", "coin")
    grouped_box(ax_resources, "visible_resources", "Visible Loose Resources", "wood + stone")
    grouped_box(ax_houses, "visible_own_houses", "Own Houses in Visible Area", "count")
    grouped_box(
        ax_other_houses,
        "visible_other_houses",
        "Other Agents' Houses Visible Before Decision",
        "count",
    )

    # 4. Scatter: travel points against background of ordinary contexts.
    ax_scatter.scatter(
        nontravel_plot["current_coin"],
        nontravel_plot["visible_resources"],
        s=18,
        color=nontravel_color,
        alpha=0.22,
        label="non-travel sampled",
    )
    if not travel_df.empty:
        sizes = 45 + 20 * travel_df["skill_build_payment"].rank(pct=True).fillna(0.5)
        ax_scatter.scatter(
            travel_df["current_coin"],
            travel_df["visible_resources"],
            s=sizes,
            color=[agent_colors.get(int(a), travel_color) for a in travel_df["agent"]],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            label="travel",
        )
        for _, row in travel_df.iterrows():
            ax_scatter.annotate(
                str(int(row["agent"])),
                (row["current_coin"], row["visible_resources"]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax_scatter.set_title("Coin vs Visible Resources")
    ax_scatter.set_xlabel("current coin")
    ax_scatter.set_ylabel("visible wood + stone")
    ax_scatter.grid(True, alpha=0.25)

    # 5. Aggregated traveler heatmap: one row per traveler, readable even with many events.
    heat_features = [
        "current_coin",
        "recent_income",
        "visible_resources",
        "visible_other_houses",
        "visible_own_houses",
        "skill_build_payment",
    ]
    if travel_df.empty:
        ax_heat.text(0.5, 0.5, "No travel events in this log", ha="center", va="center")
        ax_heat.set_axis_off()
    else:
        traveler_profiles = (
            travel_df
            .groupby("agent", as_index=False)[heat_features]
            .mean()
            .sort_values("agent")
        )
        mat = traveler_profiles[heat_features].to_numpy(dtype=float)
        col_mean = np.nanmean(nontravel_df[heat_features].to_numpy(dtype=float), axis=0)
        col_std = np.nanstd(nontravel_df[heat_features].to_numpy(dtype=float), axis=0)
        col_std[col_std <= 1e-12] = 1.0
        zmat = (mat - col_mean) / col_std
        im = ax_heat.imshow(zmat, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
        ax_heat.set_yticks(np.arange(len(traveler_profiles)))
        ax_heat.set_yticklabels([f"agent {int(a)}" for a in traveler_profiles["agent"]], fontsize=9)
        ax_heat.set_xticks(np.arange(len(heat_features)))
        ax_heat.set_xticklabels([feature_labels[f] for f in heat_features], rotation=15, ha="right")
        ax_heat.set_title(f"Traveler Profiles at Travel Moments\nz-score relative to {baseline_label}")
        fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.01, label="z-score")

    # 6. Compact readout.
    ax_table.axis("off")
    n_events = int(len(travel_df))
    travelers = sorted(travel_df["agent"].unique()) if n_events else []
    top_agents = agent_df.sort_values("travel_events", ascending=False).head(4)
    if n_events:
        summary_text = (
            f"Travel events: {n_events}\n"
            f"Travelers: {', '.join(str(int(a)) for a in travelers)}\n"
            f"Baseline: {baseline_label}\n\n"
            f"Travel mean coin: {travel_df['current_coin'].mean():.1f}\n"
            f"Travel mean visible resources: {travel_df['visible_resources'].mean():.1f}\n"
            f"Travel mean own houses visible: {travel_df['visible_own_houses'].mean():.1f}\n\n"
            "Top travelers\n"
            + "\n".join(
                f"a{int(r.agent)}: {int(r.travel_events)} events, skill {r.skill_build_payment:.0f}"
                for _, r in top_agents.iterrows()
            )
        )
    else:
        summary_text = (
            "No travel events in focal log.\n"
            f"Baseline: {baseline_label}\n"
            f"visible radius: {visible_radius}\n"
            f"income window: {income_window}"
        )
    ax_table.set_title("Readout")
    ax_table.text(
        0.02,
        0.95,
        summary_text,
        transform=ax_table.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.35,
    )

    legend_handles = [
        Line2D([0], [0], marker="|", color=travel_color, linestyle="None", markersize=13, markeredgewidth=2, label="travel event"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markeredgecolor="white", label="travel context, agent-colored"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=nontravel_color, label="non-travel sample"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=True)
    fig.suptitle(
        "Travel Decision Context Dashboard",
        fontsize=16,
        fontweight="bold",
    )

    return fig, context_df, agent_df


def extract_tax_period_travel_context_table(
    log,
    period=100,
    visible_radius=5,
    income_window=100,
):
    """Aggregate travel context to one row per agent per tax period."""
    import numpy as np
    import pandas as pd

    step_df = extract_travel_context_table(
        log,
        visible_radius=visible_radius,
        income_window=income_window,
        include_all_agent_steps=True,
    )
    if step_df.empty:
        return step_df

    step_df = step_df.copy()
    step_df["tax_period"] = (step_df["timestep"].astype(int) // int(period)) + 1
    step_df["travel_count"] = step_df["did_travel"].astype(int)

    agg = (
        step_df
        .sort_values(["agent", "tax_period", "timestep"])
        .groupby(["agent", "tax_period"], as_index=False)
        .agg(
            did_travel=("did_travel", "max"),
            travel_count=("travel_count", "sum"),
            period_start_timestep=("timestep", "min"),
            period_end_timestep=("timestep", "max"),
            planner_region=("planner_region", "first"),
            location_region_start=("location_region", "first"),
            location_region_end=("location_region", "last"),
            skill_build_payment=("skill_build_payment", "first"),
            period_start_coin=("current_coin", "first"),
            period_end_coin=("current_coin", "last"),
            mean_coin=("current_coin", "mean"),
            mean_recent_income=("recent_income", "mean"),
            mean_visible_resources=("visible_resources", "mean"),
            mean_visible_resource_sources=("visible_resource_sources", "mean"),
            mean_visible_own_houses=("visible_own_houses", "mean"),
            mean_visible_other_houses=("visible_other_houses", "mean"),
            own_houses_total=("own_houses_total", "mean"),
            mean_labor=("labor", "mean"),
            final_labor=("labor", "last"),
            mean_utility=("utility", "mean"),
        )
    )
    agg["period_coin_change"] = agg["period_end_coin"] - agg["period_start_coin"]
    agg["did_travel"] = agg["did_travel"].astype(bool)
    return agg


def _dense_log_items(log_or_run):
    """Return ``(rollout_id, dense_log)`` pairs for a single log or many logs."""
    if isinstance(log_or_run, dict) and isinstance(log_or_run.get("states"), list):
        return [(0, log_or_run)]

    if isinstance(log_or_run, dict):
        logs = _extract_logs_from_run(log_or_run)
        if logs:
            return [
                (k, v)
                for k, v in logs.items()
                if isinstance(v, dict) and isinstance(v.get("states"), list)
            ]
        nested = []
        for key in ["final", "episodes", "dense_logs", "logs", "data"]:
            if key in log_or_run:
                nested.extend(_dense_log_items(log_or_run[key]))
        if nested:
            return nested
        return [
            (k, v)
            for k, v in log_or_run.items()
            if isinstance(v, dict) and isinstance(v.get("states"), list)
        ]

    if isinstance(log_or_run, (list, tuple)):
        return [
            (i, v)
            for i, v in enumerate(log_or_run)
            if isinstance(v, dict) and isinstance(v.get("states"), list)
        ]

    return []


def tax_period_travel_context_table_from_dense_logs(
    log_or_run,
    period=100,
    visible_radius=5,
    income_window=100,
):
    """Pool the tax-period travel context table across one or more dense logs."""
    import pandas as pd

    frames = []
    for rollout_id, dense_log in _dense_log_items(log_or_run):
        df = extract_tax_period_travel_context_table(
            dense_log,
            period=period,
            visible_radius=visible_radius,
            income_window=income_window,
        ).copy()
        if df.empty:
            continue
        df["rollout_id"] = rollout_id
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_tax_period_travel_context_dashboard(
    log,
    period=100,
    visible_radius=5,
    income_window=100,
    figsize=(18, 13),
):
    """
    Compare agents who travel vs agents who do not within each tax period.

    Usage:
        log = plotting._extract_logs_from_run(runs[2])[0]
        fig, period_df = plotting.plot_tax_period_travel_context_dashboard(log)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    period_df = tax_period_travel_context_table_from_dense_logs(
        log,
        period=period,
        visible_radius=visible_radius,
        income_window=income_window,
    )
    if period_df.empty:
        raise ValueError("No agent-period context rows could be constructed.")

    features = [
        "mean_coin",
        "period_coin_change",
        "mean_visible_resources",
        "mean_visible_other_houses",
        "mean_visible_own_houses",
        "skill_build_payment",
    ]
    feature_labels = {
        "mean_coin": "mean coin",
        "period_coin_change": "coin change in period",
        "mean_visible_resources": "visible wood+stone",
        "mean_visible_other_houses": "visible other houses",
        "mean_visible_own_houses": "visible own houses",
        "skill_build_payment": "skill",
    }

    def smd_for_period(dfp, feature):
        travelers = dfp[dfp["did_travel"]][feature].dropna().to_numpy(dtype=float)
        stayers = dfp[~dfp["did_travel"]][feature].dropna().to_numpy(dtype=float)
        if len(travelers) == 0 or len(stayers) == 0:
            return np.nan
        pooled = np.sqrt((np.nanvar(travelers) + np.nanvar(stayers)) / 2.0)
        if pooled <= 1e-12:
            return 0.0
        return float((np.nanmean(travelers) - np.nanmean(stayers)) / pooled)

    smd_rows = []
    for group_key, dfp in period_df.groupby(["rollout_id", "tax_period"]):
        rollout_id, tax_period = group_key
        for feature in features:
            smd_rows.append({
                "rollout_id": rollout_id,
                "tax_period": tax_period,
                "feature": feature,
                "label": feature_labels[feature],
                "standardized_difference": smd_for_period(dfp, feature),
            })
    smd_df = pd.DataFrame(smd_rows)

    agent_colors = _make_agent_colors(period_df["agent"].unique())
    travel_color = "#c92a2a"
    nontravel_color = "0.72"

    if figsize == (18, 13):
        figsize = (18, 8)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.1, 1.1],
        width_ratios=[1.0, 1.0, 1.0, 1.0],
    )

    ax_coin = fig.add_subplot(gs[0, 0])
    ax_resources = fig.add_subplot(gs[0, 1])
    ax_houses = fig.add_subplot(gs[0, 2])
    ax_other_houses = fig.add_subplot(gs[0, 3])
    ax_smd = fig.add_subplot(gs[1, :2])
    ax_scatter = fig.add_subplot(gs[1, 2:])

    agents = sorted(period_df["agent"].unique())
    tax_periods = sorted(period_df["tax_period"].unique())
    agent_to_y = {aid: i for i, aid in enumerate(agents)}
    period_to_x = {tp: i for i, tp in enumerate(tax_periods)}

    counts = (
        period_df.groupby("tax_period", as_index=False)
        .agg(
            traveling_agents=("did_travel", "sum"),
            travel_events=("travel_count", "sum"),
        )
    )

    avg_smd = (
        smd_df.groupby(["feature", "label"], as_index=False)["standardized_difference"]
        .mean()
        .sort_values("standardized_difference")
    )
    ax_smd.barh(
        avg_smd["label"],
        avg_smd["standardized_difference"],
        color=[travel_color if v > 0 else "#2b8a3e" for v in avg_smd["standardized_difference"]],
        alpha=0.85,
    )
    ax_smd.axvline(0, color="0.2", linewidth=1)
    ax_smd.set_title("Average Difference Across Tax Periods\ntravelers vs non-travelers")
    ax_smd.set_xlabel("Higher among travelers (measured in std from mean)")
    ax_smd.grid(True, axis="x", alpha=0.25)

    n_rollouts = period_df["rollout_id"].nunique()
    box_df = period_df
    if n_rollouts > 1:
        box_df = (
            period_df.groupby(["rollout_id", "did_travel"], as_index=False)[features]
            .mean()
        )

    def grouped_box(ax, feature, title, ylabel):
        data = [
            box_df.loc[~box_df["did_travel"], feature].dropna().to_numpy(dtype=float),
            box_df.loc[box_df["did_travel"], feature].dropna().to_numpy(dtype=float),
        ]
        bp = ax.boxplot(
            data,
            labels=["no travel\nin period", "travel\nin period"],
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            meanline=True,
            meanprops=dict(color="black", linewidth=2.4, linestyle="-"),
            medianprops=dict(linewidth=0),
        )
        for patch, color in zip(bp["boxes"], [nontravel_color, travel_color]):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

    grouped_box(ax_coin, "mean_coin", "Coin During Period", "coin")
    grouped_box(ax_resources, "mean_visible_resources", "Visible Resources During Period", "wood + stone")
    grouped_box(ax_houses, "mean_visible_own_houses", "Own Houses Visible During Period", "count")
    grouped_box(
        ax_other_houses,
        "mean_visible_other_houses",
        "Other Agents' Houses Visible During Period",
        "count",
    )

    ax_scatter.scatter(
        period_df.loc[~period_df["did_travel"], "period_coin_change"],
        period_df.loc[~period_df["did_travel"], "mean_visible_other_houses"],
        s=28,
        color=nontravel_color,
        alpha=0.35,
        label="no travel in period",
    )
    travel_periods = period_df[period_df["did_travel"]]
    ax_scatter.scatter(
        travel_periods["period_coin_change"],
        travel_periods["mean_visible_other_houses"],
        s=55,
        color=[agent_colors.get(int(a), travel_color) for a in travel_periods["agent"]],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
        label="travel in period",
    )
    trend = travel_periods[["period_coin_change", "mean_visible_other_houses"]].dropna()
    if len(trend) >= 2 and trend["period_coin_change"].nunique() >= 2:
        xfit = trend["period_coin_change"].to_numpy(dtype=float)
        yfit = trend["mean_visible_other_houses"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(xfit, yfit, 1)
        xs = np.linspace(float(np.min(xfit)), float(np.max(xfit)), 100)
        ax_scatter.plot(
            xs,
            slope * xs + intercept,
            color="0.15",
            linewidth=2.2,
            alpha=0.9,
            label="travel-point least-squares fit",
        )
    ax_scatter.set_title("Agent-Period Contexts")
    ax_scatter.set_xlabel("period income / coin change")
    ax_scatter.set_ylabel("mean visible other agents' houses")
    ax_scatter.grid(True, alpha=0.25)
    fig.suptitle("Tax-Period Travel Context: Travelers vs Non-Travelers", fontsize=16, fontweight="bold")

    return fig, period_df


def plot_travel_timeline_by_agent(log, figsize=(12, 3.5)):
    """
    Plot travel events by agent over time using the same stable agent colors as
    breakdown_all_agents.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    aids = _numeric_agent_ids(log)
    agent_colors = _make_agent_colors(aids)
    travel_by_agent = _travel_events_by_agent(log)
    n_steps = max(0, len(log.get("states", [])) - 1)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    for y, aid in enumerate(aids):
        color = agent_colors.get(int(aid), "0.5")
        ax.hlines(y, 0, n_steps, color=color, alpha=0.28, linewidth=5)
        ts = [
            int(event["t"])
            for event in travel_by_agent.get(int(aid), [])
            if "t" in event
        ]
        if ts:
            ax.scatter(
                ts,
                np.full(len(ts), y),
                marker="|",
                s=220,
                linewidths=2.4,
                color=color,
                alpha=0.98,
                zorder=3,
            )

    ax.set_title("Who Travels, and When")
    ax.set_xlabel("timestep")
    ax.set_ylabel("agent")
    ax.set_yticks(np.arange(len(aids)))
    ax.set_yticklabels([str(int(aid)) for aid in aids])
    ax.set_xlim(0, n_steps)
    ax.grid(True, axis="x", alpha=0.25)

    return fig


def _tax_schedules_by_region_period(log, period=100, rate_disc=0.05):
    """Return regional schedules keyed by tax period using tax logs when present."""
    import numpy as np

    schedules = {}
    cutoffs = None
    period_counts = {"top": 0, "bottom": 0}
    found_periodic = False

    for region, key in [("top", "PeriodicTax-p_top"), ("bottom", "PeriodicTax-p_bottom")]:
        for event in log.get(key, []) or []:
            if not isinstance(event, dict) or not event:
                continue
            schedule = np.asarray(event.get("schedule", []), dtype=float)
            if len(schedule) == 0:
                continue
            if cutoffs is None and "cutoffs" in event:
                cutoffs = np.asarray(event["cutoffs"], dtype=float)
            period_counts[region] += 1
            schedules.setdefault(period_counts[region], {})[region] = schedule
            found_periodic = True

    if cutoffs is None:
        cutoffs = np.asarray([0.0, 9.7, 39.475, 84.2, 160.725, 204.1, 510.3], dtype=float)

    if found_periodic:
        return schedules, cutoffs

    action_schedules = _all_current_planner_schedules_from_actions(
        log,
        period=period,
        rate_disc=rate_disc,
        cutoffs=cutoffs,
    )
    for tax_period, planner_schedules in action_schedules.items():
        schedules[tax_period] = {
            "top": planner_schedules.get("p_top", np.zeros(len(cutoffs), dtype=float)),
            "bottom": planner_schedules.get("p_bottom", np.zeros(len(cutoffs), dtype=float)),
        }
    return schedules, cutoffs


def tax_period_travel_counterfactual_table(
    log,
    period=100,
    visible_radius=5,
    income_window=100,
    rate_disc=0.05,
):
    """
    Agent-period table with current-region and other-region tax comparisons.
    """
    import numpy as np

    df = extract_tax_period_travel_context_table(
        log,
        period=period,
        visible_radius=visible_radius,
        income_window=income_window,
    ).copy()
    schedules, cutoffs = _tax_schedules_by_region_period(
        log,
        period=period,
        rate_disc=rate_disc,
    )

    def schedule_for(tax_period, region):
        return np.asarray(
            schedules.get(int(tax_period), {}).get(region, np.zeros(len(cutoffs), dtype=float)),
            dtype=float,
        )

    current_avg = []
    other_avg = []
    current_top = []
    other_top = []
    current_due = []
    other_due = []
    taxable_income = []

    for _, row in df.iterrows():
        current_region = row["location_region_start"]
        other_region = "bottom" if current_region == "top" else "top"
        current_schedule = schedule_for(row["tax_period"], current_region)
        other_schedule = schedule_for(row["tax_period"], other_region)
        income = max(0.0, float(row["period_coin_change"]))

        taxable_income.append(income)
        current_avg.append(float(np.nanmean(current_schedule)) if len(current_schedule) else np.nan)
        other_avg.append(float(np.nanmean(other_schedule)) if len(other_schedule) else np.nan)
        current_top.append(float(current_schedule[-1]) if len(current_schedule) else np.nan)
        other_top.append(float(other_schedule[-1]) if len(other_schedule) else np.nan)
        current_due.append(_tax_due_for_schedule(income, current_schedule, cutoffs))
        other_due.append(_tax_due_for_schedule(income, other_schedule, cutoffs))

    df["taxable_period_income"] = taxable_income
    df["current_region_avg_tax"] = current_avg
    df["other_region_avg_tax"] = other_avg
    df["current_region_top_tax"] = current_top
    df["other_region_top_tax"] = other_top
    df["current_region_tax_due"] = current_due
    df["other_region_tax_due"] = other_due
    df["other_minus_current_avg_tax"] = df["other_region_avg_tax"] - df["current_region_avg_tax"]
    df["other_minus_current_tax_due"] = df["other_region_tax_due"] - df["current_region_tax_due"]
    return df


def tax_period_travel_counterfactual_table_from_dense_logs(
    log_or_run,
    period=100,
    visible_radius=5,
    income_window=100,
    rate_disc=0.05,
):
    """Pool the travel counterfactual tax table across one or more dense logs."""
    import pandas as pd

    frames = []
    for rollout_id, dense_log in _dense_log_items(log_or_run):
        df = tax_period_travel_counterfactual_table(
            dense_log,
            period=period,
            visible_radius=visible_radius,
            income_window=income_window,
            rate_disc=rate_disc,
        ).copy()
        if df.empty:
            continue
        df["rollout_id"] = rollout_id
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_tax_travel_counterfactuals(
    log,
    period=100,
    visible_radius=5,
    income_window=100,
    rate_disc=0.05,
    figsize=(16, 11),
):
    """
    Separate tax-focused figure comparing tax faced by travel vs non-travel
    agent-periods and current-region versus other-region counterfactual taxes.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    df = tax_period_travel_counterfactual_table_from_dense_logs(
        log,
        period=period,
        visible_radius=visible_radius,
        income_window=income_window,
        rate_disc=rate_disc,
    )
    if df.empty:
        raise ValueError("No agent-period counterfactual tax rows could be constructed.")

    travel_color = "#c92a2a"
    nontravel_color = "0.72"
    if figsize == (16, 11):
        figsize = (13, 5.5)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ax_box, ax_scatter = axes

    # 1. Boxplot: average regional tax faced.
    n_rollouts = df["rollout_id"].nunique()
    box_df = df
    if n_rollouts > 1:
        box_df = (
            df.groupby(["rollout_id", "did_travel"], as_index=False)
            .agg(current_region_avg_tax=("current_region_avg_tax", "mean"))
        )
    data = [
        box_df.loc[~box_df["did_travel"], "current_region_avg_tax"].dropna().to_numpy(dtype=float),
        box_df.loc[box_df["did_travel"], "current_region_avg_tax"].dropna().to_numpy(dtype=float),
    ]
    bp = ax_box.boxplot(
        data,
        labels=["no travel\nin period", "travel\nin period"],
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanline=True,
        meanprops=dict(color="black", linewidth=2.4, linestyle="-"),
        medianprops=dict(linewidth=0),
    )
    for patch, color in zip(bp["boxes"], [nontravel_color, travel_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    title_suffix = " (rollout averages)" if n_rollouts > 1 else ""
    ax_box.set_title(f"Average Tax Faced{title_suffix}")
    ax_box.set_ylabel("average marginal tax rate")
    ax_box.grid(True, axis="y", alpha=0.25)

    # 2. Scatter: current vs other tax due, colored by income.
    plot_df = df.dropna(subset=["current_region_tax_due", "other_region_tax_due", "taxable_period_income"])
    if not plot_df.empty:
        sc = None
        for did_travel, size, edgecolor, linewidth, alpha in [
            (False, 32, "white", 0.5, 0.35),
            (True, 70, travel_color, 1.5, 0.9),
        ]:
            dfr = plot_df[plot_df["did_travel"] == did_travel]
            if dfr.empty:
                continue
            sc = ax_scatter.scatter(
                dfr["current_region_tax_due"],
                dfr["other_region_tax_due"],
                c=dfr["taxable_period_income"],
                cmap="viridis",
                s=size,
                marker="o",
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
            )
        lo = float(np.nanmin([plot_df["current_region_tax_due"].min(), plot_df["other_region_tax_due"].min()]))
        hi = float(np.nanmax([plot_df["current_region_tax_due"].max(), plot_df["other_region_tax_due"].max()]))
        pad = max(1.0, 0.05 * (hi - lo))
        ax_scatter.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.35", linestyle="--", linewidth=1)
        ax_scatter.set_xlim(lo - pad, hi + pad)
        ax_scatter.set_ylim(lo - pad, hi + pad)
        trend = plot_df.loc[
            plot_df["did_travel"],
            ["current_region_tax_due", "other_region_tax_due"],
        ].dropna()
        if len(trend) >= 2 and trend["current_region_tax_due"].nunique() >= 2:
            xfit = trend["current_region_tax_due"].to_numpy(dtype=float)
            yfit = trend["other_region_tax_due"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(xfit, yfit, 1)
            xs = np.linspace(float(np.min(xfit)), float(np.max(xfit)), 100)
            ax_scatter.plot(
                xs,
                slope * xs + intercept,
                color="0.15",
                linewidth=2.2,
                alpha=0.9,
                label="travel-point least-squares fit",
            )
            ax_scatter.legend(loc="best", frameon=True)
        if sc is not None:
            fig.colorbar(sc, ax=ax_scatter, label="money made in period")
    ax_scatter.set_title("Current-Region Tax Due vs Other-Region Tax Due")
    ax_scatter.set_xlabel("tax due in current region")
    ax_scatter.set_ylabel("tax due in other region")
    ax_scatter.grid(True, alpha=0.25)

    fig.suptitle("Tax Context of Travel Decisions", fontsize=16, fontweight="bold")
    return fig, df


def relocation_tax_event_study_table(
    log_or_run,
    period=100,
    rate_disc=0.05,
    pre_periods=3,
    post_periods=3,
):
    """Build an event-study table of planner tax schedules around travel events."""
    import numpy as np
    import pandas as pd

    rows = []
    for rollout_id, log in _dense_log_items(log_or_run):
        schedules, _ = _tax_schedules_by_region_period(
            log,
            period=period,
            rate_disc=rate_disc,
        )
        states = log.get("states", [])
        if not states:
            continue
        waterline = _infer_waterline(log)

        for event_idx, event in enumerate(_iter_travel_events(log)):
            if not isinstance(event, dict) or "agent" not in event:
                continue
            event_t = int(event.get("t", 0))
            event_period = int(event_t // int(period)) + 1
            aid = int(event["agent"])

            if "from" in event:
                origin_region = _region_from_loc(event["from"], waterline)
            else:
                before_t = max(0, min(event_t - 1, len(states) - 1))
                origin_region = _location_region_from_state(states[before_t][str(aid)], waterline=waterline)

            if "to" in event:
                destination_region = _region_from_loc(event["to"], waterline)
            else:
                after_t = max(0, min(event_t, len(states) - 1))
                destination_region = _location_region_from_state(states[after_t][str(aid)], waterline=waterline)

            if origin_region == destination_region:
                continue

            for rel_period in range(-int(pre_periods), int(post_periods) + 1):
                tax_period = event_period + rel_period
                if tax_period < 1:
                    continue
                period_schedules = schedules.get(tax_period, {})
                for response_role, region in [
                    ("origin", origin_region),
                    ("destination", destination_region),
                ]:
                    schedule = np.asarray(period_schedules.get(region, []), dtype=float)
                    if schedule.size == 0:
                        continue
                    rows.append({
                        "rollout_id": rollout_id,
                        "event_id": f"{rollout_id}_{event_idx}",
                        "agent": aid,
                        "event_timestep": event_t,
                        "event_tax_period": event_period,
                        "tax_period": tax_period,
                        "relative_period": rel_period,
                        "response_role": response_role,
                        "planner_region": region,
                        "avg_marginal_rate": float(np.nanmean(schedule)),
                        "top_marginal_rate": float(schedule[-1]),
                        "bottom_marginal_rate": float(schedule[0]),
                        "progressivity": float(schedule[-1] - schedule[0]),
                    })

    return pd.DataFrame(rows)


def plot_relocation_tax_event_study(
    log_or_run,
    period=100,
    rate_disc=0.05,
    pre_periods=3,
    post_periods=3,
    figsize=(12, 6),
):
    """
    Event study of planner tax schedules around relocation.

    The origin line follows the planner in the region an agent leaves; the
    destination line follows the planner in the region the agent enters.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = relocation_tax_event_study_table(
        log_or_run,
        period=period,
        rate_disc=rate_disc,
        pre_periods=pre_periods,
        post_periods=post_periods,
    )
    if df.empty:
        raise ValueError("No relocation event-study rows could be constructed.")

    summary = (
        df
        .groupby(["relative_period", "response_role"], as_index=False)
        .agg(
            avg_marginal_rate=("avg_marginal_rate", "mean"),
            top_marginal_rate=("top_marginal_rate", "mean"),
            progressivity=("progressivity", "mean"),
            n_events=("event_id", "nunique"),
        )
    )

    colors = {"origin": "#4c78a8", "destination": "#f58518"}
    metrics = [
        ("avg_marginal_rate", "Average marginal tax rate"),
        ("top_marginal_rate", "Top-bracket marginal tax rate"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, constrained_layout=True)

    for ax, (metric, title) in zip(axes, metrics):
        for role in ["origin", "destination"]:
            dfr = summary[summary["response_role"] == role].sort_values("relative_period")
            ax.plot(
                dfr["relative_period"],
                dfr[metric],
                marker="o",
                linewidth=2.2,
                color=colors[role],
                label=role,
            )
        ax.axvline(0, color="0.25", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("tax periods relative to travel")
        ax.set_ylabel("tax rate")
        ax.grid(True, alpha=0.3)

    axes[0].legend(title="planner role", frameon=True)
    fig.suptitle("Planner Tax Response Around Relocation Events", fontsize=15, fontweight="bold")
    return fig, df, summary


def regional_composition_tax_coevolution_table(
    log_or_run,
    period=100,
    visible_radius=5,
    income_window=100,
    rate_disc=0.05,
    high_income_quantile=0.75,
):
    """Build tax-period regional composition, planner reward, and tax table."""
    import numpy as np
    import pandas as pd

    rows = []
    for rollout_id, log in _dense_log_items(log_or_run):
        period_df = extract_tax_period_travel_context_table(
            log,
            period=period,
            visible_radius=visible_radius,
            income_window=income_window,
        ).copy()
        if period_df.empty:
            continue
        schedules, _ = _tax_schedules_by_region_period(
            log,
            period=period,
            rate_disc=rate_disc,
        )
        income_values = period_df["period_coin_change"].to_numpy(dtype=float)
        finite_income = income_values[np.isfinite(income_values)]
        high_income_cutoff = (
            float(np.nanquantile(finite_income, high_income_quantile))
            if finite_income.size
            else np.nan
        )
        planner_rewards = {
            "top": np.asarray(log.get("planner_rewards", {}).get("p_top", []), dtype=float),
            "bottom": np.asarray(log.get("planner_rewards", {}).get("p_bottom", []), dtype=float),
        }

        for tax_period in sorted(period_df["tax_period"].dropna().unique()):
            tax_period_int = int(tax_period)
            start_t = (tax_period_int - 1) * int(period)
            end_t = tax_period_int * int(period)
            period_rows = period_df[period_df["tax_period"] == tax_period]
            period_schedules = schedules.get(tax_period_int, {})

            for region, planner_id in [("top", "p_top"), ("bottom", "p_bottom")]:
                dfr = period_rows[period_rows["location_region_end"] == region]
                schedule = np.asarray(period_schedules.get(region, []), dtype=float)
                reward_arr = planner_rewards[region]
                reward_slice = reward_arr[start_t:min(end_t, len(reward_arr))]
                reward_slice = reward_slice[np.isfinite(reward_slice)]

                period_income = dfr["period_coin_change"].to_numpy(dtype=float) if len(dfr) else np.asarray([])
                high_income_agents = (
                    int(np.sum(period_income >= high_income_cutoff))
                    if len(period_income) and np.isfinite(high_income_cutoff)
                    else 0
                )
                rows.append({
                    "rollout_id": rollout_id,
                    "tax_period": tax_period_int,
                    "planner_region": region,
                    "planner_id": planner_id,
                    "n_agents": int(len(dfr)),
                    "n_travelers": int(dfr["did_travel"].sum()) if len(dfr) else 0,
                    "high_income_agents": high_income_agents,
                    "mean_period_coin_change": float(np.nanmean(period_income)) if len(period_income) else np.nan,
                    "total_period_coin_change": float(np.nansum(period_income)) if len(period_income) else np.nan,
                    "avg_marginal_rate": float(np.nanmean(schedule)) if schedule.size else np.nan,
                    "top_marginal_rate": float(schedule[-1]) if schedule.size else np.nan,
                    "progressivity": float(schedule[-1] - schedule[0]) if schedule.size else np.nan,
                    "planner_reward_sum": float(np.sum(reward_slice)) if reward_slice.size else np.nan,
                    "planner_reward_mean": float(np.mean(reward_slice)) if reward_slice.size else np.nan,
                    "high_income_cutoff": high_income_cutoff,
                })

    return pd.DataFrame(rows)


def plot_regional_composition_tax_coevolution(
    log_or_run,
    period=100,
    visible_radius=5,
    income_window=100,
    rate_disc=0.05,
    high_income_quantile=0.75,
    figsize=(13, 9),
):
    """Plot regional composition, income, planner taxes, and planner rewards over time."""
    import pandas as pd
    import matplotlib.pyplot as plt

    df = regional_composition_tax_coevolution_table(
        log_or_run,
        period=period,
        visible_radius=visible_radius,
        income_window=income_window,
        rate_disc=rate_disc,
        high_income_quantile=high_income_quantile,
    )
    if df.empty:
        raise ValueError("No regional co-evolution rows could be constructed.")

    summary = (
        df
        .groupby(["tax_period", "planner_region"], as_index=False)
        .agg(
            n_agents=("n_agents", "mean"),
            n_travelers=("n_travelers", "mean"),
            high_income_agents=("high_income_agents", "mean"),
            mean_period_coin_change=("mean_period_coin_change", "mean"),
            total_period_coin_change=("total_period_coin_change", "mean"),
            avg_marginal_rate=("avg_marginal_rate", "mean"),
            top_marginal_rate=("top_marginal_rate", "mean"),
            planner_reward_sum=("planner_reward_sum", "mean"),
        )
    )

    colors = {"top": "#1f77b4", "bottom": "#ff7f0e"}
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)

    panels = [
        ("high_income_agents", "High-income agents in region"),
        ("mean_period_coin_change", "Mean period income / coin change"),
        ("planner_reward_sum", "Planner reward over period"),
    ]

    for ax, (metric, ylabel) in zip(axes[:3], panels):
        for region in ["top", "bottom"]:
            dfr = summary[summary["planner_region"] == region].sort_values("tax_period")
            ax.plot(
                dfr["tax_period"],
                dfr[metric],
                marker="o",
                linewidth=2,
                color=colors[region],
                label=region,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    ax_tax = axes[3]
    for region in ["top", "bottom"]:
        dfr = summary[summary["planner_region"] == region].sort_values("tax_period")
        ax_tax.plot(
            dfr["tax_period"],
            dfr["avg_marginal_rate"],
            marker="o",
            linewidth=2,
            color=colors[region],
            label=f"{region} avg",
        )
        ax_tax.plot(
            dfr["tax_period"],
            dfr["top_marginal_rate"],
            marker="^",
            linewidth=1.7,
            linestyle="--",
            color=colors[region],
            alpha=0.85,
            label=f"{region} top bracket",
        )
    ax_tax.set_ylabel("tax rate")
    ax_tax.set_xlabel("tax period")
    ax_tax.grid(True, alpha=0.3)

    axes[0].legend(title="region", ncol=2, frameon=True)
    ax_tax.legend(title="tax series", ncol=2, frameon=True)
    fig.suptitle("Regional Composition and Planner Tax Co-Evolution", fontsize=15, fontweight="bold")
    return fig, df, summary


def travel_regression_table_for_run(
    run,
    period=100,
    visible_radius=5,
    income_window=100,
    cluster_by="agent",
):
    """Build the pooled agent-period regression table for all dense logs in a run."""
    import pandas as pd

    logs = _extract_logs_from_run(run)
    if not logs:
        raise ValueError("No dense logs found in run.")

    frames = []
    for log_key, log in logs.items():
        df = extract_tax_period_travel_context_table(
            log,
            period=period,
            visible_radius=visible_radius,
            income_window=income_window,
        ).copy()
        if df.empty:
            continue
        df["rollout_id"] = log_key
        df["agent_cluster"] = df["agent"].astype(str)
        df["rollout_agent_cluster"] = df["rollout_id"].astype(str) + "_a" + df["agent"].astype(str)
        frames.append(df)

    if not frames:
        raise ValueError("No agent-period rows could be constructed from this run.")

    out = pd.concat(frames, ignore_index=True)
    out["did_travel_int"] = out["did_travel"].astype(int)
    if cluster_by == "agent":
        out["cluster_id"] = out["agent_cluster"]
    elif cluster_by == "rollout_agent":
        out["cluster_id"] = out["rollout_agent_cluster"]
    else:
        raise ValueError("cluster_by must be 'agent' or 'rollout_agent'.")
    return out


def travel_tax_regression_table_for_run(
    run,
    period=100,
    visible_radius=5,
    income_window=100,
    rate_disc=0.05,
    cluster_by="agent",
):
    """Build a pooled agent-period regression table with tax counterfactuals."""
    import pandas as pd

    logs = _extract_logs_from_run(run)
    if not logs:
        raise ValueError("No dense logs found in run.")

    frames = []
    for log_key, log in logs.items():
        df = tax_period_travel_counterfactual_table(
            log,
            period=period,
            visible_radius=visible_radius,
            income_window=income_window,
            rate_disc=rate_disc,
        ).copy()
        if df.empty:
            continue
        df["rollout_id"] = log_key
        df["agent_cluster"] = df["agent"].astype(str)
        df["rollout_agent_cluster"] = df["rollout_id"].astype(str) + "_a" + df["agent"].astype(str)
        frames.append(df)

    if not frames:
        raise ValueError("No agent-period tax rows could be constructed from this run.")

    out = pd.concat(frames, ignore_index=True)
    out["did_travel_int"] = out["did_travel"].astype(int)
    if cluster_by == "agent":
        out["cluster_id"] = out["agent_cluster"]
    elif cluster_by == "rollout_agent":
        out["cluster_id"] = out["rollout_agent_cluster"]
    else:
        raise ValueError("cluster_by must be 'agent' or 'rollout_agent'.")
    return out


def plot_travel_probability_regression(
    run,
    period=100,
    visible_radius=5,
    income_window=100,
    cluster_by="agent",
    include_tax_period_fixed_effects=True,
    include_agent_fixed_effects=False,
    figsize=(16, 11),
):
    """
    Fit logistic regressions for travel probability across all dense logs in a run.

    Returns
    -------
    fig, model_tables, regression_df, results
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    try:
        import statsmodels.api as sm
    except ModuleNotFoundError:
        sm = None

    df = travel_regression_table_for_run(
        run,
        period=period,
        visible_radius=visible_radius,
        income_window=income_window,
        cluster_by=cluster_by,
    )

    base_features = [
        "mean_coin",
        "period_coin_change",
        "mean_visible_resources",
        "mean_visible_other_houses",
        "mean_visible_own_houses",
        "skill_build_payment",
    ]
    within_agent_features = [
        "mean_coin",
        "period_coin_change",
        "mean_visible_resources",
        "mean_visible_other_houses",
        "mean_visible_own_houses",
    ]
    feature_labels = {
        "mean_coin": "Mean coin",
        "period_coin_change": "Period income / coin change",
        "mean_visible_resources": "Visible resources",
        "mean_visible_other_houses": "Visible other houses",
        "mean_visible_own_houses": "Visible own houses",
        "skill_build_payment": "Skill",
    }

    model_features = within_agent_features if include_agent_fixed_effects else base_features
    model_cols = [
        "did_travel_int",
        "tax_period",
        "agent",
        "cluster_id",
        *model_features,
    ]
    model_df = df[model_cols].copy()
    model_df["did_travel_int"] = pd.to_numeric(model_df["did_travel_int"], errors="coerce")
    for feature in model_features:
        model_df[feature] = pd.to_numeric(model_df[feature], errors="coerce")
    model_df = model_df.dropna(subset=model_cols).copy()
    if model_df["did_travel_int"].nunique() < 2:
        event_rate = float(model_df["did_travel_int"].mean()) if len(model_df) else np.nan
        coef_df = pd.DataFrame([{
            "feature": feature,
            "label": feature_labels[feature],
            "coef": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "odds_ratio": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        } for feature in model_features])
        fit_df = pd.DataFrame([{
            "n_obs": int(len(model_df)),
            "n_events": int(model_df["did_travel_int"].sum()) if len(model_df) else 0,
            "event_rate": event_rate,
            "n_clusters": int(model_df["cluster_id"].nunique()) if len(model_df) else 0,
            "mcfadden_pseudo_r2": np.nan,
            "log_likelihood": np.nan,
            "ll_null": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "auc": np.nan,
            "tax_period_fixed_effects": bool(include_tax_period_fixed_effects),
            "agent_fixed_effects": bool(include_agent_fixed_effects),
            "clustered_se_by": cluster_by,
            "note": "not estimated: did_travel has no variation after filtering",
        }])

        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.axis("off")
        message = (
            "Logistic regression was not estimated.\n\n"
            "The dependent variable did_travel has no variation after filtering:\n"
            f"observations = {fit_df.at[0, 'n_obs']}, "
            f"travel events = {fit_df.at[0, 'n_events']}, "
            f"event rate = {event_rate:.3f}.\n\n"
            "A logistic model needs both travel and non-travel rows to estimate coefficients."
        )
        ax.text(0.02, 0.72, message, va="top", ha="left", fontsize=12)
        fig.suptitle("Logistic Regression: Probability of Travel", fontsize=14, fontweight="bold")
        fit_summary = pd.DataFrame([
            ["Observations", f"{fit_df.at[0, 'n_obs']}"],
            ["Travel events", f"{fit_df.at[0, 'n_events']}"],
            ["Event rate", f"{fit_df.at[0, 'event_rate']:.3f}"],
            ["Clusters", f"{fit_df.at[0, 'n_clusters']} ({cluster_by})"],
            ["Pseudo-R2 (McFadden)", "not estimated"],
            ["AUC", "not estimated"],
            ["Tax-period FE", "yes" if include_tax_period_fixed_effects else "no"],
            ["Agent FE", "yes" if include_agent_fixed_effects else "no"],
        ], columns=["statistic", "value"])
        model_tables = {"coefficients": coef_df, "fit": fit_df, "fit_summary": fit_summary}
        results = {"main": None, "X": None, "y": None, "model_df": model_df}
        return fig, model_tables, df, results

    for feature in model_features:
        std = float(model_df[feature].std())
        mean = float(model_df[feature].mean())
        model_df[f"z_{feature}"] = 0.0 if std <= 1e-12 else (model_df[feature] - mean) / std

    x_parts = [model_df[[f"z_{feature}" for feature in model_features]]]
    if include_tax_period_fixed_effects:
        x_parts.append(pd.get_dummies(model_df["tax_period"].astype(int), prefix="period", drop_first=True, dtype=float))
    if include_agent_fixed_effects:
        x_parts.append(pd.get_dummies(model_df["agent"].astype(int), prefix="agent", drop_first=True, dtype=float))
    X = pd.concat(x_parts, axis=1)
    if sm is not None:
        X = sm.add_constant(X, has_constant="add")
    else:
        X.insert(0, "const", 1.0)
    y = model_df["did_travel_int"].astype(float)

    if sm is not None:
        model = sm.Logit(y, X)
        try:
            result = model.fit(
                disp=False,
                maxiter=300,
                cov_type="cluster",
                cov_kwds={"groups": model_df["cluster_id"]},
            )
        except Exception:
            result = model.fit(disp=False, maxiter=300)
        params = result.params
        pvalues = result.pvalues
        conf = result.conf_int()
        pred = np.asarray(result.predict(X), dtype=float)
        llf = float(result.llf)
        llnull = float(result.llnull) if hasattr(result, "llnull") else np.nan
        pseudo_r2 = float(getattr(result, "prsquared", np.nan))
        aic = float(result.aic)
        bic = float(result.bic)
        result_obj = result
    else:
        import math

        def normal_cdf(x):
            return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))

        def sigmoid(z):
            z = np.clip(z, -35, 35)
            return 1.0 / (1.0 + np.exp(-z))

        X_np = X.to_numpy(dtype=float)
        y_np = y.to_numpy(dtype=float)
        beta = np.zeros(X_np.shape[1], dtype=float)
        ridge = 1e-8

        for _ in range(300):
            eta = X_np @ beta
            prob = sigmoid(eta)
            w = np.clip(prob * (1.0 - prob), 1e-8, None)
            hessian = X_np.T @ (X_np * w[:, None])
            grad = X_np.T @ (y_np - prob)
            step = np.linalg.solve(
                hessian + ridge * np.eye(hessian.shape[0]),
                grad,
            )
            beta_new = beta + step
            if np.max(np.abs(step)) < 1e-8:
                beta = beta_new
                break
            beta = beta_new

        pred = sigmoid(X_np @ beta)
        w = np.clip(pred * (1.0 - pred), 1e-8, None)
        bread = np.linalg.pinv(X_np.T @ (X_np * w[:, None]))
        scores = X_np * (y_np - pred)[:, None]
        meat = np.zeros((X_np.shape[1], X_np.shape[1]), dtype=float)
        groups = model_df["cluster_id"].to_numpy()
        unique_groups = pd.unique(groups)
        for group in unique_groups:
            sg = scores[groups == group].sum(axis=0)
            meat += np.outer(sg, sg)
        cov = bread @ meat @ bread
        n_obs = X_np.shape[0]
        n_params = X_np.shape[1]
        n_clusters = len(unique_groups)
        if n_clusters > 1 and n_obs > n_params:
            cov *= (n_clusters / (n_clusters - 1.0)) * ((n_obs - 1.0) / (n_obs - n_params))
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        z_vals = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        p_vals = np.asarray([2.0 * (1.0 - normal_cdf(abs(z))) for z in z_vals])
        ci_low = beta - 1.96 * se
        ci_high = beta + 1.96 * se

        params = pd.Series(beta, index=X.columns)
        pvalues = pd.Series(p_vals, index=X.columns)
        conf = pd.DataFrame({0: ci_low, 1: ci_high}, index=X.columns)
        eps = 1e-12
        llf = float(np.sum(y_np * np.log(pred + eps) + (1.0 - y_np) * np.log(1.0 - pred + eps)))
        mean_y = np.clip(float(np.mean(y_np)), eps, 1.0 - eps)
        llnull = float(np.sum(y_np * np.log(mean_y) + (1.0 - y_np) * np.log(1.0 - mean_y)))
        pseudo_r2 = float(1.0 - llf / llnull) if llnull != 0 else np.nan
        aic = float(2 * n_params - 2 * llf)
        bic = float(np.log(n_obs) * n_params - 2 * llf)
        result_obj = {
            "params": params,
            "pvalues": pvalues,
            "conf_int": conf,
            "cov_cluster": cov,
            "method": "numpy_logit_cluster_fallback",
        }

    def predict_from_X(X_like):
        if sm is not None:
            return np.asarray(result_obj.predict(X_like), dtype=float)
        values = X_like.to_numpy(dtype=float) if hasattr(X_like, "to_numpy") else np.asarray(X_like, dtype=float)
        beta = params.reindex(X.columns).to_numpy(dtype=float)
        z = np.clip(values @ beta, -35, 35)
        return 1.0 / (1.0 + np.exp(-z))

    coef_rows = []
    for feature in model_features:
        name = f"z_{feature}"
        coef = float(params[name])
        ci_low, ci_high = conf.loc[name].astype(float).tolist()
        coef_rows.append({
            "feature": feature,
            "label": feature_labels[feature],
            "coef": coef,
            "odds_ratio": float(np.exp(coef)),
            "ci_low": float(np.exp(ci_low)),
            "ci_high": float(np.exp(ci_high)),
            "p_value": float(pvalues[name]),
        })
    coef_df = pd.DataFrame(coef_rows).sort_values("odds_ratio")

    y_arr = y.to_numpy(dtype=float)
    # AUC via rank statistic, avoiding a hard dependency on sklearn.
    pos = pred[y_arr == 1]
    neg = pred[y_arr == 0]
    if len(pos) and len(neg):
        ranks = pd.Series(pred).rank(method="average").to_numpy(dtype=float)
        auc = float((np.sum(ranks[y_arr == 1]) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    else:
        auc = np.nan

    fit_df = pd.DataFrame([{
        "n_obs": int(len(model_df)),
        "n_events": int(y.sum()),
        "event_rate": float(y.mean()),
        "n_clusters": int(model_df["cluster_id"].nunique()),
        "mcfadden_pseudo_r2": pseudo_r2,
        "log_likelihood": llf,
        "ll_null": llnull,
        "aic": aic,
        "bic": bic,
        "auc": auc,
        "tax_period_fixed_effects": bool(include_tax_period_fixed_effects),
        "agent_fixed_effects": bool(include_agent_fixed_effects),
        "clustered_se_by": cluster_by,
    }])

    if figsize == (16, 11):
        figsize = (14, 5.5)
    fig, (ax_or, ax_p) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Plot 1: odds ratios with clustered confidence intervals.
    y_pos = np.arange(len(coef_df))
    ax_or.errorbar(
        coef_df["odds_ratio"],
        y_pos,
        xerr=[
            coef_df["odds_ratio"] - coef_df["ci_low"],
            coef_df["ci_high"] - coef_df["odds_ratio"],
        ],
        fmt="o",
        color="#1f77b4",
        ecolor="0.35",
        capsize=4,
    )
    ax_or.axvline(1.0, color="0.25", linestyle="--", linewidth=1)
    ax_or.set_yticks(y_pos)
    ax_or.set_yticklabels(coef_df["label"])
    ax_or.set_xscale("log")
    ax_or.set_xlabel("Odds ratio for +1 std increase")
    ax_or.set_title("Variable Significance: Odds Ratios")
    ax_or.grid(True, axis="x", alpha=0.25)

    # Plot 2: p-values.
    p_df = coef_df.sort_values("p_value", ascending=False)
    colors = ["#c92a2a" if p < 0.05 else "0.65" for p in p_df["p_value"]]
    ax_p.barh(p_df["label"], p_df["p_value"], color=colors, alpha=0.85)
    ax_p.axvline(0.05, color="0.2", linestyle="--", linewidth=1, label="p = 0.05")
    ax_p.set_xlim(0, min(1.0, max(0.1, float(np.nanmax(p_df["p_value"])) * 1.15)))
    ax_p.set_xlabel("Cluster-robust p-value")
    ax_p.set_title("Statistical Significance")
    ax_p.legend(frameon=True)
    ax_p.grid(True, axis="x", alpha=0.25)

    fit_summary = pd.DataFrame([
        ["Observations", f"{fit_df.at[0, 'n_obs']}"],
        ["Travel events", f"{fit_df.at[0, 'n_events']}"],
        ["Event rate", f"{fit_df.at[0, 'event_rate']:.3f}"],
        ["Clusters", f"{fit_df.at[0, 'n_clusters']} ({cluster_by})"],
        ["Pseudo-R2 (McFadden)", f"{fit_df.at[0, 'mcfadden_pseudo_r2']:.3f}"],
        ["AUC", f"{fit_df.at[0, 'auc']:.3f}"],
        ["Tax-period FE", "yes" if include_tax_period_fixed_effects else "no"],
        ["Agent FE", "yes" if include_agent_fixed_effects else "no"],
    ], columns=["statistic", "value"])

    fig.suptitle("Logistic Regression: Probability of Travel", fontsize=16, fontweight="bold")
    model_tables = {"coefficients": coef_df, "fit": fit_df, "fit_summary": fit_summary}
    results = {"main": result_obj, "X": X, "y": y, "model_df": model_df}
    return fig, model_tables, df, results


def plot_travel_probability_regression_tax(
    run,
    period=100,
    visible_radius=5,
    income_window=100,
    rate_disc=0.05,
    cluster_by="agent",
    include_tax_period_fixed_effects=True,
    include_agent_fixed_effects=False,
    figsize=(14, 5.5),
):
    """
    Logistic regression for travel probability using tax counterfactual variables.

    The regressors summarize the tax context from ``plot_tax_travel_counterfactuals``:
    the average marginal tax rate faced in the current region and the difference
    between tax due in the other region and tax due in the current region.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    try:
        import statsmodels.api as sm
    except ModuleNotFoundError:
        sm = None

    df = travel_tax_regression_table_for_run(
        run,
        period=period,
        visible_radius=visible_radius,
        income_window=income_window,
        rate_disc=rate_disc,
        cluster_by=cluster_by,
    )

    model_features = [
        "current_region_avg_tax",
        "other_minus_current_tax_due",
    ]
    feature_labels = {
        "current_region_avg_tax": "Average tax faced",
        "other_minus_current_tax_due": "Other minus current tax due",
    }

    model_cols = [
        "did_travel_int",
        "tax_period",
        "agent",
        "cluster_id",
        *model_features,
    ]
    model_df = df[model_cols].copy()
    model_df["did_travel_int"] = pd.to_numeric(model_df["did_travel_int"], errors="coerce")
    for feature in model_features:
        model_df[feature] = pd.to_numeric(model_df[feature], errors="coerce")
    model_df = model_df.dropna(subset=model_cols).copy()

    if model_df["did_travel_int"].nunique() < 2:
        event_rate = float(model_df["did_travel_int"].mean()) if len(model_df) else np.nan
        coef_df = pd.DataFrame([{
            "feature": feature,
            "label": feature_labels[feature],
            "coef": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "odds_ratio": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        } for feature in model_features])
        fit_df = pd.DataFrame([{
            "n_obs": int(len(model_df)),
            "n_events": int(model_df["did_travel_int"].sum()) if len(model_df) else 0,
            "event_rate": event_rate,
            "n_clusters": int(model_df["cluster_id"].nunique()) if len(model_df) else 0,
            "mcfadden_pseudo_r2": np.nan,
            "log_likelihood": np.nan,
            "ll_null": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "auc": np.nan,
            "tax_period_fixed_effects": bool(include_tax_period_fixed_effects),
            "agent_fixed_effects": bool(include_agent_fixed_effects),
            "clustered_se_by": cluster_by,
            "note": "not estimated: did_travel has no variation after filtering",
        }])
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.axis("off")
        ax.text(
            0.02,
            0.72,
            "Logistic regression was not estimated.\n\n"
            "The dependent variable did_travel has no variation after filtering:\n"
            f"observations = {fit_df.at[0, 'n_obs']}, "
            f"travel events = {fit_df.at[0, 'n_events']}, "
            f"event rate = {event_rate:.3f}.",
            va="top",
            ha="left",
            fontsize=12,
        )
        fig.suptitle("Tax Regression: Probability of Travel", fontsize=14, fontweight="bold")
        fit_summary = pd.DataFrame([
            ["Observations", f"{fit_df.at[0, 'n_obs']}"],
            ["Travel events", f"{fit_df.at[0, 'n_events']}"],
            ["Event rate", f"{fit_df.at[0, 'event_rate']:.3f}"],
            ["Clusters", f"{fit_df.at[0, 'n_clusters']} ({cluster_by})"],
            ["Pseudo-R2 (McFadden)", "not estimated"],
            ["AUC", "not estimated"],
            ["Tax-period FE", "yes" if include_tax_period_fixed_effects else "no"],
            ["Agent FE", "yes" if include_agent_fixed_effects else "no"],
        ], columns=["statistic", "value"])
        model_tables = {"coefficients": coef_df, "fit": fit_df, "fit_summary": fit_summary}
        results = {"main": None, "X": None, "y": None, "model_df": model_df}
        return fig, model_tables, df, results

    for feature in model_features:
        std = float(model_df[feature].std())
        mean = float(model_df[feature].mean())
        model_df[f"z_{feature}"] = 0.0 if std <= 1e-12 else (model_df[feature] - mean) / std

    x_parts = [model_df[[f"z_{feature}" for feature in model_features]]]
    if include_tax_period_fixed_effects:
        x_parts.append(pd.get_dummies(model_df["tax_period"].astype(int), prefix="period", drop_first=True, dtype=float))
    if include_agent_fixed_effects:
        x_parts.append(pd.get_dummies(model_df["agent"].astype(int), prefix="agent", drop_first=True, dtype=float))
    X = pd.concat(x_parts, axis=1)
    if sm is not None:
        X = sm.add_constant(X, has_constant="add")
    else:
        X.insert(0, "const", 1.0)
    y = model_df["did_travel_int"].astype(float)

    if sm is not None:
        model = sm.Logit(y, X)
        try:
            result = model.fit(
                disp=False,
                maxiter=300,
                cov_type="cluster",
                cov_kwds={"groups": model_df["cluster_id"]},
            )
        except Exception:
            result = model.fit(disp=False, maxiter=300)
        params = result.params
        pvalues = result.pvalues
        conf = result.conf_int()
        pred = np.asarray(result.predict(X), dtype=float)
        llf = float(result.llf)
        llnull = float(result.llnull) if hasattr(result, "llnull") else np.nan
        pseudo_r2 = float(getattr(result, "prsquared", np.nan))
        aic = float(result.aic)
        bic = float(result.bic)
        result_obj = result
    else:
        import math

        def normal_cdf(x):
            return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))

        def sigmoid(z):
            z = np.clip(z, -35, 35)
            return 1.0 / (1.0 + np.exp(-z))

        X_np = X.to_numpy(dtype=float)
        y_np = y.to_numpy(dtype=float)
        beta = np.zeros(X_np.shape[1], dtype=float)
        ridge = 1e-8

        for _ in range(300):
            eta = X_np @ beta
            prob = sigmoid(eta)
            w = np.clip(prob * (1.0 - prob), 1e-8, None)
            hessian = X_np.T @ (X_np * w[:, None])
            grad = X_np.T @ (y_np - prob)
            step = np.linalg.solve(
                hessian + ridge * np.eye(hessian.shape[0]),
                grad,
            )
            beta_new = beta + step
            if np.max(np.abs(step)) < 1e-8:
                beta = beta_new
                break
            beta = beta_new

        pred = sigmoid(X_np @ beta)
        w = np.clip(pred * (1.0 - pred), 1e-8, None)
        bread = np.linalg.pinv(X_np.T @ (X_np * w[:, None]))
        scores = X_np * (y_np - pred)[:, None]
        meat = np.zeros((X_np.shape[1], X_np.shape[1]), dtype=float)
        groups = model_df["cluster_id"].to_numpy()
        unique_groups = pd.unique(groups)
        for group in unique_groups:
            sg = scores[groups == group].sum(axis=0)
            meat += np.outer(sg, sg)
        cov = bread @ meat @ bread
        n_obs = X_np.shape[0]
        n_params = X_np.shape[1]
        n_clusters = len(unique_groups)
        if n_clusters > 1 and n_obs > n_params:
            cov *= (n_clusters / (n_clusters - 1.0)) * ((n_obs - 1.0) / (n_obs - n_params))
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        z_vals = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        p_vals = np.asarray([2.0 * (1.0 - normal_cdf(abs(z))) for z in z_vals])
        ci_low = beta - 1.96 * se
        ci_high = beta + 1.96 * se

        params = pd.Series(beta, index=X.columns)
        pvalues = pd.Series(p_vals, index=X.columns)
        conf = pd.DataFrame({0: ci_low, 1: ci_high}, index=X.columns)
        eps = 1e-12
        llf = float(np.sum(y_np * np.log(pred + eps) + (1.0 - y_np) * np.log(1.0 - pred + eps)))
        mean_y = np.clip(float(np.mean(y_np)), eps, 1.0 - eps)
        llnull = float(np.sum(y_np * np.log(mean_y) + (1.0 - y_np) * np.log(1.0 - mean_y)))
        pseudo_r2 = float(1.0 - llf / llnull) if llnull != 0 else np.nan
        aic = float(2 * n_params - 2 * llf)
        bic = float(np.log(n_obs) * n_params - 2 * llf)
        result_obj = {
            "params": params,
            "pvalues": pvalues,
            "conf_int": conf,
            "cov_cluster": cov,
            "method": "numpy_logit_cluster_fallback",
        }

    coef_rows = []
    for feature in model_features:
        name = f"z_{feature}"
        coef = float(params[name])
        ci_low, ci_high = conf.loc[name].astype(float).tolist()
        coef_rows.append({
            "feature": feature,
            "label": feature_labels[feature],
            "coef": coef,
            "odds_ratio": float(np.exp(coef)),
            "ci_low": float(np.exp(ci_low)),
            "ci_high": float(np.exp(ci_high)),
            "p_value": float(pvalues[name]),
        })
    coef_df = pd.DataFrame(coef_rows).sort_values("odds_ratio")

    y_arr = y.to_numpy(dtype=float)
    pos = pred[y_arr == 1]
    neg = pred[y_arr == 0]
    if len(pos) and len(neg):
        ranks = pd.Series(pred).rank(method="average").to_numpy(dtype=float)
        auc = float((np.sum(ranks[y_arr == 1]) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    else:
        auc = np.nan

    fit_df = pd.DataFrame([{
        "n_obs": int(len(model_df)),
        "n_events": int(y.sum()),
        "event_rate": float(y.mean()),
        "n_clusters": int(model_df["cluster_id"].nunique()),
        "mcfadden_pseudo_r2": pseudo_r2,
        "log_likelihood": llf,
        "ll_null": llnull,
        "aic": aic,
        "bic": bic,
        "auc": auc,
        "tax_period_fixed_effects": bool(include_tax_period_fixed_effects),
        "agent_fixed_effects": bool(include_agent_fixed_effects),
        "clustered_se_by": cluster_by,
    }])

    fig, (ax_or, ax_p) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    y_pos = np.arange(len(coef_df))
    ax_or.errorbar(
        coef_df["odds_ratio"],
        y_pos,
        xerr=[
            coef_df["odds_ratio"] - coef_df["ci_low"],
            coef_df["ci_high"] - coef_df["odds_ratio"],
        ],
        fmt="o",
        color="#1f77b4",
        ecolor="0.35",
        capsize=4,
    )
    ax_or.axvline(1.0, color="0.25", linestyle="--", linewidth=1)
    ax_or.set_yticks(y_pos)
    ax_or.set_yticklabels(coef_df["label"])
    ax_or.set_xscale("log")
    ax_or.set_xlabel("Odds ratio for +1 std increase")
    ax_or.set_title("Tax Variables: Odds Ratios")
    ax_or.grid(True, axis="x", alpha=0.25)

    p_df = coef_df.sort_values("p_value", ascending=False)
    colors = ["#c92a2a" if p < 0.05 else "0.65" for p in p_df["p_value"]]
    ax_p.barh(p_df["label"], p_df["p_value"], color=colors, alpha=0.85)
    ax_p.axvline(0.05, color="0.2", linestyle="--", linewidth=1, label="p = 0.05")
    ax_p.set_xlim(0, min(1.0, max(0.1, float(np.nanmax(p_df["p_value"])) * 1.15)))
    ax_p.set_xlabel("Cluster-robust p-value")
    ax_p.set_title("Statistical Significance")
    ax_p.legend(frameon=True)
    ax_p.grid(True, axis="x", alpha=0.25)

    fit_summary = pd.DataFrame([
        ["Observations", f"{fit_df.at[0, 'n_obs']}"],
        ["Travel events", f"{fit_df.at[0, 'n_events']}"],
        ["Event rate", f"{fit_df.at[0, 'event_rate']:.3f}"],
        ["Clusters", f"{fit_df.at[0, 'n_clusters']} ({cluster_by})"],
        ["Pseudo-R2 (McFadden)", f"{fit_df.at[0, 'mcfadden_pseudo_r2']:.3f}"],
        ["AUC", f"{fit_df.at[0, 'auc']:.3f}"],
        ["Tax-period FE", "yes" if include_tax_period_fixed_effects else "no"],
        ["Agent FE", "yes" if include_agent_fixed_effects else "no"],
    ], columns=["statistic", "value"])

    fig.suptitle("Tax Regression: Probability of Travel", fontsize=16, fontweight="bold")
    model_tables = {"coefficients": coef_df, "fit": fit_df, "fit_summary": fit_summary}
    results = {"main": result_obj, "X": X, "y": y, "model_df": model_df}
    return fig, model_tables, df, results


def _simple_logit_slope_table(df, features, group_col=None):
    """Small no-dependency univariate logit slopes for actual travel probability."""
    import math
    import numpy as np
    import pandas as pd

    def normal_cdf(x):
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))

    def sigmoid(z):
        z = np.clip(z, -35, 35)
        return 1.0 / (1.0 + np.exp(-z))

    groups = [("all", df)] if group_col is None else list(df.groupby(group_col, dropna=False))
    rows = []
    for group_name, dfg in groups:
        y = dfg["did_travel_int"].to_numpy(dtype=float)
        if len(y) < 4 or len(np.unique(y)) < 2:
            for feature in features:
                rows.append({
                    "group": group_name,
                    "feature": feature,
                    "coef": np.nan,
                    "odds_ratio": np.nan,
                    "p_value": np.nan,
                    "n_obs": int(len(y)),
                    "event_rate": float(np.mean(y)) if len(y) else np.nan,
                })
            continue

        for feature in features:
            x_raw = dfg[feature].to_numpy(dtype=float)
            ok = np.isfinite(x_raw) & np.isfinite(y)
            x_raw = x_raw[ok]
            yy = y[ok]
            feature_n = int(len(yy))
            feature_event_rate = float(np.mean(yy)) if feature_n else np.nan
            if len(yy) < 4 or len(np.unique(yy)) < 2 or float(np.std(x_raw)) <= 1e-12:
                coef = odds = p_value = np.nan
            else:
                x = (x_raw - np.mean(x_raw)) / np.std(x_raw)
                X = np.column_stack([np.ones(len(x)), x])
                beta = np.zeros(2, dtype=float)
                for _ in range(100):
                    p = sigmoid(X @ beta)
                    w = np.clip(p * (1.0 - p), 1e-8, None)
                    h = X.T @ (X * w[:, None])
                    g = X.T @ (yy - p)
                    step = np.linalg.solve(h + 1e-8 * np.eye(2), g)
                    beta_new = beta + step
                    if np.max(np.abs(step)) < 1e-8:
                        beta = beta_new
                        break
                    beta = beta_new
                p = sigmoid(X @ beta)
                w = np.clip(p * (1.0 - p), 1e-8, None)
                cov = np.linalg.pinv(X.T @ (X * w[:, None]))
                se = float(np.sqrt(max(cov[1, 1], 0.0)))
                coef = float(beta[1])
                odds = float(np.exp(coef))
                z = 0.0 if se <= 0 else coef / se
                p_value = float(2.0 * (1.0 - normal_cdf(abs(z))))

            rows.append({
                "group": group_name,
                "feature": feature,
                "coef": coef,
                "odds_ratio": odds,
                "p_value": p_value,
                "n_obs": feature_n,
                "event_rate": feature_event_rate,
            })

    return pd.DataFrame(rows)


def plot_travel_probability_by_skill(
    run,
    period=100,
    visible_radius=5,
    income_window=100,
    rate_disc=0.05,
    n_bins=5,
    figsize=(16, 10),
):
    """
    Plot actual observed travel probability by variable, grouped by skill level.

    Returns
    -------
    fig, coefficient_table, probability_table, regression_df
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = travel_tax_regression_table_for_run(
        run,
        period=period,
        visible_radius=visible_radius,
        income_window=income_window,
        rate_disc=rate_disc,
        cluster_by="agent",
    )
    df["did_travel_int"] = pd.to_numeric(df["did_travel_int"], errors="coerce")
    features = [
        "mean_coin",
        "period_coin_change",
        "mean_visible_resources",
        "mean_visible_other_houses",
        "mean_visible_own_houses",
        "current_region_avg_tax",
    ]
    labels = {
        "mean_coin": "Mean coin",
        "period_coin_change": "Period income / coin change",
        "mean_visible_resources": "Visible resources",
        "mean_visible_other_houses": "Visible other houses",
        "mean_visible_own_houses": "Visible own houses",
        "current_region_avg_tax": "Average tax faced",
    }

    valid_outcome = df["did_travel_int"].dropna()
    if valid_outcome.nunique() < 2:
        event_rate = float(valid_outcome.mean()) if len(valid_outcome) else np.nan
        coefficient_table = _simple_logit_slope_table(df, features, group_col=None)
        coefficient_table["feature_label"] = coefficient_table["feature"].map(labels)
        probability_table = pd.DataFrame()
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.axis("off")
        message = (
            "Travel probability by skill was not plotted.\n\n"
            "The selected run/filter has no variation in travel decisions:\n"
            f"observations = {len(valid_outcome)}, "
            f"travel events = {int(valid_outcome.sum()) if len(valid_outcome) else 0}, "
            f"event rate = {event_rate:.3f}.\n\n"
            "This means every included agent-period is either travel or non-travel, "
            "so variable effects by skill cannot be compared."
        )
        ax.text(0.02, 0.72, message, va="top", ha="left", fontsize=12)
        fig.suptitle("Actual Travel Probability by Variable and Skill Level", fontsize=14, fontweight="bold")
        return fig, coefficient_table, probability_table, df

    skill_values = sorted(v for v in df["skill_build_payment"].dropna().unique())
    skill_rank = {v: i + 1 for i, v in enumerate(skill_values)}
    df["skill_group"] = df["skill_build_payment"].map(
        lambda v: f"skill {skill_rank.get(v, '?')} ({v:.0f})" if pd.notna(v) else "skill ?"
    )

    prob_rows = []
    for feature in features:
        valid = df[[feature, "did_travel_int", "skill_group"]].dropna().copy()
        if valid.empty:
            continue
        valid["did_travel_int"] = pd.to_numeric(valid["did_travel_int"], errors="coerce")
        valid[feature] = pd.to_numeric(valid[feature], errors="coerce")
        valid = valid.dropna(subset=[feature, "did_travel_int"])
        if valid.empty:
            continue
        if valid[feature].nunique() <= n_bins:
            valid["_bin"] = valid[feature]
            valid["_bin_mid"] = valid[feature].astype(float)
        else:
            valid["_bin"] = pd.qcut(valid[feature], q=min(n_bins, valid[feature].nunique()), duplicates="drop")
            valid["_bin_mid"] = valid["_bin"].apply(lambda interval: float(interval.mid))
        valid["_bin_mid"] = pd.to_numeric(valid["_bin_mid"], errors="coerce")

        grouped = (
            valid
            .groupby(["skill_group", "_bin"], observed=True, as_index=False)
            .agg(
                bin_mid=("_bin_mid", "mean"),
                travel_probability=("did_travel_int", "mean"),
                n=("did_travel_int", "size"),
                n_travel=("did_travel_int", "sum"),
            )
        )
        grouped["feature"] = feature
        grouped["feature_label"] = labels[feature]
        prob_rows.append(grouped)

    probability_table = pd.concat(prob_rows, ignore_index=True) if prob_rows else pd.DataFrame()
    coefficient_table = _simple_logit_slope_table(df, features, group_col="skill_group")
    coefficient_table["feature_label"] = coefficient_table["feature"].map(labels)

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()
    colors = _make_agent_colors(range(max(1, len(skill_values))))
    skill_colors = {
        group: colors[i % len(colors)]
        for i, group in enumerate(sorted(df["skill_group"].dropna().unique()))
    }

    for ax, feature in zip(axes_flat, features):
        dff = probability_table[probability_table["feature"] == feature]
        for skill_group, dfg in dff.groupby("skill_group"):
            dfg = dfg.sort_values("bin_mid")
            ax.plot(
                dfg["bin_mid"],
                dfg["travel_probability"],
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=skill_colors.get(skill_group),
                alpha=0.9,
                label=skill_group,
            )
            ax.scatter(
                dfg["bin_mid"],
                dfg["travel_probability"],
                s=np.clip(14 + 2.2 * dfg["n"], 24, 130),
                color=skill_colors.get(skill_group),
                alpha=0.65,
                edgecolor="white",
                linewidth=0.6,
            )
        ax.set_title(labels[feature])
        ax.set_xlabel(labels[feature])
        ax.set_ylabel("Actual Pr(travel)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(features):]:
        ax.set_visible(False)

    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(legend_labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            ncol=min(4, len(by_label)),
            frameon=True,
        )
    fig.suptitle(
        "Actual Travel Probability by Variable and Skill Level",
        fontsize=16,
        fontweight="bold",
    )

    return fig, coefficient_table, probability_table, df


def region_residence_table(log):
    """Return per-agent time shares and average stay lengths by region."""
    import numpy as np
    import pandas as pd

    aids = _numeric_agent_ids(log)
    waterline = _infer_waterline(log)
    rows = []

    for aid in aids:
        seq = _region_sequence(log, aid, waterline=waterline)
        n = max(1, len(seq))
        time_top = int(sum(region == "top" for region in seq))
        time_bottom = int(sum(region == "bottom" for region in seq))

        spells = {"top": [], "bottom": []}
        if seq:
            current_region = seq[0]
            current_len = 1
            for region in seq[1:]:
                if region == current_region:
                    current_len += 1
                else:
                    spells[current_region].append(current_len)
                    current_region = region
                    current_len = 1
            spells[current_region].append(current_len)

        rows.append({
            "agent": int(aid),
            "time_top": time_top,
            "time_bottom": time_bottom,
            "share_top": time_top / n,
            "share_bottom": time_bottom / n,
            "avg_stay_top": float(np.mean(spells["top"])) if spells["top"] else 0.0,
            "avg_stay_bottom": float(np.mean(spells["bottom"])) if spells["bottom"] else 0.0,
            "n_stays_top": int(len(spells["top"])),
            "n_stays_bottom": int(len(spells["bottom"])),
        })

    return pd.DataFrame(rows)


def plot_region_residence_summary(obj, mode="auto", title=None, figsize=(11, 4.8)):
    """
    Plot top/bottom residence time and average stay lengths.

    Parameters
    ----------
    obj : dense log or run dict
        For a single dense log, pass ``log`` and use mode="single".
        For averages across all dense logs in a run, pass ``runs[2]`` and use
        mode="average".
    mode : {"auto", "single", "average"}
        ``auto`` uses single mode for a raw dense log and average mode for a
        run/dense_logs object. ``single`` treats obj as one dense log unless obj
        is a run with ``dense_log``. ``average`` averages per-agent statistics
        across all dense logs in the run.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if isinstance(obj, dict) and "states" in obj:
        logs = {0: obj}
    else:
        logs = _extract_logs_from_run(obj)

    if not logs:
        raise ValueError("No dense logs found.")

    if mode == "auto":
        mode = "single" if isinstance(obj, dict) and "states" in obj else "average"

    if mode == "single":
        first_key = next(iter(logs))
        tables = [region_residence_table(logs[first_key]).assign(log_key=first_key)]
    elif mode == "average":
        tables = [
            region_residence_table(log).assign(log_key=log_key)
            for log_key, log in logs.items()
        ]
    else:
        raise ValueError("mode must be 'auto', 'single', or 'average'.")

    raw_df = pd.concat(tables, ignore_index=True)

    if mode == "average":
        df = (
            raw_df
            .groupby("agent", as_index=False)
            .agg(
                time_top=("time_top", "mean"),
                time_bottom=("time_bottom", "mean"),
                share_top=("share_top", "mean"),
                share_bottom=("share_bottom", "mean"),
                avg_stay_top=("avg_stay_top", "mean"),
                avg_stay_bottom=("avg_stay_bottom", "mean"),
                n_stays_top=("n_stays_top", "mean"),
                n_stays_bottom=("n_stays_bottom", "mean"),
            )
        )
    else:
        df = raw_df.drop(columns=["log_key"])

    top_total = float(df["time_top"].sum())
    bottom_total = float(df["time_bottom"].sum())
    agent_colors = _make_agent_colors(df["agent"].astype(int).tolist())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [0.85, 1.6]},
        constrained_layout=True,
    )
    ax_pie, ax_bar = axes

    ax_pie.pie(
        [top_total, bottom_total],
        labels=["top", "bottom"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#1f77b4", "#d62728"],
        wedgeprops=dict(edgecolor="white", linewidth=1),
        textprops=dict(fontsize=10),
    )
    ax_pie.set_title("Average Time Spent by Region")

    x = np.arange(len(df))
    width = 0.38
    ordered = df.sort_values("agent").reset_index(drop=True)
    ax_bar.bar(
        x - width / 2,
        ordered["avg_stay_top"],
        width=width,
        color="#1f77b4",
        alpha=0.82,
        label="top",
    )
    ax_bar.bar(
        x + width / 2,
        ordered["avg_stay_bottom"],
        width=width,
        color="#d62728",
        alpha=0.82,
        label="bottom",
    )
    for xpos, aid in zip(x, ordered["agent"]):
        ax_bar.scatter(
            [xpos],
            [0],
            marker="s",
            s=45,
            color=agent_colors.get(int(aid), "0.4"),
            clip_on=False,
            zorder=4,
        )

    ax_bar.set_title("Average Stay Length by Agent")
    ax_bar.set_xlabel("agent")
    ax_bar.set_ylabel("timesteps per stay")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(int(a)) for a in ordered["agent"]])
    ax_bar.grid(True, axis="y", alpha=0.25)
    ax_bar.legend(frameon=True)

    if title is None:
        if mode == "single":
            title = f"Single Dense Log {raw_df['log_key'].iloc[0]}"
        else:
            title = f"Average Across {raw_df['log_key'].nunique()} Dense Logs"
    fig.suptitle(f"Region Residence Summary: {title}", fontsize=14, fontweight="bold")

    return fig, df, raw_df


def plot_regional_composition_tax_phase(
    feedback_df,
    composition_metric="share_high_skill",
    next_policy_metric="next_top_marginal_rate",
    figsize=(8, 6),
):
    """
    Phase plot with x_{r,k} on the x-axis and tax policy_{r,k+1} on the y-axis.
    """
    if composition_metric not in feedback_df.columns:
        raise ValueError(f"Unknown composition_metric {composition_metric!r}.")
    if next_policy_metric not in feedback_df.columns:
        raise ValueError(f"Unknown next_policy_metric {next_policy_metric!r}.")

    labels = {
        "share_high_skill": "share high-skill agents in region r during k",
        "population_share": "population share in region r during k",
        "production": "tax base / production in region r during k",
        "equality": "equality in region r during k",
        "next_top_marginal_rate": "top marginal tax rate in k+1",
        "next_avg_marginal_rate": "average marginal tax rate in k+1",
        "next_progressivity": "tax progressivity in k+1",
    }
    colors = {"top": "#1f77b4", "bottom": "#ff7f0e"}

    df = feedback_df.dropna(subset=[composition_metric, next_policy_metric]).copy()
    if df.empty:
        raise ValueError("No non-missing lagged composition/policy pairs are available.")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    max_period = max(1, int(df["tax_period"].max()))

    for region in ["top", "bottom"]:
        dfr = df[df["planner_region"] == region].sort_values("tax_period")
        if dfr.empty:
            continue

        x = dfr[composition_metric].to_numpy(dtype=float)
        y = dfr[next_policy_metric].to_numpy(dtype=float)
        c = dfr["tax_period"].to_numpy(dtype=float)

        ax.plot(x, y, color=colors[region], alpha=0.35, linewidth=1.5)
        sc = ax.scatter(
            x,
            y,
            c=c,
            cmap="viridis",
            vmin=1,
            vmax=max_period,
            s=65,
            edgecolor=colors[region],
            linewidth=1.4,
            label=region,
            zorder=3,
        )

        for i in range(len(x) - 1):
            ax.annotate(
                "",
                xy=(x[i + 1], y[i + 1]),
                xytext=(x[i], y[i]),
                arrowprops=dict(arrowstyle="->", color=colors[region], lw=1.4, alpha=0.75),
            )

    ax.set_xlabel(labels.get(composition_metric, composition_metric))
    ax.set_ylabel(labels.get(next_policy_metric, next_policy_metric))
    ax.set_title("Co-Evolution of Regional Composition and Tax Policy")
    ax.grid(True, alpha=0.3)
    ax.legend(title="region")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("tax period k")
    return fig
