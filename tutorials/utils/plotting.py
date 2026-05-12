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
        "#17becf",  # cyan, replacing default tab10 red slot
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#d62728",  # red, replacing the default gray slot
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
        "metrics": metrics,
        "dense_logs": dense_logs,
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
            "gini_final_coin",
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

    x = np.arange(len(metrics))
    group_width = 0.62
    bar_width = group_width / max(1, len(run_names))
    offsets = (np.arange(len(run_names)) - (len(run_names) - 1) / 2) * bar_width

    for run_i, name in enumerate(run_names):
        vals = summary_df.loc[name, metrics].to_numpy(dtype=float)
        yerr = np.array(
            [err_lookup[name].get(metric, np.nan) for metric in metrics],
            dtype=float,
        )
        yerr = np.where(np.isfinite(yerr), yerr, 0.0)

        ax.bar(
            x + offsets[run_i],
            vals,
            color=colors[name],
            width=bar_width * 0.82,
            alpha=0.9,
            yerr=None if errorbar is None else yerr,
            capsize=3 if errorbar is not None else 0,
            ecolor="black",
            linewidth=0,
            label=short_labels[name],
        )

    ax.set_title("Summary Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels([metric.replace("_", " ").title() for metric in metrics], rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

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

        axes[0].plot(
            x,
            smooth_series(df["market_size"], smooth_window),
            color=color,
            linewidth=2.2,
            label=label,
        )
        axes[1].plot(
            x,
            display_price_series(df["mean_price"]),
            color=color,
            linewidth=2.2,
            label=label,
        )
        axes[2].plot(
            x,
            smooth_series(df["trade_count"], smooth_window),
            color=color,
            linewidth=2.2,
            label=label,
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
            buyer_region = None
            seller_region = None

            if buyer is not None and str(buyer) in state_t:
                buyer_region = _location_region_from_state(
                    state_t[str(buyer)], waterline=waterline
                )
            if seller is not None and str(seller) in state_t:
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
            "price", "buyer_cost", "tariff"
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

def extract_planner_redistribution_by_period(log, period=100, rate_disc=0.05):
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
                "tax_period": tax_period,
                "timestep": tax_t,
                "planner_region": region,
                "income": income_total,
                "income_tax_collected": income_tax_total,
                "tariff_revenue": tariff_by_region[region],
                "redistributed": redistributed,
                "income_tax_funded_redistribution": income_tax_total,
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
    figsize=(14, 10),
):
    """
    Plot one trade-enabled run, either one dense log or the average across logs.

    Top-left: units of Wood and Stone traded within-region versus cross-region
    over the full rollout. Top-right: average untaxed transaction price by
    commodity and route type. Bottom: redistribution by planner and tax period,
    split into income-tax-funded and other redistribution.
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

    if trade_raw.empty:
        trade_units = pd.DataFrame(columns=["tax_period", "commodity", "route_group", "units", "units_std"])
        price_summary = pd.DataFrame(columns=["commodity", "route_group", "avg_price"])
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
            .agg(avg_price=("price", "mean"))
        )
        price_summary = (
            per_rollout_prices
            .groupby(["commodity", "route_group"], as_index=False)
            .agg(avg_price=("avg_price", "mean"))
        )
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
            .agg(avg_price=("price", "mean"))
        )

    if redist_raw.empty:
        redist = pd.DataFrame(columns=[
            "tax_period", "planner_region", "income_tax_funded_redistribution",
            "non_income_tax_redistribution", "redistributed"
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
                income_tax_funded_redistribution=("income_tax_funded_redistribution", "mean"),
                non_income_tax_redistribution=("non_income_tax_redistribution", "mean"),
            )
        )
    else:
        redist = redist_raw.copy()

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.25, 0.9, 1.35],
    )
    ax_trade = fig.add_subplot(gs[0, :])
    ax_pies = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax_redist = fig.add_subplot(gs[2, :])

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
            yerr=within_err if mode == "average" else None,
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
            yerr=cross_err if mode == "average" else None,
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
            price_label = "within" if route_group == "within region" else "cross"
            price_lines.append(f"{price_label} avg seller price: {'n/a' if pd.isna(price) else f'{price:.2f}'}")

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

    tax_periods = sorted(redist["tax_period"].dropna().unique()) if not redist.empty else []
    planners = ["top", "bottom"]
    bar_width = 0.36
    period_x = np.arange(len(tax_periods))
    planner_offsets = {"top": -bar_width / 2, "bottom": bar_width / 2}
    planner_colors = {"top": "#ff7f0e", "bottom": "#9467bd"}

    for planner in planners:
        income_vals = []
        other_vals = []
        tariff_vals = []
        for tax_period in tax_periods:
            match = redist[
                (redist["tax_period"] == tax_period)
                & (redist["planner_region"] == planner)
            ]
            income_vals.append(
                float(match["income_tax_funded_redistribution"].sum())
                if len(match) else 0.0
            )
            other_vals.append(
                float(match["non_income_tax_redistribution"].sum())
                if len(match) else 0.0
            )
            tariff_vals.append(
                float(match["tariff_revenue"].sum())
                if len(match) and "tariff_revenue" in match else 0.0
            )
        pos = period_x + planner_offsets[planner]
        ax_redist.bar(
            pos,
            income_vals,
            width=bar_width,
            label=f"{planner}: from income tax",
            color=planner_colors[planner],
            edgecolor="white",
            linewidth=0.8,
        )
        ax_redist.bar(
            pos,
            other_vals,
            bottom=income_vals,
            width=bar_width,
            label=f"{planner}: import/travel tax",
            color=planner_colors[planner],
            alpha=0.35,
            edgecolor="white",
            linewidth=0.8,
        )
        totals = np.asarray(income_vals, dtype=float) + np.asarray(other_vals, dtype=float)
        for xpos, total, tariff in zip(pos, totals, tariff_vals):
            if tariff <= 0:
                continue
            ax_redist.text(
                xpos,
                total,
                f"{tariff:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color="0.2",
            )

    ax_redist.set_title("Redistribution by Planner and Tax Period")
    ax_redist.set_xlabel("Tax period")
    ax_redist.set_ylabel("Coin redistributed")
    ax_redist.set_xticks(period_x)
    ax_redist.set_xticklabels([str(int(k)) for k in tax_periods])
    ax_redist.legend(ncol=2, frameon=True)
    ax_redist.grid(True, axis="y", alpha=0.3)

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


def breakdown_all_agents_from_result_folder(result_dir, episode_key=0, remap_key="build_payment", n_cols=4):
    dense_log, dense_logs = get_dense_log_from_result_folder(result_dir, episode_key=episode_key)
    breakdown = breakdown_all_agents(dense_log, remap_key=remap_key, n_cols=n_cols)
    return breakdown, dense_log, dense_logs




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
    df_income, counts, labels = _income_bracket_counts(log, brackets, period=period)
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
            pivot.values,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
        )

        ax.set_title(f"{region} planner: agents per income bracket")
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
