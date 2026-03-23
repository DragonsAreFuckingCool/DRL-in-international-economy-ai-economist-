# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

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
      - uses up to n_cols columns (8 agents -> 2 rows of 4)
      - labels legend so highest/lowest skill agents are explicit
      - does NOT overwrite breakdown()

    Returns:
        (fig0, fig1, fig2), incomes, endows, c_trades, all_builds
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt

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
    for i, aid in enumerate(aidx):
        if i == 0:
            rank_labels.append(f"Agent {aid} (Lowest Skill)")
        elif i == len(aidx) - 1:
            rank_labels.append(f"Agent {aid} (Highest Skill)")
        else:
            rank_labels.append(f"Agent {aid}")

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

    # Labor
    ax = axes[3]
    for i in range(n):
        ax.plot(
            [x[str(aidx[i])]["endogenous"]["Labor"] for x in log["states"]],
            label=rank_labels[i],
            color=cmap(i),
        )
    ax.set_title("Labor")
    ax.grid(True)

    # Utility (if remap_key exists and you want final curves from state)
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
            # fallback: plot coin again if utility not stored in state
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

    # Put legend on rightmost time-series axis
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    # --- Movement tracks / trade timelines for ALL agents ---
    tmp_map = np.array(log["world"][0]["Stone"])
    n_cols = min(n_cols, n)
    n_agent_rows = int(math.ceil(n / n_cols))

    total_rows = n_agent_rows + (n_agent_rows if trading_active else 0)
    fig2, axes = plt.subplots(
        total_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * total_rows),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    # Turn off unused axes
    for rr in range(total_rows):
        for cc in range(n_cols):
            idx = rr * n_cols + cc
            if rr < n_agent_rows:
                if idx >= n:
                    axes[rr, cc].axis("off")
            else:
                trade_idx = (rr - n_agent_rows) * n_cols + cc
                if trade_idx >= n:
                    axes[rr, cc].axis("off")

    # Trajectories
    for i in range(n):
        rr = i // n_cols
        cc = i % n_cols
        ax = axes[rr, cc]

        rows = np.array([x[str(aidx[i])]["loc"][0] for x in log["states"]]) * -1
        cols = np.array([x[str(aidx[i])]["loc"][1] for x in log["states"]])

        ax.plot(cols[::20], rows[::20], color=cmap(i))
        ax.plot(cols[0], rows[0], "r*", markersize=12)
        ax.plot(cols[-1], rows[-1], "g*", markersize=12)
        ax.set_title(rank_labels[i])
        ax.set_xlim([-1, 1 + tmp_map.shape[1]])
        ax.set_ylim([-(1 + tmp_map.shape[0]), 1])
        ax.grid(True)

    # Trade timelines
    if trading_active:
        for i in range(n):
            rr = n_agent_rows + (i // n_cols)
            cc = i % n_cols
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
            ax.set_title(rank_labels[i])

    return (fig0, fig1, fig2), incomes, endows, c_trades, all_builds