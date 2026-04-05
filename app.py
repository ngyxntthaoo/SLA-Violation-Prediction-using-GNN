"""
GNN SLA Violation Prediction — Streamlit Demo
Run:  streamlit run GNN/app.py
"""
from __future__ import annotations
import os, random, time, warnings
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SLA Violation Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ───────────────────────────────────────────────────────────────────
GREEN, RED, BLUE = "#2ecc71", "#e74c3c", "#3498db"

# ═════════════════════════════════════════════════════════════════════════════
# 1.  CORE FUNCTIONS (mirrored from notebook)
# ═════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_synthetic_event_log(n_cases: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    base_date = pd.Timestamp("2024-01-01", tz="UTC")
    MAIN_FLOW = ["Submit Application", "Verify Documents", "Credit Check", "Assess Risk"]
    REWORK = ["Request Additional Info", "Provide Additional Info"]
    APPROVE = ["Approve", "Finalize"]
    REJECT  = ["Reject", "Notify Applicant"]
    records = []
    for i in range(n_cases):
        cid = f"C_{i:05d}"
        t = base_date + timedelta(hours=float(rng.uniform(0, 24 * 180)))
        is_complex = rng.random() < 0.35
        n_rework = int(rng.choice([0, 1, 2, 3], p=[0.45, 0.25, 0.18, 0.12])) if is_complex else 0
        will_reject = rng.random() < 0.15
        for act in MAIN_FLOW:
            records.append((cid, act, t, f"Agent_{rng.randint(1, 20):02d}"))
            t += timedelta(hours=float(rng.exponential(36 if is_complex else 10)))
        for _ in range(n_rework):
            for act in REWORK:
                records.append((cid, act, t, f"Agent_{rng.randint(1, 20):02d}"))
                t += timedelta(hours=float(rng.exponential(72 if is_complex else 36)))
        for act in APPROVE if not will_reject else REJECT:
            records.append((cid, act, t, f"Agent_{rng.randint(1, 20):02d}"))
            t += timedelta(hours=float(rng.exponential(8)))
    df = pd.DataFrame(records, columns=["case_id", "activity", "timestamp", "resource"])
    return df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)


def generate_sla_labels(df: pd.DataFrame, threshold_days: int = 30) -> pd.DataFrame:
    ct = df.groupby("case_id")["timestamp"].agg(["min", "max"])
    ct["duration_days"] = (ct["max"] - ct["min"]).dt.total_seconds() / 86400
    ct["label"] = (ct["duration_days"] > threshold_days).astype(int)
    return ct[["duration_days", "label"]].reset_index()


def build_activity_index(df: pd.DataFrame) -> dict[str, int]:
    return {a: i for i, a in enumerate(sorted(df["activity"].unique()))}


def case_to_graph(case_df, activity_idx, label, case_id, prefix_len=None, _sorted=False):
    if not _sorted:
        case_df = case_df.sort_values("timestamp").reset_index(drop=True)
    if prefix_len is not None:
        case_df = case_df.iloc[:prefix_len]
    n, n_act = len(case_df), len(activity_idx)
    acts, ts = case_df["activity"].values, case_df["timestamp"].values
    t0 = ts[0]
    features = np.zeros((n, n_act + 2), dtype=np.float32)
    for i in range(n):
        features[i, activity_idx.get(acts[i], 0)] = 1.0
        features[i, n_act]     = (ts[i] - t0) / np.timedelta64(1, "D")
        features[i, n_act + 1] = ((ts[i] - ts[i-1]) / np.timedelta64(1, "D")) if i > 0 else 0.0
    ei = torch.tensor([list(range(n-1)), list(range(1, n))], dtype=torch.long) if n > 1 \
         else torch.zeros((2, 0), dtype=torch.long)
    g = Data(x=torch.from_numpy(features), edge_index=ei,
              y=torch.tensor([label], dtype=torch.float))
    g.case_id    = case_id
    g.activities = case_df["activity"].tolist()
    return g


class SLA_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, dropout=0.3):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden)
        self.conv2   = GCNConv(hidden, hidden // 2)
        self.lin     = torch.nn.Linear(hidden // 2, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze(-1)

    def get_graph_embedding(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  CACHED DATA + MODEL PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Generating event log…")
def get_event_log(n_cases: int, sla_days: int, seed: int):
    df = generate_synthetic_event_log(n_cases=n_cases, seed=seed)
    labels_df = generate_sla_labels(df, threshold_days=sla_days)
    return df, labels_df


@st.cache_resource(show_spinner="Building graphs & training GCN…")
def build_and_train(n_cases: int, sla_days: int, seed: int, epochs: int,
                    hidden: int, lr: float, batch_size: int):
    seed_everything(seed)
    df, labels_df = get_event_log(n_cases, sla_days, seed)
    activity_idx  = build_activity_index(df)
    N_ACT         = len(activity_idx)
    IN_CH         = N_ACT + 2
    PREFIX_PCTS   = [0.2, 0.4, 0.6, 0.8, 1.0]
    DEVICE        = torch.device("cpu")

    # ── splits ──
    case_ids, case_labels = labels_df["case_id"].values, labels_df["label"].values
    tr_cids, te_cids = train_test_split(case_ids, test_size=0.20, random_state=seed,
                                        stratify=case_labels)
    tr_lbl = labels_df[labels_df["case_id"].isin(tr_cids)]["label"].values
    tr_cids, va_cids = train_test_split(tr_cids, test_size=0.10/0.80, random_state=seed,
                                        stratify=tr_lbl)
    split_map = {c: "train" for c in tr_cids}
    split_map.update({c: "val" for c in va_cids})
    split_map.update({c: "test" for c in te_cids})

    # ── build prefix graphs ──
    case_groups = {cid: cdf.sort_values("timestamp").reset_index(drop=True)
                   for cid, cdf in df.groupby("case_id")}
    label_map   = labels_df.set_index("case_id")["label"].to_dict()

    train_data, val_data, test_data, full_graphs = [], [], [], []
    for cid, cdf in case_groups.items():
        if cid not in split_map or cid not in label_map:
            continue
        lbl  = int(label_map[cid])
        sp   = split_map[cid]
        full_graphs.append(case_to_graph(cdf, activity_idx, lbl, cid, _sorted=True))
        bucket = train_data if sp == "train" else (val_data if sp == "val" else test_data)
        for pct in PREFIX_PCTS:
            plen = max(2, int(len(cdf) * pct))
            if plen == len(cdf) and pct < 1.0:
                continue
            bucket.append(case_to_graph(cdf, activity_idx, lbl, cid, prefix_len=plen, _sorted=True))

    # ── z-normalise temporal features (train stats only) ──
    ts_ = np.concatenate([g.x[:, -2].numpy() for g in train_data])
    tp_ = np.concatenate([g.x[:, -1].numpy() for g in train_data])
    NORM = dict(ts_m=ts_.mean(), ts_s=ts_.std()+1e-8, tp_m=tp_.mean(), tp_s=tp_.std()+1e-8)

    def normalise(graphs):
        for g in graphs:
            g.x[:, -2] = (g.x[:, -2] - NORM["ts_m"]) / NORM["ts_s"]
            g.x[:, -1] = (g.x[:, -1] - NORM["tp_m"]) / NORM["tp_s"]
    for grp in [train_data, val_data, test_data, full_graphs]:
        normalise(grp)

    # ── model ──
    model = SLA_GCN(in_channels=IN_CH, hidden=hidden).to(DEVICE)
    pos   = sum(int(g.y.item()) for g in train_data)
    neg   = len(train_data) - pos
    pos_w = torch.tensor([np.sqrt(neg / (pos + 1e-8))], dtype=torch.float)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loader_tr = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(val_data,   batch_size=batch_size)

    best_f1, patience_cnt, best_state = 0.0, 0, None
    history = {"train_loss": [], "val_f1": []}
    PATIENCE = 20
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in loader_tr:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out  = model(batch.x, batch.edge_index, batch.batch)
            loss = F.binary_cross_entropy_with_logits(out, batch.y, pos_weight=pos_w.to(DEVICE))
            loss.backward(); opt.step()
            total_loss += loss.item()
        history["train_loss"].append(total_loss / len(loader_tr))

        # ── val F1 ──
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader_va:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index, batch.batch)
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                trues.extend(batch.y.cpu().numpy())
        vf1 = f1_score(trues, [p > 0.5 for p in preds], zero_division=0)
        history["val_f1"].append(vf1)
        sched.step(vf1)
        if vf1 > best_f1:
            best_f1, patience_cnt = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── test evaluation (full graphs only) ──
    te_full = [g for g in test_data if
               any(g2.case_id == g.case_id and len(g2.activities) == len(g.activities)
                   for g2 in full_graphs)]
    te_full = [g for g in full_graphs if split_map.get(g.case_id) == "test"]
    loader_te = DataLoader(te_full, batch_size=batch_size)
    model.eval()
    y_prob, y_true = [], []
    with torch.no_grad():
        for batch in loader_te:
            batch = batch.to(DEVICE)
            y_prob.extend(torch.sigmoid(model(batch.x, batch.edge_index, batch.batch)).numpy())
            y_true.extend(batch.y.numpy())
    y_pred = [p > 0.5 for p in y_prob]

    metrics = dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, zero_division=0),
        recall    = recall_score(y_true, y_pred, zero_division=0),
        f1        = f1_score(y_true, y_pred, zero_division=0),
        roc_auc   = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        pr_auc    = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
    )

    # ── prefix-stratified evaluation ──
    prefix_rows = []
    for pct in PREFIX_PCTS:
        grp = [g for g in test_data if
               abs(len(g.activities) / max(
                   next((len(fg.activities) for fg in full_graphs if fg.case_id == g.case_id), 1), 1)
                   - pct) < 0.15]
        if not grp:
            continue
        ldr  = DataLoader(grp, batch_size=batch_size)
        yp, yt = [], []
        with torch.no_grad():
            for b in ldr:
                b = b.to(DEVICE)
                yp.extend(torch.sigmoid(model(b.x, b.edge_index, b.batch)).numpy())
                yt.extend(b.y.numpy())
        prefix_rows.append(dict(
            completion=f"{int(pct*100)}%",
            f1        =f1_score(yt, [p>0.5 for p in yp], zero_division=0),
            precision =precision_score(yt, [p>0.5 for p in yp], zero_division=0),
            recall    =recall_score(yt, [p>0.5 for p in yp], zero_division=0),
            roc_auc   =roc_auc_score(yt, yp) if len(set(yt))>1 else float("nan"),
        ))

    return dict(
        model        = model,
        df           = df,
        labels_df    = labels_df,
        activity_idx = activity_idx,
        full_graphs  = full_graphs,
        train_data   = train_data,
        val_data     = val_data,
        test_data    = test_data,
        split_map    = split_map,
        case_groups  = case_groups,
        label_map    = label_map,
        NORM         = NORM,
        metrics      = metrics,
        y_prob       = y_prob,
        y_true       = y_true,
        history      = history,
        prefix_rows  = prefix_rows,
        IN_CH        = IN_CH,
        DEVICE       = DEVICE,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3.  VISUALISATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _break_cycles(G):
    visited, stack, back = set(), set(), []
    def dfs(u):
        visited.add(u); stack.add(u)
        for v in list(G.successors(u)):
            if v not in visited:
                dfs(v)
            elif v in stack:
                back.append((u, v))
        stack.discard(u)
    for n in G.nodes():
        if n not in visited:
            dfs(n)
    return back


def _layered_layout(G, x_gap=4.0, y_gap=2.0):
    back = _break_cycles(G)
    H = G.copy()
    H.remove_edges_from(back)
    try:
        order = list(nx.topological_sort(H))
    except nx.NetworkXUnfeasible:
        return nx.spring_layout(G, seed=42, k=2.5)
    layer = {}
    for n in order:
        preds = list(H.predecessors(n))
        layer[n] = max((layer[p] for p in preds), default=-1) + 1
    by_layer = defaultdict(list)
    for n, l in layer.items():
        by_layer[l].append(n)
    pos = {}
    for l, nodes in by_layer.items():
        for j, n in enumerate(nodes):
            pos[n] = (l * x_gap, -(j - (len(nodes)-1)/2) * y_gap)
    return pos


def _min_dist_push(pos, min_d=1.4, iters=60):
    pos = {k: list(v) for k, v in pos.items()}
    nodes = list(pos.keys())
    for _ in range(iters):
        moved = False
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                a, b = nodes[i], nodes[j]
                dx = pos[b][0] - pos[a][0]
                dy = pos[b][1] - pos[a][1]
                d  = max((dx**2 + dy**2)**0.5, 1e-6)
                if d < min_d:
                    push = (min_d - d) / 2
                    nx_, ny_ = (dx/d)*push, (dy/d)*push
                    pos[b][0] += nx_; pos[b][1] += ny_
                    pos[a][0] -= nx_; pos[a][1] -= ny_
                    moved = True
        if not moved:
            break
    return {k: tuple(v) for k, v in pos.items()}



def _sla_stats(df, labels_df):
    """Return per-activity (total, viol, sla_rate) and per-edge count dicts."""
    violated_cids = set(labels_df[labels_df["label"] == 1]["case_id"])
    act_total, act_viol, transitions = Counter(), Counter(), Counter()
    for cid, cdf in df.groupby("case_id"):
        acts = cdf.sort_values("timestamp")["activity"].tolist()
        for a in acts:
            act_total[a] += 1
            if cid in violated_cids:
                act_viol[a] += 1
        for a, b in zip(acts, acts[1:]):
            transitions[(a, b)] += 1
    sla_rate = {a: act_viol[a] / act_total[a] for a in act_total}
    return act_total, act_viol, sla_rate, transitions


def plotly_bpmn(df, labels_df) -> go.Figure:
    """BPMN-style DFG rendered with Plotly — same style as SLA Impact Graph."""
    act_total, act_viol, sla_rate, transitions = _sla_stats(df, labels_df)

    G = nx.DiGraph()
    for a in act_total:
        G.add_node(a, sla=sla_rate[a], total=act_total[a])
    for (a, b), cnt in transitions.items():
        G.add_edge(a, b, weight=cnt)

    pos     = _layered_layout(G)
    pos     = _min_dist_push(pos, min_d=1.6)
    back    = set(map(tuple, _break_cycles(G)))
    cmap    = cm.get_cmap("RdYlGn_r")
    max_cnt = max(act_total.values())
    max_w   = max(transitions.values())

    # ── Edge traces — normal (grey) and back-edges (purple) ──
    edge_traces = []
    annotations = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        w        = 0.5 + 5.0 * d["weight"] / max_w
        is_back  = (u, v) in back
        color    = "rgba(155,89,182,0.7)" if is_back else "rgba(149,165,166,0.55)"

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=w, color=color, dash="dot" if is_back else "solid"),
            hoverinfo="none", showlegend=False,
        ))
        # Arrow annotation
        annotations.append(dict(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.2,
            arrowwidth=max(0.5, 4 * d["weight"] / max_w),
            arrowcolor=color,
        ))
        # Frequency label at edge midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        annotations.append(dict(
            x=mx, y=my, xref="x", yref="y",
            text=f"<b>{d['weight']}</b>",
            showarrow=False,
            font=dict(size=9, color="#555"),
            bgcolor="rgba(255,255,255,0.75)",
            borderpad=2,
        ))

    # ── Node trace ──
    node_x, node_y, node_text, node_hover, node_color, node_size = [], [], [], [], [], []
    for n in G.nodes():
        x, y  = pos[n]
        rate  = sla_rate.get(n, 0)
        total = act_total[n]
        viol  = act_viol.get(n, 0)
        node_x.append(x); node_y.append(y)
        node_text.append(f"<b>{n}</b>")
        node_hover.append(
            f"<b>{n}</b><br>"
            f"Appearances: {total:,}<br>"
            f"In violated cases: {viol:,}<br>"
            f"SLA risk: {rate:.1%}"
        )
        node_color.append(rate)
        node_size.append(20 + 40 * total / max_cnt)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle center",
        textfont=dict(size=11, color="white"),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="RdYlGn",
            reversescale=True,
            colorbar=dict(title="SLA risk", thickness=14, len=0.6),
            cmin=0, cmax=1,
            line=dict(width=2, color="#2c3e50"),
            symbol="square",
        ),
        showlegend=False,
    )

    # ── Legend traces (dummy scatter for back-edge indicator) ──
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode="lines",
                   line=dict(color="rgba(149,165,166,0.8)", width=2),
                   name="Forward edge"),
        go.Scatter(x=[None], y=[None], mode="lines",
                   line=dict(color="rgba(155,89,182,0.8)", width=2, dash="dot"),
                   name="Back-edge (cycle)"),
    ]

    fig = go.Figure(data=edge_traces + [node_trace] + legend_traces)
    fig.update_layout(
        title=dict(
            text="Directly Follows Graph (BPMN-style) — hover nodes for details",
            font=dict(size=14),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#ccc", borderwidth=1),
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=annotations,
        plot_bgcolor="#f9f9fb",
        margin=dict(l=20, r=20, t=50, b=20),
        height=580,
    )
    return fig


def plotly_sla_process_graph(df, labels_df) -> go.Figure:
    """Aggregated process graph rendered with Plotly (zoom/pan/hover)."""
    act_total, act_viol, sla_rate, transitions = _sla_stats(df, labels_df)

    G = nx.DiGraph()
    for a in act_total:
        G.add_node(a, sla=sla_rate[a], total=act_total[a])
    for (a, b), cnt in transitions.items():
        G.add_edge(a, b, weight=cnt)

    pos     = _layered_layout(G)
    pos     = _min_dist_push(pos, min_d=1.6)
    cmap    = cm.get_cmap("RdYlGn_r")
    max_cnt = max(act_total.values())
    max_w   = max(transitions.values())

    # ── Edge traces (one per edge for variable width) ──
    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        w = 0.5 + 5.0 * d["weight"] / max_w
        # arrow midpoint annotation done via shape
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=w, color="rgba(149,165,166,0.6)"),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Node trace ──
    node_x, node_y, node_text, node_hover, node_color, node_size = [], [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        rate  = sla_rate.get(n, 0)
        total = act_total[n]
        viol  = act_viol.get(n, 0)
        node_x.append(x); node_y.append(y)
        node_text.append(f"<b>{n}</b>")
        node_hover.append(
            f"<b>{n}</b><br>"
            f"Appearances: {total:,}<br>"
            f"In violated cases: {viol:,}<br>"
            f"SLA risk: {rate:.1%}"
        )
        node_color.append(rate)
        node_size.append(20 + 40 * total / max_cnt)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle center",
        textfont=dict(size=11, color="white"),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="RdYlGn",
            reversescale=True,
            colorbar=dict(title="SLA risk", thickness=14, len=0.6),
            cmin=0, cmax=1,
            line=dict(width=2, color="#2c3e50"),
            symbol="square",
        ),
        showlegend=False,
    )

    # ── Edge label annotations (frequency) ──
    annotations = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        annotations.append(dict(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.2, arrowwidth=max(0.5, 4*d["weight"]/max_w),
            arrowcolor="rgba(149,165,166,0.55)",
        ))
        mx, my = (x0+x1)/2, (y0+y1)/2
        annotations.append(dict(
            x=mx, y=my, xref="x", yref="y",
            text=f"<b>{d['weight']}</b>",
            showarrow=False,
            font=dict(size=9, color="#555"),
            bgcolor="rgba(255,255,255,0.7)",
            borderpad=2,
        ))

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=dict(text="Process SLA Impact Graph — hover nodes for details",
                   font=dict(size=14)),
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=annotations,
        plot_bgcolor="#f9f9fb",
        margin=dict(l=20, r=20, t=50, b=20),
        height=560,
    )
    return fig


def predict_prefix(case_df, model, activity_idx, n_events, norm):
    model.eval()
    g = case_to_graph(case_df, activity_idx, 0, "demo", prefix_len=n_events, _sorted=True)
    g.x[:, -2] = (g.x[:, -2] - norm["ts_m"]) / norm["ts_s"]
    g.x[:, -1] = (g.x[:, -1] - norm["tp_m"]) / norm["tp_s"]
    with torch.no_grad():
        logit = model(g.x, g.edge_index, torch.zeros(g.num_nodes, dtype=torch.long))
        return float(torch.sigmoid(logit).item())


def gauge_chart(value: float, label="Risk Score") -> go.Figure:
    """Plotly gauge showing probability 0→1."""
    color = "#2ecc71" if value < 0.35 else ("#f39c12" if value < 0.65 else "#e74c3c")
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = value * 100,
        title = {"text": label, "font": {"size": 18}},
        number= {"suffix": "%", "font": {"size": 32}},
        gauge = {
            "axis"    : {"range": [0, 100], "tickwidth": 1},
            "bar"     : {"color": color, "thickness": 0.3},
            "bgcolor" : "white",
            "steps"   : [
                {"range": [0,  35],  "color": "#d5f5e3"},
                {"range": [35, 65],  "color": "#fdebd0"},
                {"range": [65, 100], "color": "#fadbd8"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "value": 50},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 4.  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🔮 SLA Predictor")
    st.caption("GNN-based process monitoring demo")
    st.divider()

    st.subheader("⚙️ Configuration")
    n_cases  = st.slider("Number of cases",    500, 5000, 3000, 500)
    sla_days = st.slider("SLA threshold (days)", 10, 90,  30,   5)
    epochs   = st.slider("Training epochs",    20,  200, 80,   10)
    hidden   = st.select_slider("GCN hidden dim", [64, 128, 256], 128)
    lr       = st.select_slider("Learning rate",  [1e-4, 5e-4, 1e-3], 5e-4)
    batch_sz = st.select_slider("Batch size",    [32, 64, 128], 64)
    seed     = st.number_input("Random seed", value=42, step=1)

    if st.button("🚀 Train / Reload", use_container_width=True, type="primary"):
        build_and_train.clear()
        get_event_log.clear()
        st.rerun()

    st.divider()
    st.caption("Adjust parameters and click **Train / Reload** to retrain.")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  LOAD (or train) everything
# ═════════════════════════════════════════════════════════════════════════════

state = build_and_train(n_cases, sla_days, seed, epochs, hidden, lr, batch_sz)
df, labels_df    = state["df"], state["labels_df"]
model            = state["model"]
activity_idx     = state["activity_idx"]
full_graphs      = state["full_graphs"]
case_groups      = state["case_groups"]
label_map        = state["label_map"]
split_map        = state["split_map"]
NORM             = state["NORM"]
metrics          = state["metrics"]
y_prob, y_true   = state["y_prob"], state["y_true"]
history          = state["history"]
prefix_rows      = state["prefix_rows"]

# ═════════════════════════════════════════════════════════════════════════════
# 6.  TABS
# ═════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Overview",
    "🔄 Process Mining",
    "🎯 Model Performance",
    "⚡ Early Warning Demo",
    "🔍 Bottleneck Analysis",
])


# ── TAB 1: Dataset Overview ─────────────────────────────────────────────────
with tab1:
    st.header("Dataset Overview")

    n_viol = int(labels_df["label"].sum())
    n_ok   = len(labels_df) - n_viol
    n_act  = df["activity"].nunique()
    n_ev   = len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cases",    f"{len(labels_df):,}")
    c2.metric("Violated Cases", f"{n_viol:,}",   f"{n_viol/len(labels_df):.1%}")
    c3.metric("Activity Types", f"{n_act}")
    c4.metric("Total Events",   f"{n_ev:,}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            names=["On-time", "Violated"], values=[n_ok, n_viol],
            color_discrete_sequence=[GREEN, RED],
            title="SLA Label Distribution",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_dur = px.histogram(
            labels_df, x="duration_days", color="label",
            color_discrete_map={0: GREEN, 1: RED},
            labels={"duration_days": "Duration (days)", "label": "SLA Violated"},
            title="Case Duration Distribution",
            barmode="overlay", nbins=50, opacity=0.7,
        )
        fig_dur.add_vline(x=sla_days, line_dash="dash", line_color="black",
                          annotation_text=f"SLA={sla_days}d")
        st.plotly_chart(fig_dur, use_container_width=True)

    st.subheader("Events per Case")
    epcase = df.groupby("case_id").size().reset_index(name="n_events")
    fig_ev = px.histogram(epcase, x="n_events", nbins=30, color_discrete_sequence=[BLUE],
                          labels={"n_events": "Events per case"}, title="Event Count Distribution")
    st.plotly_chart(fig_ev, use_container_width=True)

    st.subheader("Activity Frequency")
    act_freq = df["activity"].value_counts().reset_index()
    act_freq.columns = ["activity", "count"]
    fig_act = px.bar(act_freq, x="activity", y="count", color="count",
                     color_continuous_scale="Blues", title="Activity Frequency")
    st.plotly_chart(fig_act, use_container_width=True)


# ── TAB 2: Process Mining ────────────────────────────────────────────────────
with tab2:
    st.header("Process Mining Visualizations")

    st.subheader("Directly Follows Graph (BPMN-style)")
    st.caption("Node colour = SLA risk · Edge width = transition frequency · Hover for details · Purple dashed = back-edge (cycle)")
    with st.spinner("Rendering BPMN diagram…"):
        fig_bpmn = plotly_bpmn(df, labels_df)
    st.plotly_chart(fig_bpmn, use_container_width=True)

    st.divider()
    st.subheader("Aggregated SLA Impact Graph")
    st.caption("Node size ∝ frequency · Red = high SLA risk · Hover for per-activity stats")
    with st.spinner("Rendering SLA graph…"):
        fig_sla = plotly_sla_process_graph(df, labels_df)
    st.plotly_chart(fig_sla, use_container_width=True)

    st.divider()
    st.subheader("Activity SLA Risk Table")
    violated_cids = set(labels_df[labels_df["label"] == 1]["case_id"])
    rows = []
    for act in sorted(df["activity"].unique()):
        total  = sum(1 for _, cdf in df.groupby("case_id") if act in cdf["activity"].values)
        viol   = sum(1 for cid, cdf in df.groupby("case_id")
                     if cid in violated_cids and act in cdf["activity"].values)
        rows.append({"Activity": act, "Appearances": total,
                     "In Violated Cases": viol, "SLA Risk Rate": viol / total if total else 0})
    risk_df = pd.DataFrame(rows).sort_values("SLA Risk Rate", ascending=False)
    st.dataframe(
        risk_df.style.background_gradient(cmap="RdYlGn_r", subset=["SLA Risk Rate"])
                     .format({"SLA Risk Rate": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )


# ── TAB 3: Model Performance ─────────────────────────────────────────────────
with tab3:
    st.header("Model Performance")

    m = metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy",  f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall",    f"{m['recall']:.3f}")
    c4.metric("F1-Score",  f"{m['f1']:.3f}")
    c5.metric("ROC-AUC",   f"{m['roc_auc']:.3f}")
    c6.metric("PR-AUC",    f"{m['pr_auc']:.3f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training History")
        hist_df = pd.DataFrame({
            "epoch"     : list(range(1, len(history["train_loss"])+1)),
            "Train Loss": history["train_loss"],
            "Val F1"    : history["val_f1"],
        })
        fig_hist = px.line(hist_df, x="epoch", y=["Train Loss", "Val F1"],
                           labels={"value": "Score", "variable": "Metric"},
                           color_discrete_map={"Train Loss": RED, "Val F1": BLUE})
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")
        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"ROC (AUC={m['roc_auc']:.3f})",
                                          line=dict(color=BLUE, width=2)))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                          line=dict(color="gray", dash="dash"), name="Random"))
            fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                                   title="ROC Curve", height=380)
            st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader("Prefix-Stratified Evaluation (Early Warning)")
    if prefix_rows:
        pref_df = pd.DataFrame(prefix_rows)
        fig_pref = px.line(pref_df, x="completion",
                           y=["f1", "precision", "recall", "roc_auc"],
                           markers=True, title="Model Performance at Each Completion Level",
                           labels={"value": "Score", "variable": "Metric"},
                           color_discrete_map={"f1":"#3498db","precision":"#2ecc71",
                                               "recall":"#e74c3c","roc_auc":"#9b59b6"})
        st.plotly_chart(fig_pref, use_container_width=True)

        st.dataframe(
            pref_df.style.background_gradient(cmap="Blues", subset=["f1","roc_auc"])
                         .format({c: "{:.3f}" for c in ["f1","precision","recall","roc_auc"]}),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Precision-Recall Curve")
    if len(set(y_true)) > 1:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                     name=f"PR (AUC={m['pr_auc']:.3f})",
                                     line=dict(color=GREEN, width=2)))
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                              title="Precision-Recall Curve", height=350)
        st.plotly_chart(fig_pr, use_container_width=True)


# ── TAB 4: Early Warning Demo ─────────────────────────────────────────────────
with tab4:
    st.header("⚡ Early Warning Demo")
    st.caption("Select a test case and drag the slider to see how the model's risk score evolves.")

    # ── Case filter ──
    all_test_cids = sorted([cid for cid, sp in split_map.items() if sp == "test"])
    filter_col, picker_col = st.columns([1, 3])
    with filter_col:
        case_filter = st.radio(
            "Show cases",
            ["All", "🔴 Violated", "🟢 On-time"],
            horizontal=False,
        )
    if case_filter == "🔴 Violated":
        pool = [c for c in all_test_cids if label_map.get(c) == 1]
    elif case_filter == "🟢 On-time":
        pool = [c for c in all_test_cids if label_map.get(c) == 0]
    else:
        pool = all_test_cids

    # Annotate each option with its label so it's visible in the dropdown
    options_display = {
        c: f"{c}  {'🔴 VIOLATED' if label_map.get(c) == 1 else '🟢 On-time'}"
        for c in pool
    }
    with picker_col:
        sel_display = st.selectbox(
            f"Select case ({len(pool)} shown)",
            list(options_display.values()),
            index=0,
        )
    # Reverse-lookup the raw case_id from the display string
    sel_cid = next(c for c, d in options_display.items() if d == sel_display)

    if sel_cid:
        case_df = case_groups[sel_cid].sort_values("timestamp").reset_index(drop=True)
        n_ev    = len(case_df)
        lbl     = label_map.get(sel_cid, 0)
        dur     = labels_df[labels_df["case_id"] == sel_cid]["duration_days"].values
        dur_str = f"{dur[0]:.1f} days" if len(dur) else "—"

        col1, col2, col3 = st.columns(3)
        col1.metric("Ground Truth",  "🔴 VIOLATED" if lbl else "🟢 On-time")
        col2.metric("Total Events",  n_ev)
        col3.metric("Duration",      dur_str)

        n_shown = st.slider("Events revealed", 2, n_ev, min(n_ev, max(2, n_ev // 2)))

        risk = predict_prefix(case_df, model, activity_idx, n_shown, NORM)
        pct  = round(100 * n_shown / n_ev)

        col_g, col_t = st.columns([1, 1])
        with col_g:
            st.plotly_chart(gauge_chart(risk, f"SLA Risk — {pct}% complete"),
                            use_container_width=True)
        with col_t:
            verdict = "🔴 HIGH RISK — intervene now" if risk > 0.65 \
                      else ("🟡 MODERATE — monitor closely" if risk > 0.35
                            else "🟢 LOW RISK — on track")
            st.markdown(f"### {verdict}")
            st.markdown(f"**Violation probability:** `{risk:.1%}`")
            st.markdown(f"**Events seen:** `{n_shown}` / `{n_ev}`")
            st.markdown(f"**Completion:** `{pct}%`")

        st.divider()
        st.subheader("Risk Evolution Over Time")
        risk_curve = []
        for k in range(2, n_ev + 1):
            r = predict_prefix(case_df, model, activity_idx, k, NORM)
            risk_curve.append({"events": k, "risk": r,
                                "completion": f"{round(100*k/n_ev)}%"})
        rc_df = pd.DataFrame(risk_curve)
        fig_rc = go.Figure()
        fig_rc.add_hrect(y0=0.65, y1=1.0, fillcolor=RED,   opacity=0.08, line_width=0)
        fig_rc.add_hrect(y0=0.35, y1=0.65, fillcolor="#f39c12", opacity=0.08, line_width=0)
        fig_rc.add_hrect(y0=0.0,  y1=0.35, fillcolor=GREEN, opacity=0.08, line_width=0)
        fig_rc.add_trace(go.Scatter(x=rc_df["events"], y=rc_df["risk"],
                                     mode="lines+markers",
                                     line=dict(color=RED if lbl else GREEN, width=2.5),
                                     name="Risk Score",
                                     hovertemplate="Events: %{x}<br>Risk: %{y:.1%}"))
        fig_rc.add_hline(y=0.5, line_dash="dash", line_color="black",
                          annotation_text="Decision Threshold (50%)")
        fig_rc.add_vline(x=n_shown, line_dash="dot", line_color=BLUE,
                          annotation_text=f"Current ({n_shown} events)")
        fig_rc.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%"),
                              xaxis_title="Events Revealed",
                              yaxis_title="Violation Probability",
                              title=f"Risk Evolution — Case {sel_cid}",
                              height=380)
        st.plotly_chart(fig_rc, use_container_width=True)

        st.divider()
        st.subheader("🔍 Per-Case Bottleneck Analysis")
        st.caption("Hai góc nhìn: thời gian chờ thực tế (từ dữ liệu) + độ bất thường học được (từ GCN)")

        visible_df = case_df.iloc[:n_shown].copy()
        ts_vals    = visible_df["timestamp"].values
        acts       = visible_df["activity"].tolist()
        t0         = ts_vals[0]

        wait_days  = [0.0] + [
            (ts_vals[i] - ts_vals[i-1]) / np.timedelta64(1, "D")
            for i in range(1, len(ts_vals))
        ]
        cum_days   = [(ts_vals[i] - t0) / np.timedelta64(1, "D") for i in range(len(ts_vals))]

        # ── GCN node embedding norm for this case ──
        model.eval()
        g = case_to_graph(visible_df, activity_idx, 0, sel_cid, _sorted=True)
        g.x[:, -2] = (g.x[:, -2] - NORM["ts_m"]) / NORM["ts_s"]
        g.x[:, -1] = (g.x[:, -1] - NORM["tp_m"]) / NORM["tp_s"]
        with torch.no_grad():
            h = F.relu(model.conv1(g.x, g.edge_index))
            h = F.relu(model.conv2(h, g.edge_index))
        node_norms = h.norm(dim=1).numpy().tolist()

        # Build combined table
        btl_df = pd.DataFrame({
            "Bước":           range(1, len(acts) + 1),
            "Hoạt động":      acts,
            "Chờ (ngày)":     [round(w, 2) for w in wait_days],
            "Tích lũy (ngày)":  [round(c, 2) for c in cum_days],
            "GCN Norm":       [round(n, 4) for n in node_norms],
        })

        max_wait_idx = int(np.argmax(wait_days))
        max_norm_idx = int(np.argmax(node_norms))

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            st.markdown("**① Thời gian chờ thực tế (time_since_prev)**")
            colors_wait = [
                RED if i == max_wait_idx else ("#f39c12" if wait_days[i] > np.mean(wait_days) else BLUE)
                for i in range(len(acts))
            ]
            fig_wait = go.Figure(go.Bar(
                x=wait_days, y=acts,
                orientation="h",
                marker_color=colors_wait,
                text=[f"{w:.1f}d" for w in wait_days],
                textposition="outside",
                hovertemplate="%{y}<br>Thời gian chờ: %{x:.2f} ngày<extra></extra>",
            ))
            fig_wait.update_layout(
                title=f"Nút thắt: <b>{acts[max_wait_idx]}</b> ({wait_days[max_wait_idx]:.1f} ngày)",
                xaxis_title="Thời gian chờ (ngày)",
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="#f9f9fb",
                height=60 + 40 * len(acts),
                margin=dict(l=10, r=60, t=50, b=30),
            )
            st.plotly_chart(fig_wait, use_container_width=True)

        with col_b2:
            st.markdown("**② Độ bất thường học được (GCN embedding norm)**")
            colors_norm = [
                RED if i == max_norm_idx else ("#f39c12" if node_norms[i] > np.mean(node_norms) else BLUE)
                for i in range(len(acts))
            ]
            fig_norm = go.Figure(go.Bar(
                x=node_norms, y=acts,
                orientation="h",
                marker_color=colors_norm,
                text=[f"{n:.3f}" for n in node_norms],
                textposition="outside",
                hovertemplate="%{y}<br>GCN norm: %{x:.4f}<extra></extra>",
            ))
            fig_norm.update_layout(
                title=f"Mô hình chú ý: <b>{acts[max_norm_idx]}</b> (norm={node_norms[max_norm_idx]:.3f})",
                xaxis_title="GCN Embedding Norm",
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="#f9f9fb",
                height=60 + 40 * len(acts),
                margin=dict(l=10, r=60, t=50, b=30),
            )
            st.plotly_chart(fig_norm, use_container_width=True)

        # Summary verdict
        same = acts[max_wait_idx] == acts[max_norm_idx]
        if same:
            st.success(
                f"✅ Cả hai phương pháp đều chỉ ra **{acts[max_wait_idx]}** "
                f"(chờ {wait_days[max_wait_idx]:.1f} ngày, norm={node_norms[max_norm_idx]:.3f}) "
                f"là nút thắt chính của hồ sơ này."
            )
        else:
            st.warning(
                f"⚠️ Thời gian thực tế chỉ ra **{acts[max_wait_idx]}** "
                f"({wait_days[max_wait_idx]:.1f} ngày), "
                f"nhưng GCN chú ý đến **{acts[max_norm_idx]}** "
                f"(norm={node_norms[max_norm_idx]:.3f}) — "
                f"có thể bước này tuy không chờ lâu nhưng mang tín hiệu bất thường về cấu trúc quy trình."
            )

        st.subheader("Partial Trace")
        styled = btl_df.style \
            .background_gradient(cmap="Reds", subset=["Chờ (ngày)", "GCN Norm"]) \
            .format({"Chờ (ngày)": "{:.2f}", "Tích lũy (ngày)": "{:.2f}", "GCN Norm": "{:.4f}"}) \
            .apply(lambda row: [
                "background-color: #fde8e8; font-weight: bold"
                if row["Bước"] - 1 == max_wait_idx or row["Bước"] - 1 == max_norm_idx
                else "" for _ in row
            ], axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ── TAB 5: Bottleneck Analysis ────────────────────────────────────────────────
with tab5:
    st.header("🔍 Bottleneck Detection")
    st.caption("Activities most associated with SLA violations (by mean GCN node embedding magnitude).")

    model.eval()
    activity_importance = defaultdict(list)
    loader_full = DataLoader(full_graphs, batch_size=64)
    all_embs, all_lbls = [], []

    with torch.no_grad():
        for batch in loader_full:
            batch = batch.to(state["DEVICE"])
            x = F.relu(model.conv1(batch.x, batch.edge_index))
            x = F.relu(model.conv2(x, batch.edge_index))
            n_dim = batch.x.shape[1] - 2  # number of activities

            for graph_idx in batch.batch.unique():
                mask    = (batch.batch == graph_idx)
                g_nodes = batch.x[mask]
                g_emb   = x[mask]
                lbl     = int(batch.y[graph_idx].item())
                for node_i in range(g_nodes.shape[0]):
                    act_vec = g_nodes[node_i, :n_dim]
                    act_id  = int(act_vec.argmax().item())
                    act_nm  = next((k for k, v in activity_idx.items() if v == act_id), "?")
                    imp     = float(g_emb[node_i].norm().item())
                    if lbl == 1:
                        activity_importance[act_nm].append(imp)

                # t-SNE embeddings
                g_pool = global_mean_pool(x[mask], torch.zeros(mask.sum(), dtype=torch.long))
                all_embs.append(g_pool.numpy())
                all_lbls.append(lbl)

    imp_df = pd.DataFrame([
        {"Activity": a, "Mean Importance": np.mean(v), "Count": len(v)}
        for a, v in activity_importance.items()
    ]).sort_values("Mean Importance", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig_imp = px.bar(imp_df, x="Mean Importance", y="Activity",
                         orientation="h", color="Mean Importance",
                         color_continuous_scale="Reds",
                         title="Activity Importance in Violated Cases",
                         labels={"Mean Importance": "Mean Node Embedding Norm"})
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.subheader("Importance Table")
        st.dataframe(
            imp_df.style.background_gradient(cmap="Reds", subset=["Mean Importance"])
                        .format({"Mean Importance": "{:.4f}"}),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.subheader("Graph Embedding Space (t-SNE)")
    if len(all_embs) > 10:
        embs = np.concatenate(all_embs, axis=0)
        lbls = np.array(all_lbls)
        tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(embs)-1),
                    max_iter=500)
        emb2d = tsne.fit_transform(embs)
        tsne_df = pd.DataFrame({
            "x": emb2d[:, 0], "y": emb2d[:, 1],
            "label": ["Violated" if l else "On-time" for l in lbls],
        })
        fig_tsne = px.scatter(tsne_df, x="x", y="y", color="label",
                               color_discrete_map={"On-time": GREEN, "Violated": RED},
                               title="t-SNE of Graph Embeddings",
                               labels={"x": "t-SNE 1", "y": "t-SNE 2"},
                               opacity=0.7)
        st.plotly_chart(fig_tsne, use_container_width=True)
