"""
query_kg.py — Takes ranked anomaly signals and traverses the knowledge graph
to retrieve mechanistically relevant subgraphs for each signal.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Optional

import networkx as nx

from parse_patient import AnomalySignal


def load_graph(path: str = "kg.pkl") -> nx.DiGraph:
    if not Path(path).exists():
        # Build on demand if pickle not present
        from build_kg import load_graph as _load, save_graph
        G = _load()
        save_graph(G, path)
        return G
    with open(path, "rb") as f:
        return pickle.load(f)


def _node_summary(G: nx.DiGraph, node_id: str) -> dict:
    attrs = G.nodes.get(node_id, {})
    return {
        "node_id": node_id,
        "name": attrs.get("name", node_id),
        "node_type": attrs.get("node_type", ""),
        "domain": attrs.get("domain", ""),
        "notes": str(attrs.get("notes", ""))[:200],
        "best_biomarker_role": attrs.get("best_biomarker_role", ""),
        "ontology_id": attrs.get("ontology_id", ""),
    }


def _edge_summary(attrs: dict) -> dict:
    return {
        "predicate": attrs.get("predicate", ""),
        "confidence": attrs.get("confidence", ""),
        "module": attrs.get("module", ""),
        "notes": str(attrs.get("notes", ""))[:150],
        "pmid": attrs.get("pmid", ""),
    }


def query_signal(
    G: nx.DiGraph,
    signal: AnomalySignal,
    hops: int = 2,
    max_neighbors: int = 12,
) -> dict[str, Any]:
    """
    For a given signal node, retrieve:
      - Direct successors (what this biomarker activates/causes)
      - Direct predecessors (what encodes/drives this biomarker)
      - Conditions this signal is diagnostic/associated with
      - Drugs that target connected pathways
      - Symptoms downstream
    Returns a structured subgraph dict ready for the LLM prompt.
    """
    node_id = signal.node_id
    if node_id not in G:
        return {"node_id": node_id, "found": False, "subgraph": []}

    edges_out: list[dict] = []
    edges_in: list[dict] = []
    visited: set[str] = {node_id}
    frontier: list[str] = [node_id]

    for _ in range(hops):
        next_frontier: list[str] = []
        for n in frontier:
            # Outgoing edges: what does this node activate/cause/associate with?
            for _, nbr, edata in list(G.out_edges(n, data=True))[:max_neighbors]:
                if nbr not in visited:
                    edges_out.append({
                        "from": _node_summary(G, n),
                        "to": _node_summary(G, nbr),
                        "edge": _edge_summary(edata),
                    })
                    visited.add(nbr)
                    next_frontier.append(nbr)
            # Incoming edges: what drives/encodes this node?
            for src, _, edata in list(G.in_edges(n, data=True))[:max_neighbors]:
                if src not in visited:
                    edges_in.append({
                        "from": _node_summary(G, src),
                        "to": _node_summary(G, n),
                        "edge": _edge_summary(edata),
                    })
                    visited.add(src)
                    next_frontier.append(src)
        frontier = next_frontier

    # Collect drugs targeting any node in the subgraph
    drug_edges: list[dict] = []
    for nid in visited:
        for src, _, edata in G.in_edges(nid, data=True):
            src_type = G.nodes.get(src, {}).get("node_type", "")
            pred = edata.get("predicate", "")
            if src_type == "Drug" and pred in ("inhibits", "targets", "treated_by"):
                if src not in visited:
                    drug_edges.append({
                        "from": _node_summary(G, src),
                        "to": _node_summary(G, nid),
                        "edge": _edge_summary(edata),
                    })

    return {
        "node_id": node_id,
        "found": True,
        "signal_name": signal.name,
        "signal_direction": signal.direction,
        "severity_score": signal.severity_score,
        "fold_change": signal.fold_change,
        "role": signal.role,
        "downstream_edges": edges_out,
        "upstream_edges": edges_in,
        "drug_edges": drug_edges,
        "nodes_visited": len(visited),
    }


def query_all_signals(
    signals: list[AnomalySignal],
    G: nx.DiGraph | None = None,
    top_n: int = 15,
) -> list[dict]:
    if G is None:
        G = load_graph()
    results = []
    for sig in signals[:top_n]:
        result = query_signal(G, sig)
        results.append(result)
    return results


def subgraph_for_patient(
    signals: list[AnomalySignal],
    G: nx.DiGraph | None = None,
) -> dict[str, Any]:
    """
    Build a unified subgraph representation covering all top signals.
    Returns node set + edge list for visualization highlighting and LLM context.
    """
    if G is None:
        G = load_graph()

    all_nodes: set[str] = set()
    all_edges: list[dict] = []
    seen_edges: set[tuple] = set()

    for sig in signals[:20]:
        node_id = sig.node_id
        if node_id not in G:
            continue
        all_nodes.add(node_id)
        for _, nbr, edata in G.out_edges(node_id, data=True):
            key = (node_id, nbr)
            if key not in seen_edges:
                all_edges.append({"src": node_id, "dst": nbr, **_edge_summary(edata)})
                seen_edges.add(key)
                all_nodes.add(nbr)
        for src, _, edata in G.in_edges(node_id, data=True):
            key = (src, node_id)
            if key not in seen_edges:
                all_edges.append({"src": src, "dst": node_id, **_edge_summary(edata)})
                seen_edges.add(key)
                all_nodes.add(src)

    return {
        "node_ids": list(all_nodes),
        "edges": all_edges,
        "signal_node_ids": [s.node_id for s in signals[:20] if s.node_id in G],
    }


if __name__ == "__main__":
    from parse_patient import extract_signals

    signals = extract_signals("data/patient_js0047.json")
    G = load_graph()
    print(f"KG loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Querying top {min(5, len(signals))} signals...\n")

    results = query_all_signals(signals, G, top_n=5)
    for r in results:
        print(f"Signal: {r['signal_name']} ({r['signal_direction']})")
        print(f"  Downstream edges: {len(r['downstream_edges'])}")
        print(f"  Upstream edges:   {len(r['upstream_edges'])}")
        print(f"  Drug connections: {len(r['drug_edges'])}")
        print()
