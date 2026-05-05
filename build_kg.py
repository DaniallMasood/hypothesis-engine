"""
build_kg.py — Load nodes.csv + triples.csv into a NetworkX DiGraph and produce
a pyvis interactive HTML visualization. Saves the graph as kg.pkl for reuse.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

NODE_COLORS = {
    "Biomarker": "#4e79a7",
    "Condition": "#e15759",
    "Pathway": "#f28e2b",
    "Gene": "#76b7b2",
    "Drug": "#59a14f",
    "Symptom": "#edc948",
}
CONFIDENCE_WIDTH = {"high": 3.5, "moderate": 2.0, "low": 1.0}
EDGE_COLORS = {
    "activates": "#e15759",
    "inhibits": "#59a14f",
    "associated_with": "#9c755f",
    "mechanism_of": "#f28e2b",
    "diagnostic_of": "#4e79a7",
    "treated_by": "#76b7b2",
    "has_phenotype": "#edc948",
    "comorbid_with": "#b07aa1",
    "predicts_severity_of": "#ff9da7",
    "encodes": "#bab0ac",
    "is_subtype_of": "#d37295",
    "regulates": "#fabfd2",
    "produces": "#8cd17d",
    "is_target_of": "#a0cbe8",
}


def load_graph() -> nx.DiGraph:
    nodes_df = pd.read_csv(DATA_DIR / "nodes.csv")
    triples_df = pd.read_csv(DATA_DIR / "triples.csv")

    G = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        attrs = row.dropna().to_dict()
        G.add_node(row["node_id"], **attrs)

    for _, row in triples_df.iterrows():
        G.add_edge(
            row["subject_id"],
            row["object_id"],
            triple_id=row["triple_id"],
            predicate=row["predicate"],
            confidence=row.get("confidence", "moderate"),
            evidence_type=row.get("evidence_type", ""),
            pmid=row.get("pmid", ""),
            module=row.get("module", ""),
            notes=row.get("notes", ""),
        )

    return G


def build_visualization(G: nx.DiGraph, highlight_nodes: list[str] | None = None) -> str:
    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    highlight_nodes = set(highlight_nodes or [])

    for node_id, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type", "Unknown")
        color = NODE_COLORS.get(node_type, "#aaa")
        label = attrs.get("name", node_id)
        # Truncate long labels for readability
        if len(label) > 30:
            label = label[:27] + "..."
        size = 25 if node_id in highlight_nodes else 15
        border = "#ffffff" if node_id in highlight_nodes else color
        tooltip = (
            f"<b>{attrs.get('name', node_id)}</b><br>"
            f"Type: {node_type}<br>"
            f"Domain: {attrs.get('domain', '')}<br>"
            f"Role: {attrs.get('best_biomarker_role', '')}<br>"
            f"Notes: {str(attrs.get('notes', ''))[:120]}"
        )
        net.add_node(
            node_id,
            label=label,
            color={"background": color, "border": border},
            size=size,
            title=tooltip,
            font={"size": 11, "color": "white"},
            borderWidth=3 if node_id in highlight_nodes else 1,
        )

    for src, dst, attrs in G.edges(data=True):
        pred = attrs.get("predicate", "")
        color = EDGE_COLORS.get(pred, "#888")
        width = CONFIDENCE_WIDTH.get(attrs.get("confidence", "moderate"), 2.0)
        net.add_edge(
            src,
            dst,
            label=pred,
            color=color,
            width=width,
            title=f"{pred} [{attrs.get('confidence', '')}]\n{attrs.get('notes', '')[:100]}",
            arrows="to",
            font={"size": 8, "color": "#cccccc"},
        )

    legend_html = _legend_html()
    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {
        "stabilization": {"iterations": 200}
      }
    }
    """)

    out_path = OUTPUT_DIR / "kg_visualization.html"
    net.save_graph(str(out_path))

    # Inject legend into the HTML
    html = out_path.read_text()
    html = html.replace("</body>", legend_html + "\n</body>")
    out_path.write_text(html)

    return str(out_path)


def _legend_html() -> str:
    items = "".join(
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{c};border-radius:50%;margin-right:4px;"></span>{t}  '
        for t, c in NODE_COLORS.items()
    )
    return (
        f'<div style="position:fixed;bottom:20px;left:20px;background:rgba(0,0,0,0.7);'
        f'color:white;padding:10px 14px;border-radius:8px;font-size:12px;z-index:9999;">'
        f"<b>Node Types</b><br>{items}</div>"
    )


def save_graph(G: nx.DiGraph, path: str = "kg.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_saved_graph(path: str = "kg.pkl") -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Loading knowledge graph...")
    G = load_graph()
    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    save_graph(G)
    print("  Saved → kg.pkl")

    out = build_visualization(G)
    print(f"  Visualization → {out}")
