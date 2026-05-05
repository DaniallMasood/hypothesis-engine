"""
app.py — Streamlit interface for the patient-level hypothesis engine.
Run with: streamlit run app.py
Requires: ANTHROPIC_API_KEY environment variable
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Hypothesis Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_graph():
    from build_kg import load_graph as _load, save_graph
    import pickle
    if Path("kg.pkl").exists():
        with open("kg.pkl", "rb") as f:
            return pickle.load(f)
    G = _load()
    save_graph(G)
    return G


@st.cache_data(show_spinner=False)
def _get_patient_summary(path: str):
    from parse_patient import patient_summary
    return patient_summary(path)


@st.cache_data(show_spinner=False)
def _get_signals(path: str):
    from parse_patient import extract_signals
    return extract_signals(path)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧬 Hypothesis Engine")
    st.caption("Long COVID · Biomarker KG · Claude AI")
    st.divider()

    patient_file = st.selectbox(
        "Patient record",
        options=list(Path("data").glob("patient_*.json")),
        format_func=lambda p: p.name,
    )

    api_key = st.text_input(
        "Anthropic API key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Required only for hypothesis synthesis",
    )

    st.divider()
    run_synthesis = st.button("▶ Run Hypothesis Synthesis", type="primary", use_container_width=True)
    st.caption("Calls Claude claude-opus-4-7 with KG context.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

patient_path = str(patient_file)
summary = _get_patient_summary(patient_path)
signals = _get_signals(patient_path)
G = _get_graph()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(f"Patient {summary['patient_id']}")
demo = summary["demographics"]
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Age / Sex", f"{demo.get('age')} / {demo.get('sex', '').capitalize()}")
col_b.metric("Bell Scale", f"{summary['functional_status'].get('bell_scale_score', '?')}/100")
col_c.metric("Months post-COVID", summary["functional_status"].get("months_since_covid_onset", "?"))
col_d.metric("Anomaly Signals", summary["signal_count"])

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_kg, tab_signals, tab_hypothesis = st.tabs(
    ["📋 Overview", "🕸 Knowledge Graph", "📊 Signals", "🔬 Hypotheses"]
)

# ── Overview ───────────────────────────────────────────────────────────────
with tab_overview:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confirmed Diagnoses")
        for dx in summary["diagnoses"]:
            st.markdown(f"- **{dx}**")

        st.subheader("Current Medications")
        patient_raw = json.loads(Path(patient_path).read_text())
        for med in patient_raw.get("current_medications", []):
            response_color = {"partial": "🟡", "modest_improvement": "🟢", "subjective_improvement": "🟢"}.get(
                med.get("response", ""), "⚪"
            )
            st.markdown(f"- {response_color} **{med['name']}** {med.get('dose', '')} — {med.get('response', '')}")

    with col2:
        st.subheader("Symptoms")
        severity_icon = {"severe": "🔴", "moderate": "🟠", "mild": "🟡"}
        for symp in patient_raw.get("symptoms", []):
            icon = severity_icon.get(symp.get("severity", "mild"), "⚪")
            st.markdown(f"- {icon} **{symp['name']}** ({symp.get('severity', '')})")

        st.subheader("Pending Investigations")
        for inv in summary.get("pending_investigations", []):
            st.markdown(f"- `{inv.get('status', '?').upper()}` — {inv['test']}")

# ── Knowledge Graph ────────────────────────────────────────────────────────
with tab_kg:
    st.subheader("Interactive Knowledge Graph")
    st.caption(
        f"KG contains **{G.number_of_nodes()} nodes** and **{G.number_of_edges()} edges** "
        f"across 6 biological modules. Patient-relevant nodes are highlighted."
    )

    col_build, col_info = st.columns([1, 3])
    with col_build:
        if st.button("Build / Refresh Visualization"):
            from build_kg import build_visualization
            highlight = [s.node_id for s in signals[:20]]
            out_path = build_visualization(G, highlight_nodes=highlight)
            st.session_state["kg_html_path"] = out_path

    kg_path = st.session_state.get("kg_html_path", "output/kg_visualization.html")
    if Path(kg_path).exists():
        html_content = Path(kg_path).read_text()
        st.components.v1.html(html_content, height=820, scrolling=False)
    else:
        st.info("Click **Build / Refresh Visualization** to render the knowledge graph.")

# ── Signals ────────────────────────────────────────────────────────────────
with tab_signals:
    st.subheader("Ranked Anomaly Signals")

    import pandas as pd

    rows = []
    for sig in signals:
        rows.append({
            "Rank": signals.index(sig) + 1,
            "Node ID": sig.node_id,
            "Signal": sig.name,
            "Direction": sig.direction,
            "Entity": sig.entity,
            "Type": sig.entity_type,
            "Role": sig.role,
            "Fold Δ": sig.fold_change,
            "Score": sig.severity_score,
        })

    df = pd.DataFrame(rows)

    # Colour-code direction
    def _direction_style(val):
        colors = {"elevated": "background-color:#4e1a1a", "low": "background-color:#1a2e4e",
                  "present": "background-color:#1a3e1a", "functional": "background-color:#2e2e1a"}
        return colors.get(val, "")

    st.dataframe(
        df.style.map(_direction_style, subset=["Direction"]).format({"Fold Δ": "{:.2f}", "Score": "{:.3f}"}),
        use_container_width=True,
        height=520,
    )

    st.subheader("Domain Distribution")
    domain_counts = df["Type"].value_counts()
    st.bar_chart(domain_counts)

# ── Hypotheses ─────────────────────────────────────────────────────────────
with tab_hypothesis:
    st.subheader("Mechanistic Hypothesis Report")

    output_file = Path(f"output/hypothesis_{summary['patient_id']}.json")

    if run_synthesis:
        if not api_key:
            st.error("Please enter an Anthropic API key in the sidebar.")
        else:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            with st.spinner("Running hypothesis synthesis pipeline..."):
                try:
                    from run_agent import run_pipeline
                    result = run_pipeline(patient_path)
                    st.session_state["hypothesis_result"] = result
                    st.success("Synthesis complete!")
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")

    # Display cached result from file or session
    result = st.session_state.get("hypothesis_result")
    if result is None and output_file.exists():
        result = json.loads(output_file.read_text())

    if result:
        usage = result.get("usage", {})
        col_u1, col_u2, col_u3 = st.columns(3)
        col_u1.metric("Input tokens", f"{usage.get('input_tokens', 0):,}")
        col_u2.metric("Output tokens", f"{usage.get('output_tokens', 0):,}")
        col_u3.metric("Cache read tokens", f"{usage.get('cache_read_input_tokens', 0):,}")

        st.markdown("---")
        st.markdown(result["hypothesis_report"])

        st.download_button(
            "⬇ Download Full Report (JSON)",
            data=json.dumps(result, indent=2),
            file_name=f"hypothesis_{summary['patient_id']}.json",
            mime="application/json",
        )
    else:
        st.info(
            "No hypothesis report yet. "
            "Enter your Anthropic API key in the sidebar and click **▶ Run Hypothesis Synthesis**."
        )
