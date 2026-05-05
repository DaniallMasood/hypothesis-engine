"""
run_agent.py — Orchestrates the full hypothesis synthesis pipeline:
  1. parse_patient  → ranked anomaly signals
  2. query_kg       → relevant KG subgraphs per signal
  3. Claude API     → hypothesis synthesis with prompt caching
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic

from build_kg import load_graph
from parse_patient import AnomalySignal, extract_signals, patient_summary
from query_kg import query_all_signals, subgraph_for_patient

MODEL = "claude-opus-4-7"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_signals(signals: list[AnomalySignal]) -> str:
    lines = []
    for i, s in enumerate(signals[:20], 1):
        lines.append(
            f"{i:2}. [{s.role[:14]}] {s.name}  "
            f"direction={s.direction}  fold={s.fold_change:.2f}  score={s.severity_score:.3f}"
            f"  entity={s.entity} ({s.entity_type})"
        )
    return "\n".join(lines)


def _format_subgraph_edges(results: list[dict], max_edges: int = 60) -> str:
    lines = []
    seen = set()
    for r in results:
        for edge_list in (r.get("downstream_edges", []), r.get("upstream_edges", []), r.get("drug_edges", [])):
            for e in edge_list:
                triple = (
                    e["from"]["name"],
                    e["edge"]["predicate"],
                    e["to"]["name"],
                )
                if triple not in seen:
                    seen.add(triple)
                    conf = e["edge"].get("confidence", "")
                    mod = e["edge"].get("module", "")
                    lines.append(
                        f"  {triple[0]}  --[{triple[1]}]-->  {triple[2]}"
                        f"  [{conf}] module={mod}"
                    )
                if len(lines) >= max_edges:
                    break
    return "\n".join(lines)


def _system_prompt() -> str:
    return (
        "You are a precision-medicine hypothesis engine specialising in Long COVID, "
        "dysautonomia, MCAS, ME/CFS, and related post-viral syndromes. "
        "You reason over structured biomarker and knowledge-graph data to generate "
        "mechanistically grounded, clinically actionable hypotheses. "
        "Always cite the biological evidence (pathway, gene, biomarker) for each claim. "
        "Use structured output sections exactly as requested."
    )


def _user_prompt(summary: dict, kg_results: list[dict]) -> str:
    signals_text = _format_signals(
        [AnomalySignal(**{k: v for k, v in s.items() if k != "raw"}, raw={})
         for s in summary["all_signals"][:20]]
        if summary.get("all_signals")
        else []
    )
    edges_text = _format_subgraph_edges(kg_results)

    return f"""## Patient {summary['patient_id']}
Sex: {summary['demographics'].get('sex')}  Age: {summary['demographics'].get('age')}
Confirmed diagnoses: {', '.join(summary['diagnoses'])}
Current medications: {', '.join(summary['current_medications'])}
Functional status (Bell scale): {summary['functional_status'].get('bell_scale_score', 'N/A')}/100
Months since COVID onset: {summary['functional_status'].get('months_since_covid_onset', 'N/A')}

## Ranked Anomaly Signals ({summary['signal_count']} total, top 20 shown)
{signals_text}

## Knowledge Graph Evidence (retrieved subgraph — top 60 triples)
{edges_text}

## Pending Investigations
{json.dumps(summary.get('pending_investigations', []), indent=2)}

---

Based on the above patient data and knowledge-graph evidence, generate a structured hypothesis report with these exact sections:

### 1. Active Biological Modules
List the 3–5 dominant biological modules driving this patient's phenotype. For each module state:
- Module name
- Key supporting biomarkers
- Mechanistic pathway activated

### 2. Primary Mechanistic Hypotheses
Generate 3 ranked mechanistic hypotheses that best explain this patient's clinical picture. For each:
- **Hypothesis**: One-sentence mechanistic claim
- **Evidence**: 2–3 KG triples or biomarker findings that support it
- **Confidence**: High / Moderate / Low with rationale

### 3. Treatment Gap Analysis
For each active module, identify:
- What is currently being treated and response quality
- What mechanistic targets remain untreated
- Highest-priority next therapeutic option with mechanistic rationale

### 4. Pending Investigation Prioritisation
Rank the pending investigations by expected mechanistic yield and explain the rationale for the top 2.

### 5. Biomarker Monitoring Targets
List 3–4 biomarkers to track over the next 3–6 months as mechanistic readouts, and state what change would signal improvement vs. progression.
"""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(patient_path: str = "data/patient_js0047.json") -> dict[str, Any]:
    print("Step 1: Parsing patient record...")
    summary = patient_summary(patient_path)
    signals = extract_signals(patient_path)
    print(f"  → {len(signals)} anomaly signals extracted")

    print("Step 2: Querying knowledge graph...")
    G = load_graph()
    kg_results = query_all_signals(signals, G, top_n=18)
    subgraph = subgraph_for_patient(signals, G)
    print(f"  → {len(subgraph['node_ids'])} nodes in patient subgraph")

    print("Step 3: Calling Claude API (hypothesis synthesis)...")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system = _system_prompt()
    user = _user_prompt(summary, kg_results)

    # Use prompt caching on the large, stable system prompt + patient context block
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    )

    hypothesis_text = response.content[0].text
    usage = response.usage

    print(f"  → Synthesis complete  "
          f"(input={usage.input_tokens}, output={usage.output_tokens}, "
          f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)})")

    result = {
        "patient_id": summary["patient_id"],
        "signal_count": summary["signal_count"],
        "top_signals": summary["top_signals"],
        "subgraph_node_ids": subgraph["signal_node_ids"],
        "hypothesis_report": hypothesis_text,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        },
    }

    out_path = OUTPUT_DIR / f"hypothesis_{summary['patient_id']}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  → Saved → {out_path}")

    return result


if __name__ == "__main__":
    result = run_pipeline()
    print("\n" + "=" * 70)
    print(result["hypothesis_report"])
