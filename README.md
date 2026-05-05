# Hypothesis Engine

A patient-level mechanistic hypothesis engine for Long COVID and related post-viral syndromes. Given a structured patient record and a biomarker knowledge graph, the system extracts ranked anomaly signals, traverses the KG for mechanistic context, and synthesises clinically actionable hypotheses using the Claude API.

---

## Architecture

```
data/
  nodes.csv              ← Knowledge graph nodes (BiomarkerKB/BEST schema)
  triples.csv            ← 120 directed edges across 6 biological modules
  patient_js0047.json    ← Sample Long COVID patient record

build_kg.py             ← Loads CSVs into NetworkX DiGraph; exports pyvis HTML
parse_patient.py        ← Parses patient JSON; returns ranked AnomalySignal list
query_kg.py             ← Traverses KG from signal nodes; returns subgraph dicts
run_agent.py            ← Claude API pipeline: signals → KG → hypothesis report
app.py                  ← Streamlit UI
```

### Data Model

**Nodes** follow the [BiomarkerKB/BEST](https://biomarkerkb.org) schema with six node types:

| Type | Example | Key fields |
|---|---|---|
| Biomarker | `BM001` Elevated IL-6 | `best_biomarker_role`, `loinc_code`, `domain` |
| Condition | `COND001` Long COVID | `ontology_id` (MONDO) |
| Pathway | `PATH001` JAK-STAT | `ontology_id` (GO) |
| Gene | `GENE004` IL6 | `ontology_id` (HGNC) |
| Drug | `DRUG001` Propranolol | `subtype`, `drugbank_id` |
| Symptom | `SYMP001` PEM | `ontology_id` (HP) |

**Triples** encode directed mechanistic relationships:
`activates · inhibits · associated_with · mechanism_of · diagnostic_of · treated_by · has_phenotype · comorbid_with · encodes · regulates · produces · is_target_of`

Each triple carries `confidence` (high/moderate/low), `evidence_type`, and `module` (one of `immune_inflammation · viral_persistence · mast_cell · hyperadrenergic_pots · coagulation_microclot · hpa_axis · neuroinflammation`).

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Build the knowledge graph

```bash
python build_kg.py
# → kg.pkl (serialised graph)
# → output/kg_visualization.html (interactive pyvis)
```

### 4. Parse the patient and inspect signals

```bash
python parse_patient.py
```

### 5. Query the KG from patient signals

```bash
python query_kg.py
```

### 6. Run the full hypothesis pipeline

```bash
python run_agent.py
# → output/hypothesis_JS-0047.json
```

### 7. Launch the Streamlit UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — enter your API key in the sidebar and click **▶ Run Hypothesis Synthesis**.

---

## Pipeline Detail

### Signal Extraction (`parse_patient.py`)

Each anomaly signal is scored as:

```
severity_score = 0.6 × role_weight + 0.4 × (min(fold_change, 5) / 5)
```

Where `role_weight` is 1.0 for diagnostic biomarkers, 0.85 for prognostic, 0.7 for monitoring. Signals are deduplicated by `node_id` and ranked descending.

### KG Traversal (`query_kg.py`)

For each top-N signal, a 2-hop BFS retrieves:
- **Downstream**: what this biomarker activates/causes
- **Upstream**: what gene/pathway drives this biomarker
- **Drug connections**: treatments targeting any node in the subgraph

### Claude API Integration (`run_agent.py`)

Uses `claude-opus-4-7` with **prompt caching** (`cache_control: ephemeral`) on both the system prompt and the patient context block. This reduces latency and cost on repeated runs for the same patient record.

The prompt delivers:
1. Ranked signal table with fold-change and role
2. Up to 60 KG triples from the patient subgraph
3. Pending investigations list

The model returns a structured 5-section report: Active Modules → Primary Hypotheses → Treatment Gaps → Investigation Prioritisation → Monitoring Targets.

---

## Sample Patient: JS-0047

34-year-old female, 23 months post-COVID. Confirmed: Long COVID, Hyperadrenergic POTS, MCAS, Fibromyalgia. Bell scale 35/100. Unable to work.

**Top biomarker signals:**

| Biomarker | Direction | Fold Δ |
|---|---|---|
| Anti-ADRB1/ADRB2 autoantibody | present | 1.50 |
| Anti-beta2GPI IgG | present | 1.50 |
| ANA 1:160 | present | 1.50 |
| Tryptase 14.2 ng/mL | elevated | 1.25 |
| Norepinephrine 890 pg/mL on standing | elevated | 1.48 |
| DHEA-S 54 μg/dL (ref ≥98) | low | 1.81 |
| Ferritin 11 ng/mL (ref ≥20) | low | 1.82 |
| NK cell activity 12% (ref ≥20%) | low | 1.67 |

**Active biological modules:** Hyperadrenergic POTS · MCAS / Mast cell degranulation · HPA axis dysfunction · Coagulation / Microclot · Immune exhaustion / Viral persistence · Neuroinflammation

---

## Project Structure

```
hypothesis-engine/
├── data/
│   ├── nodes.csv
│   ├── triples.csv
│   └── patient_js0047.json
├── output/                     ← generated at runtime
│   ├── kg_visualization.html
│   └── hypothesis_JS-0047.json
├── build_kg.py
├── parse_patient.py
├── query_kg.py
├── run_agent.py
├── app.py
├── requirements.txt
└── README.md
```

---

## License

MIT
