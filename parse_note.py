"""
parse_note.py — Clinical note preprocessing agent.
Calls Claude API to extract structured patient JSON from unstructured clinical notes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"
OUTPUT_PATH = Path("data/patient_parsed.json")

_PROMPT = """\
You are a clinical data structuring agent for a complex chronic illness research system.

Your job is to read an unstructured clinical note and extract a structured patient record
following the BEST FDA-NIH biomarker data model.

Output ONLY valid JSON matching this schema exactly. No preamble, no explanation, no markdown.

Schema:
{
  "patient_id": "string",
  "age": number,
  "sex": "string",
  "diagnoses": ["string"],
  "biomarkers": [
    {
      "node_id": "string or null if unknown",
      "assessed_biomarker_entity": "string",
      "assessed_entity_type": "protein|metabolite|cell|antibody|hormone|cytokine|coagulation_factor|other",
      "reporting_term": "increased|decreased|presence of|absence of|abnormal",
      "specimen": "string",
      "condition": "string",
      "best_biomarker_role": "diagnostic|monitoring|prognostic|risk|unknown",
      "value": "string or null",
      "unit": "string or null",
      "loinc_code": "string or null"
    }
  ],
  "functional_tests": [
    {
      "test_name": "string",
      "result": "string",
      "interpretation": "string"
    }
  ],
  "medications": ["string"],
  "clinical_timeline": "string summarizing key events"
}

Extract every measurable finding, lab value, and functional test result mentioned.
For reporting_term: use increased/decreased for quantitative findings, presence of/absence of for qualitative ones.
For node_id: leave null — the system will match to the knowledge graph automatically.
If a field cannot be determined, use null.

Clinical note:
CLINICAL_NOTE_PLACEHOLDER"""


def _normalize_role(role: str) -> str:
    role = (role or "monitoring").lower().strip()
    if "biomarker" not in role:
        role = f"{role} biomarker"
    return role


def _transform_to_pipeline_format(parsed: dict) -> dict:
    """Transform flat Claude-parsed schema into pipeline-compatible patient JSON format."""
    biomarkers_flat = parsed.get("biomarkers", [])

    elevated: list[dict] = []
    low: list[dict] = []

    for i, bm in enumerate(biomarkers_flat):
        reporting_term = (bm.get("reporting_term") or "").lower().strip()
        node_id = bm.get("node_id") or f"BM{i + 1:03d}"
        entity = bm.get("assessed_biomarker_entity", "")
        name = f"{reporting_term.capitalize()} {entity}".strip()
        role = _normalize_role(bm.get("best_biomarker_role", "monitoring"))

        record = {
            "node_id": node_id,
            "name": name,
            "assessed_biomarker_entity": entity,
            "assessed_entity_type": bm.get("assessed_entity_type", "other"),
            "reporting_term": reporting_term,
            "specimen": bm.get("specimen", "serum"),
            "best_biomarker_role": role,
            "value": bm.get("value"),
            "unit": bm.get("unit"),
            "loinc_code": bm.get("loinc_code"),
        }

        if reporting_term in ("increased", "presence of", "abnormal"):
            elevated.append(record)
        elif reporting_term in ("decreased", "absence of"):
            low.append(record)
        else:
            elevated.append(record)

    functional_tests = []
    for i, ft in enumerate(parsed.get("functional_tests", [])):
        functional_tests.append({
            "node_id": f"FT{i + 1:03d}",
            "test_name": ft.get("test_name", ""),
            "result": ft.get("result", ""),
            "interpretation": ft.get("interpretation", ""),
        })

    return {
        "patient_id": parsed.get("patient_id", "PARSED"),
        "demographics": {
            "sex": parsed.get("sex", "unknown"),
            "age": parsed.get("age"),
            "enrolled": None,
        },
        "diagnoses": [
            {"name": d, "id": None, "confirmed": True}
            for d in parsed.get("diagnoses", [])
        ],
        "biomarkers": {
            "elevated": elevated,
            "low": low,
        },
        "functional_tests": functional_tests,
        "symptoms": [],
        "current_medications": [
            {"name": m, "dose": "", "response": "unknown"}
            for m in parsed.get("medications", [])
        ],
        "prior_treatments_no_response": [],
        "pending_investigations": [],
        "clinical_timeline": [
            {
                "date": "N/A",
                "event": "Clinical history",
                "details": parsed.get("clinical_timeline", ""),
            }
        ],
        "functional_status": {
            "bell_scale_score": None,
            "months_since_covid_onset": None,
        },
    }


def parse_note(clinical_note: str) -> dict:
    """
    Parse unstructured clinical note text into a structured patient record.

    Makes one Claude API call, transforms the result into the pipeline-compatible
    patient JSON format, saves to data/patient_parsed.json, and returns the dict.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = _PROMPT.replace("CLINICAL_NOTE_PLACEHOLDER", clinical_note)

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown code fences if model adds them despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0].strip()

    parsed_flat = json.loads(raw_text)
    result = _transform_to_pipeline_format(parsed_flat)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    import sys

    note_path = sys.argv[1] if len(sys.argv) > 1 else "data/demo_clinical_note.txt"
    note_text = Path(note_path).read_text()
    result = parse_note(note_text)
    print(f"Parsed patient_id: {result['patient_id']}")
    print(f"Biomarkers elevated: {len(result['biomarkers']['elevated'])}")
    print(f"Biomarkers low:      {len(result['biomarkers']['low'])}")
    print(f"Functional tests:    {len(result['functional_tests'])}")
    print(f"Medications:         {len(result['current_medications'])}")
    print(f"Saved → {OUTPUT_PATH}")
