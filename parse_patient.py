"""
parse_patient.py — Parse a BiomarkerKB-structured patient JSON and return
a ranked list of anomaly signals for downstream KG querying.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnomalySignal:
    node_id: str
    name: str
    direction: str           # "elevated" | "low" | "present" | "absent" | "functional"
    entity: str
    entity_type: str
    role: str                # best_biomarker_role
    domain: str
    fold_change: float       # deviation from reference (higher = more extreme)
    severity_score: float    # 0–1 composite rank score
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "direction": self.direction,
            "entity": self.entity,
            "entity_type": self.entity_type,
            "role": self.role,
            "domain": self.domain,
            "fold_change": round(self.fold_change, 2),
            "severity_score": round(self.severity_score, 3),
        }


# Role weight: diagnostic biomarkers carry more mechanistic signal than monitoring
ROLE_WEIGHTS = {
    "diagnostic biomarker": 1.0,
    "prognostic biomarker": 0.85,
    "monitoring biomarker": 0.7,
    "risk biomarker": 0.6,
}

SEVERITY_WEIGHTS = {
    "severe": 1.0,
    "moderate": 0.65,
    "mild": 0.35,
}


def _fold(record: dict) -> float:
    """Return fold-change above/below reference, defaulting to 1.5 for qualitative results."""
    for key in ("fold_above_ref", "fold_below_ref"):
        if key in record:
            return float(record[key])
    # qualitative positives (e.g. antibody present, tryptase threshold crossed)
    return 1.5


def _score(fold: float, role: str) -> float:
    rw = ROLE_WEIGHTS.get(role, 0.5)
    # Cap fold at 5x to keep score bounded; log-scale dampening
    fold_norm = min(fold, 5.0) / 5.0
    return round(0.6 * rw + 0.4 * fold_norm, 3)


def parse_biomarkers(patient: dict) -> list[AnomalySignal]:
    signals: list[AnomalySignal] = []
    bm = patient.get("biomarkers", {})

    for record in bm.get("elevated", []):
        fold = _fold(record)
        role = record.get("best_biomarker_role", "monitoring biomarker")
        signals.append(AnomalySignal(
            node_id=record["node_id"],
            name=record["name"],
            direction="elevated",
            entity=record.get("assessed_biomarker_entity", ""),
            entity_type=record.get("assessed_entity_type", ""),
            role=role,
            domain=record.get("specimen", ""),
            fold_change=fold,
            severity_score=_score(fold, role),
            raw=record,
        ))

    for record in bm.get("low", []):
        fold = _fold(record)
        role = record.get("best_biomarker_role", "monitoring biomarker")
        signals.append(AnomalySignal(
            node_id=record["node_id"],
            name=record["name"],
            direction="low",
            entity=record.get("assessed_biomarker_entity", ""),
            entity_type=record.get("assessed_entity_type", ""),
            role=role,
            domain=record.get("specimen", ""),
            fold_change=fold,
            severity_score=_score(fold, role),
            raw=record,
        ))

    return signals


def parse_functional_tests(patient: dict) -> list[AnomalySignal]:
    signals: list[AnomalySignal] = []
    for test in patient.get("functional_tests", []):
        if "node_id" not in test:
            continue
        # Calculate fold from value / ref_low if available
        val = test.get("value")
        ref_low = test.get("ref_low")
        if isinstance(val, (int, float)) and ref_low:
            fold = float(ref_low) / float(val) if float(val) > 0 else 2.0
        else:
            fold = 2.0

        interpretation = test.get("interpretation", "")
        # PEM confirmation and POTS confirmation are high-priority
        role = "diagnostic biomarker" if "confirm" in interpretation else "monitoring biomarker"
        signals.append(AnomalySignal(
            node_id=test["node_id"],
            name=test["test_name"],
            direction="functional",
            entity=test.get("assessed_biomarker_entity", test["test_name"]),
            entity_type=test.get("assessed_entity_type", "functional"),
            role=role,
            domain="functional",
            fold_change=fold,
            severity_score=_score(fold, role),
            raw=test,
        ))
    return signals


def parse_symptoms(patient: dict) -> list[AnomalySignal]:
    signals: list[AnomalySignal] = []
    for symp in patient.get("symptoms", []):
        severity = symp.get("severity", "mild")
        sw = SEVERITY_WEIGHTS.get(severity, 0.35)
        signals.append(AnomalySignal(
            node_id=symp["node_id"],
            name=symp["name"],
            direction="present",
            entity=symp["name"],
            entity_type="symptom",
            role="monitoring biomarker",
            domain=symp.get("domain", ""),
            fold_change=1.0 + sw,
            severity_score=sw,
            raw=symp,
        ))
    return signals


def extract_signals(patient_path: str | Path) -> list[AnomalySignal]:
    patient = json.loads(Path(patient_path).read_text())

    signals: list[AnomalySignal] = []
    signals.extend(parse_biomarkers(patient))
    signals.extend(parse_functional_tests(patient))
    signals.extend(parse_symptoms(patient))

    # Deduplicate by node_id, keep highest score
    seen: dict[str, AnomalySignal] = {}
    for sig in signals:
        if sig.node_id not in seen or sig.severity_score > seen[sig.node_id].severity_score:
            seen[sig.node_id] = sig

    ranked = sorted(seen.values(), key=lambda s: s.severity_score, reverse=True)
    return ranked


def patient_summary(patient_path: str | Path) -> dict:
    patient = json.loads(Path(patient_path).read_text())
    signals = extract_signals(patient_path)
    return {
        "patient_id": patient["patient_id"],
        "demographics": patient["demographics"],
        "diagnoses": [d["name"] for d in patient.get("diagnoses", [])],
        "signal_count": len(signals),
        "top_signals": [s.to_dict() for s in signals[:10]],
        "all_signals": [s.to_dict() for s in signals],
        "current_medications": [m["name"] for m in patient.get("current_medications", [])],
        "pending_investigations": patient.get("pending_investigations", []),
        "functional_status": patient.get("functional_status", {}),
    }


if __name__ == "__main__":
    summary = patient_summary("data/patient_js0047.json")
    print(f"Patient: {summary['patient_id']}")
    print(f"Diagnoses: {', '.join(summary['diagnoses'])}")
    print(f"Total anomaly signals: {summary['signal_count']}")
    print("\nTop 10 ranked signals:")
    for i, sig in enumerate(summary["top_signals"], 1):
        print(
            f"  {i:2}. [{sig['role'][:12]}] {sig['name']}"
            f"  fold={sig['fold_change']:.2f}  score={sig['severity_score']:.3f}"
        )
