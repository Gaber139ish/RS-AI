from typing import Dict, Any


def generate_proof(old_metric: float, new_metric: float, meta: Dict[str, Any] | None = None) -> dict:
    return {
        "statement": "new_metric <= old_metric",
        "old": float(old_metric),
        "new": float(new_metric),
        "proof": "zkml_stub",
        "meta": meta or {},
    }


def verify_proof(proof: dict) -> bool:
    try:
        return float(proof.get('new', 1.0)) <= float(proof.get('old', 0.0)) and proof.get('proof') == 'zkml_stub'
    except Exception:
        return False