from typing import Dict, Any


def red_team_oracle(sample: Any) -> Dict[str, Any]:
    # Return flags for potentially unsafe content
    return {"unsafe": False, "reason": "stub"}


def eval_oracle(metrics: Dict[str, Any]) -> Dict[str, Any]:
    # Summarize helpfulness metrics
    return {"usefulness": float(metrics.get('ai_score', 0.0))}