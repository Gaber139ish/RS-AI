from typing import List, Dict, Any
from tools.dp import add_dp_noise, secure_aggregate


def aggregate_metrics(metrics_list: List[Dict[str, Any]], sigma: float = 1e-3) -> Dict[str, Any]:
    total_loss = 0.0
    for m in metrics_list:
        total_loss += float(m.get('loss', 0.0))
    noisy = float(add_dp_noise(total_loss, sigma=sigma))
    return {"loss_sum_noisy": noisy}