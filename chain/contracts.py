from typing import Any, Dict, List


class Contract:
    def __init__(self, kind: str, params: Dict[str, Any]):
        self.kind = kind
        self.params = params

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Returns list of actions
        actions: List[Dict[str, Any]] = []
        if self.kind == 'mint_if_metric':
            metric = self.params.get('metric')
            threshold = float(self.params.get('threshold', 0.0))
            amount = float(self.params.get('amount', 0.0))
            who = self.params.get('to')
            val = float(context.get('metrics', {}).get(metric, 0.0))
            if val >= threshold and who is not None and amount > 0.0:
                actions.append({'type': 'mint', 'to': who, 'amount': amount})
        elif self.kind == 'transfer_if':
            metric = self.params.get('metric')
            threshold = float(self.params.get('threshold', 0.0))
            amount = float(self.params.get('amount', 0.0))
            who_from = self.params.get('from')
            who_to = self.params.get('to')
            val = float(context.get('metrics', {}).get(metric, 0.0))
            if val >= threshold and who_from and who_to and amount > 0.0:
                actions.append({'type': 'transfer', 'from': who_from, 'to': who_to, 'amount': amount})
        return actions


class ContractEngine:
    def __init__(self, contracts: List[Contract]):
        self.contracts = contracts

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for c in self.contracts:
            actions.extend(c.evaluate(context))
        return actions

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> 'ContractEngine':
        items = cfg.get('contracts', []) if cfg else []
        return ContractEngine([Contract(it.get('kind'), it.get('params', {})) for it in items])