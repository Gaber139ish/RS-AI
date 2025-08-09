from typing import Dict, Any, Optional
import time


class WhyEngine:
    def __init__(self, policies: Optional[Dict[str, Any]] = None):
        self.policies = policies or {}

    def reason_for(self, action: str, context: Dict[str, Any]) -> str:
        ethics = self.policies.get('ethics', 'Do no harm, maximize beneficial impact')
        privacy = self.policies.get('privacy', 'Protect user and data privacy; avoid storing sensitive data')
        security = self.policies.get('security', 'Maintain integrity and resilience; least privilege')
        objective = context.get('objective', 'Advance RS-AI capability and utility')
        guard = f"Ethics: {ethics}. Privacy: {privacy}. Security: {security}."
        return f"Action: {action}. Objective: {objective}. {guard}"

    def record(self, logbook, reason: str):
        if not logbook:
            return
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        entry = f"**Why** [{ts}]\n- {reason}\n"
        logbook.record(entry)