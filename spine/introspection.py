"""
Self-diagnostic routines to assess module performance.
"""
import logging

logger = logging.getLogger(__name__)

from tools.why import WhyEngine
from tools.policy import PolicyEnforcer


class Introspection:
    def __init__(self, spine=None, logbook=None, policies=None):
        self.spine = spine
        self.logbook = logbook
        self.why = WhyEngine(policies or {})
        self.policy = PolicyEnforcer(policies or {})

    def process(self, x):
        # Just pass through
        return x

    def evaluate(self, x, targets=None):
        report = self.assess(x, targets)
        # Minimal loss for spine
        return 0.0

    def assess(self, inputs, targets):
        """Run each module on data, record metrics"""
        report = {}
        x = inputs
        for name, module in (self.spine.modules.items() if self.spine else []):
            try:
                out = module.process(x)
                loss = module.evaluate(out, targets) if hasattr(module, 'evaluate') else 0.0
                report[name] = float(loss)
                x = out
            except Exception:
                report[name] = None
        if self.logbook:
            self.logbook.record_introspection(report)
        # Metacognition: explain why we assessed
        reason = self.why.reason_for('introspection_assess', {"objective": "monitor performance and enforce policies"})
        self.why.record(self.logbook, reason)
        return report

class Introspector:
    def analyze(self, output):
        return {
            "meta_trainer": float((output.mean() % 1.0).item()),
            "curiosity": float((output.std() % 1.0).item()),
            "introspection": float((output.max() % 1.0).item())
        }

# Factory for plugin system
def create(config=None):
    policies = (config or {}).get('policies', {}) if isinstance(config, dict) else {}
    return Introspection(policies=policies)
