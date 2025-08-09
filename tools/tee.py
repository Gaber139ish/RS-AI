import hashlib
from typing import Dict, Any


def attest_run(context: Dict[str, Any]) -> Dict[str, Any]:
    blob = repr(sorted(context.items())).encode('utf-8')
    h = hashlib.sha256(blob).hexdigest()
    return {"attestation": h, "context": context}