from __future__ import annotations

import hashlib
from datetime import datetime


def generate_job_id() -> str:
    """Generate a short unique job ID."""
    ts = datetime.now().isoformat()
    return hashlib.md5(ts.encode()).hexdigest()[:8]
