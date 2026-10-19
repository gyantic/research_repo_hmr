"""Debug-print switch shared by SAM3DBody and SAM3DBodyEstimator.

SAM3D_QUIET=1 (set by run_publisher_ue.py) suppresses the per-frame timing and
debug prints so they do not perturb real-time latency. The flag is read once at
import; scripts may toggle ``QUIET`` at runtime (see profile_components.py).
"""

import os

QUIET = os.environ.get("SAM3D_QUIET", "0") == "1"


def quiet_print(*args, **kwargs):
    if not QUIET:
        print(*args, **kwargs)
