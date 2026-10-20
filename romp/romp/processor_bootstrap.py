"""Build a ROMP ``Video_processor`` from a short ROMP argument list.

Shared by romp_video_demo.py and measurements/romp/benchmark_romp_udp_ue.py.
Run them from the ``romp/`` directory (config paths such as ``configs/video.yml``
are relative to it) with ``PYTHONPATH="$PWD:$PWD/romp"``.
"""
import sys


def build_video_processor(configs_yml, inputs, gpu, extra_args=()):
    """Return an initialised ``Video_processor``.

    ``gpu`` is ROMP's string GPU id ("0", or "-1" for CPU). ``extra_args`` are
    appended ROMP CLI arguments (e.g. ``["--model_precision", "bf16"]``) and
    override values from ``configs_yml``.
    """
    arg_list = ["--configs_yml", configs_yml, "--inputs", inputs, "--gpu", gpu, *extra_args]
    # ROMP modules inspect sys.argv at import time (they append a default
    # --configs_yml when none is present), so replace it before importing them.
    sys.argv = [sys.argv[0]] + arg_list

    from romp.lib.config import parse_args, ConfigContext
    from romp.predict.video import Video_processor

    with ConfigContext(parse_args(arg_list)) as cfg:
        return Video_processor(args_set=cfg)
