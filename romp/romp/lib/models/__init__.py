import sys, os
lib_dir = os.path.dirname(__file__)
root_dir = os.path.join(lib_dir.replace(os.path.basename(lib_dir),''))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def build_model():
    from .build import build_model as _build_model
    return _build_model()
