"""Installation check: python -m cavesim.check"""
import sys
from pathlib import Path


def main():
    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    import numpy as np
    import pybullet
    import scipy
    import skimage
    import trimesh
    import websockets
    print(f"numpy {np.__version__}, scipy {scipy.__version__}, scikit-image {skimage.__version__}, "
          f"trimesh {trimesh.__version__}, websockets {websockets.__version__}, pybullet API {pybullet.getAPIVersion()}")

    a = np.random.default_rng(0).random((500, 500))
    assert np.allclose((a @ np.eye(500)), a), "matrix multiply is broken (BLAS)"
    print("numpy BLAS ok")

    from .api import Agent
    from .config import SimConfig
    from .sim import run_mission
    maps = Path(__file__).resolve().parent.parent / "maps"
    score = run_mission(maps / "small_2", Agent, n_drones=2, cfg=SimConfig(mission_time_s=20))
    assert not score["crashed"], "hovering drones crashed: physics is broken"
    print("20 s hover test on small_2 ok (PyBullet collision + LiDAR)")
    print("\nOK. Next: python -m cavesim.run")


if __name__ == "__main__":
    main()
