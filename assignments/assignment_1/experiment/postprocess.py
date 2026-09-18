import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from A1_template_2026 import load_targets, show_body

from .settings import RESULTS_DIR, CAM_FOVY


def render_targets(cam_fovy: float = CAM_FOVY) -> None:
    """
    Save frame of every target body
    """
    for i, graph in enumerate(load_targets()):
        file_name = str(RESULTS_DIR / f"target_{i:02d}")
        show_body(graph, mode="frame", file_name=file_name, cam_fovy=cam_fovy)
        print(f"saved {file_name}.png")


if __name__ == "__main__":
    render_targets()