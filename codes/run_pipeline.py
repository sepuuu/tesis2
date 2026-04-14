"""
Orquestador simple para correr el pipeline completo sin pasar parámetros:
1) export (main_dualcam con PIPELINE_STAGE=export)
2) offline_stitch
3) offline_crosscam
4) manual_review (visual + manual_overrides.json)
5) offline_crosscam final (reaplica overrides manuales)
6) render final (main_dualcam con PIPELINE_STAGE=render)

Ejecuta:
    python codes/run_pipeline.py
Lee RUN_DIR y demás ajustes desde config.py.
"""

import os
import sys
import importlib
import time

import config
import main_dualcam
import offline_stitch
import offline_crosscam
import manual_review
 #id switch 41,1,  35,47,  50,  57,  72
def _run_main(stage: str) -> None:
    # Ajusta PIPELINE_STAGE y ruta del mapa para render
    config.PIPELINE_STAGE = stage
    if stage == "render":
        # Asegura ruta por defecto al map generado
        config.OFFLINE_MAP_PATH = os.path.join(config.RUN_DIR, "crosscam_map.json")
        # Si existe el flag de render offline, actívalo
        try:
            config.RENDER_USE_STITCHED = True
        except Exception:
            pass
    print(f"[RUN] main_dualcam | PIPELINE_STAGE={config.PIPELINE_STAGE} | RUN_DIR={config.RUN_DIR}")
    t0 = time.time()
    main_dualcam.process_dual_camera(
        main_dualcam.VIDEO_PATH_CAM1,
        main_dualcam.VIDEO_PATH_CAM2,
        main_dualcam.view_transformer1,
        main_dualcam.view_transformer2,
    )
    dt = time.time() - t0
    print(f"[DONE] main_dualcam stage={stage} | dur={_fmt_dur(dt)} ({dt:.2f}s)")


def _run_offline_stitch() -> None:
    argv_bak = sys.argv[:]
    sys.argv = ["offline_stitch.py"]  # sin args, usa RUN_DIR de config
    print(f"[RUN] offline_stitch | RUN_DIR={config.RUN_DIR}")
    t0 = time.time()
    offline_stitch.main()
    sys.argv = argv_bak
    dt = time.time() - t0
    print(f"[DONE] offline_stitch | dur={_fmt_dur(dt)} ({dt:.2f}s)")


def _run_offline_crosscam() -> None:
    argv_bak = sys.argv[:]
    sys.argv = ["offline_crosscam.py"]  # sin args, usa RUN_DIR de config
    print(f"[RUN] offline_crosscam | RUN_DIR={config.RUN_DIR}")
    t0 = time.time()
    offline_crosscam.main()
    sys.argv = argv_bak
    dt = time.time() - t0
    print(f"[DONE] offline_crosscam | dur={_fmt_dur(dt)} ({dt:.2f}s)")


def _run_manual_review() -> None:
    argv_bak = sys.argv[:]
    sys.argv = ["manual_review.py"]  # sin args, usa RUN_DIR de config
    print(f"[RUN] manual_review | RUN_DIR={config.RUN_DIR}")
    t0 = time.time()
    manual_review.main()
    sys.argv = argv_bak
    dt = time.time() - t0
    print(f"[DONE] manual_review | dur={_fmt_dur(dt)} ({dt:.2f}s)")


def _fmt_dur(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    run_dir = getattr(config, "RUN_DIR", "runs/default")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[PIPELINE] START | RUN_DIR={run_dir}")
    t0_pipeline = time.time()

    _run_main("export")
    _run_offline_stitch()
    _run_offline_crosscam()
    if bool(getattr(config, "PIPELINE_ENABLE_MANUAL_REVIEW", True)):
        _run_manual_review()
        if bool(getattr(config, "PIPELINE_MANUAL_REVIEW_RERUN_CROSSCAM", True)):
            _run_offline_crosscam()
    _run_main("render")

    dt = time.time() - t0_pipeline
    print(f"[PIPELINE] DONE | dur={_fmt_dur(dt)} ({dt:.2f}s)")


if __name__ == "__main__":
    main()
