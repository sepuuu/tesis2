"""
Etapa de revision manual-visual del pipeline.

- Genera artefactos visuales reutilizando debug_viz.py
- Escribe una propuesta editable de manual_overrides.json a partir del crosscam_map actual
- Opcionalmente pausa el flujo para que el usuario revise y edite los overrides
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import config
import debug_viz


def _load_crosscam_map(run_dir: str) -> Dict[str, Dict[int, int]]:
    map_path = str(getattr(config, "OFFLINE_MAP_PATH", os.path.join(run_dir, "crosscam_map.json")))
    if not os.path.isabs(map_path):
        map_path = os.path.join(run_dir, map_path)
    if not os.path.exists(map_path):
        return {"cam1": {}, "cam2": {}}
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "cam1": {int(k): int(v) for k, v in data.get("cam1", {}).items()},
        "cam2": {int(k): int(v) for k, v in data.get("cam2", {}).items()},
    }


def _build_override_template(gid_map: Dict[str, Dict[int, int]]) -> dict:
    players: Dict[str, dict] = {}
    for cam_key in ("cam1", "cam2"):
        for sid, gid in sorted(gid_map.get(cam_key, {}).items(), key=lambda item: (item[1], item[0])):
            entry = players.setdefault(str(int(gid)), {"cam1": [], "cam2": []})
            entry[cam_key].append(int(sid))
    for gid_key in players:
        players[gid_key]["cam1"] = sorted(set(int(v) for v in players[gid_key]["cam1"]))
        players[gid_key]["cam2"] = sorted(set(int(v) for v in players[gid_key]["cam2"]))
    return {
        "players": players,
        "special_segments": {"cam1": [], "cam2": []},
    }


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _ensure_override_files(run_dir: str) -> tuple[str, str]:
    gid_map = _load_crosscam_map(run_dir)
    template = _build_override_template(gid_map)

    template_name = str(getattr(config, "MANUAL_REVIEW_TEMPLATE_FILENAME", "manual_overrides.template.json"))
    review_name = str(getattr(config, "MANUAL_REVIEW_FILENAME", "manual_overrides.json"))

    template_path = os.path.join(run_dir, template_name)
    review_path = os.path.join(run_dir, review_name)

    _write_json(template_path, template)
    if not os.path.exists(review_path):
        _write_json(review_path, template)
        print(f"[WRITE] {review_path} (creado desde crosscam_map.json)")
    else:
        print(f"[KEEP] {review_path} (ya existe, no se sobreescribe)")
    print(f"[WRITE] {template_path}")
    return review_path, template_path


def _write_instructions(run_dir: str, review_path: str, template_path: str) -> str:
    instructions_name = str(
        getattr(config, "MANUAL_REVIEW_INSTRUCTIONS_FILENAME", "manual_review_instructions.txt")
    )
    debug_dir = os.path.join(run_dir, "debug_viz")
    out_path = os.path.join(run_dir, instructions_name)
    lines = [
        "REVISION MANUAL-VISUAL DEL PIPELINE",
        "",
        f"1. Revisa los artefactos visuales en: {debug_dir}",
        "   Los mas utiles suelen ser:",
        "   - 04_render_radar_all_with_gid.mp4",
        "   - 05_crosscam_pair_error_lines.mp4",
        "   - 08_combined_cams_gid.mp4",
        "   - 09_combined_teams_gid.mp4",
        "   - 10_voronoi_radar_teams.mp4",
        "",
        f"2. Edita el archivo: {review_path}",
        f"   Si quieres reiniciar desde la propuesta automatica, usa: {template_path}",
        "",
        "3. Formato esperado:",
        '{',
        '  "players": {',
        '    "1": { "cam1": [1, 38], "cam2": [4] },',
        '    "2": { "cam1": [5], "cam2": [7, 9] }',
        "  },",
        '  "special_segments": {',
        '    "cam1": [ { "id": 38, "gid": 1, "start": 1200, "end": 1350 } ],',
        '    "cam2": []',
        "  }",
        "}",
        "",
        "Notas:",
        "- La clave de players es el GID final que quieres imponer.",
        "- En cam1/cam2 lista los stable_id stitched que pertenecen a ese jugador.",
        "- special_segments es opcional y hoy sirve sobre todo para debug_viz.",
        "",
        "Cuando termines de editar, continua el pipeline para recalcular crosscam y render final.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[WRITE] {out_path}")
    return out_path


def _run_debug_viz() -> None:
    prev_targets = getattr(config, "DEBUG_VIZ_VIDEO_TARGETS", None)
    config.DEBUG_VIZ_VIDEO_TARGETS = list(
        getattr(config, "MANUAL_REVIEW_VIDEO_TARGETS", prev_targets or ["04", "05", "08", "09", "10"])
    )
    try:
        debug_viz.main()
    finally:
        if prev_targets is None:
            try:
                delattr(config, "DEBUG_VIZ_VIDEO_TARGETS")
            except Exception:
                pass
        else:
            config.DEBUG_VIZ_VIDEO_TARGETS = prev_targets


def _pause_for_review(run_dir: str, review_path: str, instructions_path: str, pause: bool) -> None:
    if not pause:
        return
    if not getattr(sys.stdin, "isatty", lambda: False)():
        print("[REVIEW] stdin no interactivo; se omite la pausa manual.")
        return
    print("[REVIEW] Revisa los videos y edita manual_overrides.json antes del render final.")
    print(f"[REVIEW] RUN_DIR: {run_dir}")
    print(f"[REVIEW] Instrucciones: {instructions_path}")
    print(f"[REVIEW] Overrides: {review_path}")
    input("[REVIEW] Presiona ENTER para continuar con offline_crosscam final y render... ")


def run_review(*, run_dir: str, pause: bool) -> None:
    os.makedirs(run_dir, exist_ok=True)
    review_path, template_path = _ensure_override_files(run_dir)
    instructions_path = _write_instructions(run_dir, review_path, template_path)
    t0 = time.time()
    _run_debug_viz()
    dt = time.time() - t0
    print(f"[DONE] manual_review visuals | dur={dt:.2f}s")
    _pause_for_review(run_dir, review_path, instructions_path, pause=pause)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera revision manual-visual y prepara manual_overrides.json.")
    parser.add_argument("--run-dir", default=str(getattr(config, "RUN_DIR", "runs/default")))
    parser.add_argument("--no-pause", action="store_true", help="No esperar ENTER al final.")
    args = parser.parse_args()
    pause = not args.no_pause and bool(getattr(config, "PIPELINE_MANUAL_REVIEW_PAUSE", True))
    run_review(run_dir=str(args.run_dir), pause=pause)


if __name__ == "__main__":
    main()
