from dataclasses import dataclass
import os


CODES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CODES_DIR)


def repo_path(*parts: str) -> str:
    return os.path.normpath(os.path.join(REPO_ROOT, *parts))


CODES_DATA_DIR = repo_path("codes", "data")
CODES_OUTPUTS_DIR = repo_path("codes", "outputs")
# 9,30,25,41,49,2,37,29,45,42,5,58

@dataclass(frozen=True)
class PathConfig:
    video_cam1: str = repo_path("codes", "inputs", "video_largo_blancos_nuevo.mp4")
    video_cam2: str = repo_path("codes", "inputs", "video_largo_negros_nuevo.mp4")
    video_output: str = repo_path("codes", "outputs", "Pruebas-output-final.mp4")
    reid_checkpoint: str = repo_path("codes", "models", "model.pth.tar-300")
    reid_model_name: str = "resnet50"
    player_detector_path: str = repo_path("codes", "models", "players.onnx")
    ball_detector_path: str = repo_path("codes", "models", "ball.onnx")


@dataclass(frozen=True)
class DetectionConfig:
    imgsz: int = 1792
    iou: float = 0.65
    conf_thres: float = 0.3
    sample_stride: int = 30


@dataclass(frozen=True)
class FusionConfig:
    pos_thr: float = 100.0
    reid_thr: float = 0.75
    strict_pos_thr: float = 80.0
    strict_reid_thr: float = 0.65
    use_strict: bool = False
    disable: bool = True


# Flags de depuración / experimentación
# - crosscam_snap reasigna GIDs por cercanía en cancha; puede ser inestable con calibración imperfecta.
ENABLE_CROSSCAM_SNAP: bool = False

# Pipeline offline crosscam:
# - "export": corre tracking por cámara y guarda tracklets + embeddings en RUN_DIR (no usa GIDs).
# - "render": corre tracking por cámara, carga OFFLINE_MAP_PATH y dibuja/exporta usando GID fijo.
PIPELINE_STAGE: str = "export"  # "export" | "render"
RUN_DIR: str = repo_path("runs", "default")
OFFLINE_MAP_PATH: str = os.path.join(RUN_DIR, "crosscam_map.json")

# Si True, además de RUN_DIR también escribe copias en `codes/data/` (compatibilidad).
ALSO_WRITE_CODES_DATA: bool = False

# Tracking monocámara (Paso 1): BoT-SORT/ByteTrack (Ultralytics)
# - Edita `codes/botsort.yaml` para tunear: track_buffer, track_high_thresh, track_low_thresh, match_thresh, etc.
ULTRALYTICS_TRACKER_YAML: str = os.path.join(CODES_DIR, "botsort_ultra_cut.yaml")
TRACKING_DET_CONF: float = 0.35  # subir conf para detectar menos
TRACK_MIN_BBOX_H: int = 28
TRACK_MIN_BBOX_AREA: int = 700
TRACK_MAX_Y_TOP: float = 0.75  # 0..1 relativo a altura de imagen
PLAYER_CLASS_ID: int = 1

# StableIDAssigner: cómo decidir si un ds_id "sigue vivo"
# - "tracker": usa los track_ids en memoria del tracker (tracked + lost). Recomendado para estabilidad.
# - "observed": solo los ids observados este frame (más agresivo; puede fragmentar si faltan detecciones).
STABLEID_ALIVE_MODE: str = "tracker"  # "tracker" | "observed"
STABLEID_DS_DEAD_AFTER: int = 3  # frames sin "alive" para bankear (0 = inmediato). Si usas "observed", sube a 15–30.
STABLEID_ENABLE: bool = False  # desactivar para usar ds_id directo

# Embeddings (extract_embedding)
EMB_PAD: float = 0.12
EMB_MIN_HW: int = 10  # mínimo lado del recorte para extraer embedding

# Dashboard (métricas baratas sin GT)
# - crosscam_pos_thr: distancia máxima (en coords de cancha) para considerar posiciones compatibles.
DASHBOARD_CROSSCAM_POS_THR: float = 80.0


@dataclass(frozen=True)
class TrackerConfig:
    reid_weight: float = 0.5
    pos_weight: float = 0.5
    reid_threshold: float = 0.7
    pos_threshold: float = 200.0
    pos_thr_growth: float = 6.0     # unidades de cancha por frame perdido
    pos_thr_max: float = 600.0      # techo del gate base
    max_age: int = 15
    emb_momentum: float = 0.2
    pos_alpha: float = 0.4
    motion_momentum: float = 0.6
    pos_gate_mult: float = 1.2
    max_lost: int = 15
    lost_reid_weight: float = 0.6
    lost_pos_weight: float = 0.4
    lost_pos_gate_mult: float = 1.3
    lost_reid_threshold: float = 0.7
    reuse_pool_max_age: int = 0
    reuse_pos_gate_mult: float = 3.5
    reuse_reid_threshold: float = 1.0


@dataclass(frozen=True)
class GlobalIDConfig:
    reid_weight: float = 0.5
    pos_weight: float = 0.5
    reid_thr: float = 0.75
    pos_thr: float = 200.0
    n_confirm: int = 5
    t_link: float = 0.55
    t_link_recapture: float = 0.65
    recapture_min_dt: int = 30  # frames sin ver el GID para usar el gate relajado
    t_steal: float = 0.25
    steal_k: int = 7
    emb_alpha: float = 0.2
    pos_alpha: float = 0.3
    pos_thr_growth: float = 5.0   # crecimiento lineal del gate posicional por frame perdido
    pos_thr_max: float = 350.0    # techo para el gate posicional dinámico
    reid_recapture_thr: float = 0.50  # reid-only para recaptura
    rid_bonus_factor: float = 0.30    # factor para favorecer matches con mismo RID previo
    rid_mismatch_penalty: float = 4.0  # penalización si el RID no coincide
    rid_strict_locked: bool = True     # si True, bloquea asignar un LOCKED a otro RID
    max_gid_factor: float = 1.5  # multiplicador de ROSTER_SIZE
    ttl_frames: int = 450


@dataclass(frozen=True)
class RosterConfig:
    size: int = 12
    n_init: int = 3
    lost_tol: int = 90
    alpha: float = 0.6
    pos_thr: float = 250.0
    t_link: float = 0.60
    t_steal: float = 0.10
    n_confirm: int = 2
    steal_k: int = 4
    proto_emb_alpha: float = 0.07
    proto_pos_alpha: float = 0.12


@dataclass(frozen=True)
class StabilityConfig:
    sticky_miss_tol_sec: float = 0.0
    stabilize_frames_sec: float = 0.0
    radar_smooth_alpha: float = 1.0


@dataclass(frozen=True)
class TeamClassifierConfig:
    use_umap: bool = True
    umap_components: int = 8
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.05
    kmeans_n_init: int = 10
    vote_window: int = 25
    lock_votes: int = 8
    tight_lock_p: float = 0.85
    low_conf_thr: float = 0.55
    debug_dir: str = "debug"
    sample_frames: int = 180
    per_frame_limit: int = 8
    imgsz: int = 1792
    iou: float = 0.7


@dataclass(frozen=True)
class ManualReviewConfig:
    output_dir: str = repo_path("codes", "outputs", "manual_team_review")
    confidence_thr: float = 0.01
    autoclose_ms: int = 500


@dataclass(frozen=True)
class ManualInputConfig:
    enable_switch_prompt: bool = False
    enable_reattach_prompt: bool =  False

# Ayudas de depuracion
# - DEBUG_APPROX_SECOND: segundo aproximado donde ocurre el evento a inspeccionar (None para desactivar)
# - DEBUG_FRAME_WINDOW: numero de frames a cada lado del frame objetivo para definir la ventana
# - DEBUG_RUN_WINDOW_ONLY: si True, solo procesa esa ventana (util para acelerar debug)
DEBUG_APPROX_SECOND = None
DEBUG_FRAME_WINDOW = 50
DEBUG_RUN_WINDOW_ONLY = False
DEBUG_VIZ_USE_AUTO_BOUNDS: bool = True
DEBUG_VIZ_WIDTH: int = 1600
DEBUG_VIZ_HEIGHT: int = 800
DEBUG_VIZ_PAD_FRAC: float = 0.10  # padding extra si se usan bounds auto
# Si quieres forzar bounds fijos, cambia USE_AUTO_BOUNDS a False y ajusta aquí:
DEBUG_VIZ_BOUNDS = None
DEBUG_VIZ_VIDEO_TARGETS = ["04", "09", "10"]

# Revisión manual-visual del pipeline
# - Genera artefactos en RUN_DIR/debug_viz y una propuesta editable en RUN_DIR/manual_overrides.json.
# - Si hay consola interactiva, pausa antes del render para que puedas editar manual_overrides.json.
PIPELINE_ENABLE_MANUAL_REVIEW: bool = True
PIPELINE_MANUAL_REVIEW_PAUSE: bool = True
PIPELINE_MANUAL_REVIEW_RERUN_CROSSCAM: bool = True
MANUAL_REVIEW_VIDEO_TARGETS = ["04", "05", "08", "09", "10"]
MANUAL_REVIEW_TEMPLATE_FILENAME: str = "manual_overrides.template.json"
MANUAL_REVIEW_FILENAME: str = "manual_overrides.json"
MANUAL_REVIEW_INSTRUCTIONS_FILENAME: str = "manual_review_instructions.txt"

# Offline crosscam (Paso 3)
CROSSCAM_MIN_TRACKLET_LEN_FRAMES: int = 48
CROSSCAM_MIN_OVERLAP_FRAMES: int = 24
CROSSCAM_SAMPLE_STEP: int = 2
CROSSCAM_POS_THR: float = 150.0
CROSSCAM_REID_THR: float = 0.35
CROSSCAM_MIN_EMB_RATIO: float = 0.25
CROSSCAM_INLIER_THR: float = 200.0
CROSSCAM_MIN_INLIER_PCT: float = 0.75
CROSSCAM_MIN_INLIER_FRAMES: int = 12
CROSSCAM_W_POS: float = 1.0
CROSSCAM_W_REID: float = 0.3
CROSSCAM_W_TEAM: float = 0.0  # set >0 si usas team confiable
CROSSCAM_ACCEPT_MARGIN: float = 0.15
CROSSCAM_WRITE_GLOBAL_TRACKS: bool = True
CROSSCAM_POS_ACCEPT_THR: float = 150.0  # umbral estricto para consistencia/aceptación
DASHBOARD_CROSSCAM_POS_THR: float = 150.0  # usa el mismo que aceptación para métricas
RENDER_USE_STITCHED: bool = True  # en render, usar CSV stitched + crosscam_map.json (sin retrack)

# Quality masking (frames buenos para crosscam)
QC_CONF_MIN: float = 0.25
QC_AREA_MIN: float = 300.0
QC_BORDER_MIN: float = 0.0  # píxeles; 0 para desactivar
QC_USE_AREA: bool = True
QC_USE_BORDER: bool = False
QC_MIN_GOOD_FRAMES: int = 12
QC_MIN_GOOD_PCT: float = 0.5

# Balón (detección simple por cámara)
BALL_MIN_CONF: float = 0.20
BALL_MIN_AREA: float = 20.0
BALL_MAX_AREA: float = 5000.0
BALL_BORDER_MIN: float = 5.0  # píxeles al borde; descartar si es menor
BALL_MAX_JUMP_PITCH: float = 500.0  # distancia máxima permitida en cancha por frame; descarta outliers
BALL_MAX_JUMP_PITCH_STRICT: float = 300.0  # tope más estricto para saltos entre frames consecutivos
BALL_MAX_JUMP_LONG: float = 2000.0  # techo duro para saltos tras gaps largos
BALL_NEIGHBOR_JUMP_THR: float = 300.0  # si dist a prev y next supera esto, marca outlier

# Team gating/penalty
TEAM_MODE: str = "penalty"  # "gate" | "penalty"
TEAM_MIN_CONF: float = 0.7
TEAM_PENALTY: float = 1.0
TEAM_MIN_FRAMES: int = 12
TEAM_UNKNOWN_POLICY: str = "allow"  # "allow" | "block"

# Auto-alineación (RANSAC) cam2->cam1
CROSSCAM_ENABLE_ALIGN: bool = True
ALIGN_MODEL: str = "similarity"  # "similarity" | "affine"
ALIGN_RANSAC_ITERS: int = 2000
ALIGN_INLIER_THR: float = 200.0
ALIGN_MIN_INLIERS: int = 30
ALIGN_MIN_PAIRS: int = 2
ALIGN_SAMPLE_STEP: int = 3
ALIGN_SAVE_PATH: str = "crosscam_align.json"

# Handoff / asociación sin solape fuerte (complemento)
CROSSCAM_HO_ENABLE: bool = True
CROSSCAM_HO_MAX_GAP_FRAMES: int = 60
CROSSCAM_HO_POS_THR: float = 220.0
CROSSCAM_HO_REID_THR: float = 0.5
CROSSCAM_HO_W_POS: float = 1.0
CROSSCAM_HO_W_REID: float = 0.3
CROSSCAM_HO_ENDPOINT_FRAMES: int = 5  # frames al inicio/fin para promedio robusto

# Fusión best-view (global)
FUSE_W_CONF: float = 0.4
FUSE_W_AREA: float = 0.4
FUSE_W_BORDER: float = 0.2
FUSE_CONF_MIN: float = 0.25
FUSE_BORDER_MIN_PX: float = 10.0
FUSE_USE_ALIGN_FOR_CAM2: bool = True
FUSE_SWITCH_MARGIN: float = 0.15
FUSE_SWITCH_MIN_HOLD_FRAMES: int = 8
FUSE_MAX_PRED_GAP_FRAMES: int = 24
FUSE_MAX_SPEED: float = 500.0
FUSE_WRITE_FUSE_REPORT: bool = True

# Voting por ventanas (matching crosscam por consenso)
VOTE_ENABLE: bool = False
VOTE_WINDOW_FRAMES: int = 240
VOTE_STEP_FRAMES: int = 120
VOTE_MIN_WINS: int = 2
VOTE_MARGIN_WINS: int = 1
VOTE_REQUIRE_MUTUAL: bool = True
VOTE_MAX_MEAN_COST: float = 1.2
VOTE_MIN_INLIER_PCT: float = 0.75
VOTE_USE_TEAM: bool = True
VOTE_USE_QC: bool = True

# --- Anti-teleport en proyección a pitch (01_export_radar_all) ---
POS_SMOOTH_ENABLE: bool = False
POS_EMA_BETA: float = 0.20          # 0.10-0.25 recomendado
POS_SPIKE_PITCH_THR: float = 150.0  # tu umbral tipo-1
POS_SPIKE_IMG_THR: float = 15.0     # tu umbral tipo-1
POS_STATE_TTL_FRAMES: int = 240     # limpiar estados viejos (~10s a 24fps)
# Override opcional para radar hold (frames)
STICKY_MISS_TOL_FRAMES: int = 0
RAW_POS_MODE: bool = True  # si True, no filtrar ni limpiar posiciones en render
RENDER_MAPPED_ONLY: bool = True  # en render, no dibujar unmapped (sin gid)

# Anchor para proyección a cancha
PITCH_ANCHOR_MODE: str = "pseudo_foot"  # "bc" | "center" | "pseudo_foot"
PSEUDO_FOOT_FRAC: float = 0.08  # fracción de altura para subir el punto desde el pie

# =========================
# DEMO CLEAN (offline_stitch)
# =========================
STITCH_CLEAN_MIN_OBS: int = 12
STITCH_CLEAN_MIN_COVERAGE: float = 0.50
STITCH_CLEAN_BOUNDS = (-200.0, 4200.0, -200.0, 2200.0)
STITCH_DISABLE_CLEAN: bool = True  # si True, salta DEMO_CLEAN y glitch fix

# --- Offline glitch removal (A-B-A) ---
GLITCH_ENABLE: bool = True
GLITCH_PITCH_JUMP_THR: float = 180.0
GLITCH_RETURN_THR: float = 80.0
GLITCH_FIX_MODE: str = "hold"  # "interp" | "hold"
GLITCH_MAX_SKIP: int = 1
GLITCH_VEL_HOLD_THR: float = 220.0  # si delta/frame supera esto, fija al frame previo

# Clip global para normalizar ventana de procesamiento
CLIP_ENABLE: bool = True
CLIP_START_FRAME: int = 0
CLIP_END_FRAME: int = 2549  # si 0, se ignora; si >0, último frame incluido


PATHS = PathConfig()
DETECTION_CONFIG = DetectionConfig()
FUSION_CONFIG = FusionConfig()
TRACKER_CONFIG = TrackerConfig()
GLOBAL_ID_CONFIG = GlobalIDConfig()
ROSTER_CONFIG = RosterConfig()
STABILITY_CONFIG = StabilityConfig()
TEAM_CLASSIFIER_CONFIG = TeamClassifierConfig()
MANUAL_REVIEW_CONFIG = ManualReviewConfig()
MANUAL_INPUT_CONFIG = ManualInputConfig()

# DeepSORT (tracking por cámara usando embeddings externos)
DEEPSORT_CFG = dict(
    max_age=45,               # aguanta ~1.5s a 30fps sin detección
    n_init=3,                 # confirma track tras 3 matches
    max_iou_distance=0.7,
    max_cosine_distance=0.25, # si tus embeddings son buenos, baja esto
    nn_budget=100,
)
