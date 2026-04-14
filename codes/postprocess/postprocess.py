import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import hypot

# FunciÃ³n de limpieza de la trayectoria de la pelota
def clean_ball_trajectory(file_path, output_path, distance_threshold=900):
    print("Iniciando limpieza de posiciones de la pelota...")
    df = pd.read_excel(file_path)

    if 'Ball X' in df.columns and 'Ball Y' in df.columns:
        df = df.dropna(subset=['Ball X', 'Ball Y'])
    else:
        raise ValueError("Las columnas 'Ball X' o 'Ball Y' no estÃ¡n en el archivo Excel.")

    df.reset_index(drop=True, inplace=True)

    ball_x = df['Ball X'].values
    ball_y = df['Ball Y'].values

    filtered_indices = [0]
    for i in range(1, len(ball_x)):
        if np.isnan(ball_x[i]) or np.isnan(ball_y[i]):
            continue
        distance = np.sqrt((ball_x[i] - ball_x[filtered_indices[-1]])**2 +
                           (ball_y[i] - ball_y[filtered_indices[-1]])**2)
        if distance <= distance_threshold:
            filtered_indices.append(i)

    cleaned_df = df.iloc[filtered_indices].reset_index(drop=True)

    cleaned_df.to_excel(output_path, index=False)
    print(f"Limpieza de posiciones completa. Archivo guardado en: {output_path}")

    return cleaned_df

# Funciones modulares para procesar posesiÃ³n y pases
def calculate_possession(df, radius):
    print("Calculando posesiÃ³n inicial...")
    df['Distance_to_Ball'] = np.sqrt((df['Pos X'] - df['Ball X'])**2 + (df['Pos Y'] - df['Ball Y'])**2)
    df['Initial_Possession'] = df['Distance_to_Ball'] <= radius
    print("CÃ¡lculo de posesiÃ³n inicial completado.")
    return df

def validate_persistence(df, frame_threshold):
    print("Validando persistencia de posesiÃ³n...")
    df['Validated_Possession'] = False
    grouped = df.groupby('Id')
    for player_id, group in grouped:
        possession_frames = group['Initial_Possession'].rolling(window=int(frame_threshold), min_periods=1).sum()
        df.loc[group.index, 'Validated_Possession'] = possession_frames >= frame_threshold
    print("ValidaciÃ³n de persistencia completada.")
    return df

def resolve_disputes(df, dispute_frames):
    print("Resolviendo disputas de posesiÃ³n...")
    df['Unique_Possession'] = False
    grouped = df.groupby('Id')
    for frame, frame_group in df.groupby('Frame'):
        in_possession = frame_group[frame_group['Validated_Possession']]
        if len(in_possession) == 1:
            df.loc[in_possession.index, 'Unique_Possession'] = True
        elif len(in_possession) > 1:
            for player_id in in_possession['Id']:
                player_group = grouped.get_group(player_id)
                if (player_group['Validated_Possession'].rolling(window=int(dispute_frames), min_periods=1).sum() >= dispute_frames).any():
                    df.loc[in_possession[in_possession['Id'] == player_id].index, 'Unique_Possession'] = True
    print("ResoluciÃ³n de disputas completada.")
    return df

def generate_possession_excel(df, output_path):
    print("Generando archivo de posesiÃ³n...")
    df['Posesion'] = df['Unique_Possession']
    df[['Frame', 'Id', 'Pos X', 'Pos Y', 'Ball X', 'Ball Y', 'Team', 'Posesion']].to_excel(output_path, index=False)
    print(f"Archivo de posesiÃ³n generado en: {output_path}")

def detect_passes(df):
    print("Detectando pases...")
    # Usar posesiÃ³n Ãºnica por frame
    if 'Unique_Possession' not in df.columns:
        df['Unique_Possession'] = df.get('Posesion', False)

    frames = sorted(df['Frame'].unique())
    sequence = []  # (frame, id, team, bx, by)
    for f in frames:
        g = df[(df['Frame'] == f) & (df['Unique_Possession'] == True)]
        if len(g) == 0:
            continue
        if len(g) > 1 and ('Ball X' in g.columns and 'Ball Y' in g.columns):
            bx, by = g.iloc[0]['Ball X'], g.iloc[0]['Ball Y']
            if np.isfinite(bx) and np.isfinite(by):
                d = np.hypot(g['Pos X'] - bx, g['Pos Y'] - by)
                g = g.iloc[[int(np.argmin(d.values))]]
            else:
                g = g.iloc[[0]]
        else:
            g = g.iloc[[0]]
        r = g.iloc[0]
        sequence.append((int(r['Frame']), int(r['Id']), r['Team'], float(r['Ball X']), float(r['Ball Y'])))

    passes_data = []
    for i in range(1, len(sequence)):
        f_prev, id_prev, team_prev, bx_prev, by_prev = sequence[i-1]
        f_curr, id_curr, team_curr, bx_curr, by_curr = sequence[i]
        if id_prev != id_curr:
            passes_data.append({
                'id_emisor': id_prev,
                'id_receptor': id_curr,
                'team_emisor': team_prev,
                'team_receptor': team_curr,
                'Frame': f_prev,
                'frame_end': f_curr,
                'X_ball_inicio': bx_prev,
                'Y_ball_inicio': by_prev,
                'X_ball_Final': bx_curr,
                'Y_ball_final': by_curr
            })

    print("DetecciÃ³n de pases completada.")
    cols = ['id_emisor','id_receptor','team_emisor','team_receptor','Frame','frame_end','X_ball_inicio','Y_ball_inicio','X_ball_Final','Y_ball_final']
    return pd.DataFrame(passes_data, columns=cols)

def assign_unknown_teams(df, passes_df):
    print("Asignando equipos a jugadores UNKNOWN...")
    team_mapping = df[['Id', 'Team']].drop_duplicates().sort_values(by=['Id', 'Team'])
    team_mapping = team_mapping[team_mapping['Team'] != 'UNKNOWN']
    team_mapping = team_mapping.drop_duplicates(subset=['Id'], keep='last')
    team_dict = dict(zip(team_mapping['Id'], team_mapping['Team']))
    passes_df['team_emisor'] = passes_df['id_emisor'].map(team_dict).fillna(passes_df['team_emisor'])
    passes_df['team_receptor'] = passes_df['id_receptor'].map(team_dict).fillna(passes_df['team_receptor'])
    print("AsignaciÃ³n de equipos completada.")
    return passes_df

# ===================== Dificultad de pases (aprox. Voronoi) ===================== #
def _point_segment_distance(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx*wx + vy*wy
    if c1 <= 0:
        return hypot(px - x1, py - y1)
    c2 = vx*vx + vy*vy
    if c2 <= 1e-9:
        return hypot(px - x1, py - y1)
    t = c1 / c2
    if t >= 1:
        return hypot(px - x2, py - y2)
    projx, projy = x1 + t*vx, y1 + t*vy
    return hypot(px - projx, py - projy)

def _nearest_distance(points_xy, qx, qy):
    if len(points_xy) == 0:
        return float('inf')
    dx = points_xy[:, 0] - qx
    dy = points_xy[:, 1] - qy
    d = np.hypot(dx, dy)
    return float(np.min(d))

def calculate_pass_difficulty_v2(cleaned_df: pd.DataFrame, passes_df: pd.DataFrame) -> pd.DataFrame:
    if passes_df.empty:
        return passes_df
    df = cleaned_df.copy()
    for col in ['Frame', 'Id', 'Pos X', 'Pos Y', 'Ball X', 'Ball Y', 'Team']:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida en cleaned_df: {col}")
    LEN_NORM = 2000.0
    SPACE_NORM = 300.0
    meta = {
        'length': [],
        'min_opp_to_line': [],
        'receiver_clearance': [],
        'opp_dominance_ratio': [],
        'difficulty': [],
    }
    for idx, row in passes_df.iterrows():
        id_em = row.get('id_emisor'); id_rc = row.get('id_receptor')
        team_em = row.get('team_emisor')
        f0 = int(row.get('Frame', 0))
        x1 = float(row.get('X_ball_inicio', np.nan))
        y1 = float(row.get('Y_ball_inicio', np.nan))
        x2 = float(row.get('X_ball_Final', np.nan))
        y2 = float(row.get('Y_ball_final', np.nan))
        length = hypot(x2 - x1, y2 - y1) if (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)) else 0.0
        start_players = df[df['Frame'] == f0]
        opp_start = start_players[start_players['Team'] != team_em]
        opp_pts_start = opp_start[['Pos X', 'Pos Y']].to_numpy(dtype=np.float32) if len(opp_start) else np.zeros((0,2), np.float32)
        if len(opp_pts_start):
            dists = [ _point_segment_distance(px, py, x1, y1, x2, y2) for (px, py) in opp_pts_start ]
            min_opp_to_line = float(np.min(dists))
        else:
            min_opp_to_line = float('inf')
        f1 = int(row.get('frame_end', f0 + 1))
        end_players = df[df['Frame'] == f1]
        opp_end = end_players[end_players['Team'] != team_em]
        rec_row = end_players[end_players['Id'] == id_rc]
        if len(rec_row) and len(opp_end):
            rx = float(rec_row.iloc[0]['Pos X']); ry = float(rec_row.iloc[0]['Pos Y'])
            receiver_clearance = _nearest_distance(opp_end[['Pos X','Pos Y']].to_numpy(dtype=np.float32), rx, ry)
        else:
            receiver_clearance = float('inf')
        team_em_all = start_players[start_players['Team'] == team_em][['Pos X','Pos Y']].to_numpy(dtype=np.float32)
        opp_all = start_players[start_players['Team'] != team_em][['Pos X','Pos Y']].to_numpy(dtype=np.float32)
        dom_ratio = 0.0
        if len(team_em_all) and len(opp_all) and np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2):
            K = 15
            cnt_opp = 0
            for k in range(1, K+1):
                t = k / (K + 1.0)
                sx = x1 + t*(x2 - x1)
                sy = y1 + t*(y2 - y1)
                d_team = _nearest_distance(team_em_all, sx, sy)
                d_opp  = _nearest_distance(opp_all, sx, sy)
                if d_opp < d_team:
                    cnt_opp += 1
            dom_ratio = cnt_opp / float(K)
        length_norm = min(length / LEN_NORM, 1.0)
        clear_penalty = 1.0 - min((min_opp_to_line / SPACE_NORM) if np.isfinite(min_opp_to_line) else 1.0, 1.0)
        recv_penalty  = 1.0 - min((receiver_clearance / SPACE_NORM) if np.isfinite(receiver_clearance) else 1.0, 1.0)
        difficulty = 0.35*length_norm + 0.25*clear_penalty + 0.25*recv_penalty + 0.15*dom_ratio
        meta['length'].append(length)
        meta['min_opp_to_line'].append(min_opp_to_line)
        meta['receiver_clearance'].append(receiver_clearance)
        meta['opp_dominance_ratio'].append(dom_ratio)
        meta['difficulty'].append(float(np.clip(difficulty, 0.0, 1.0)))
    for k, v in meta.items():
        passes_df[k] = v
    return passes_df

def generate_pass_maps(cleaned_df, passes_df, output_dir='codes/outputs'):
    os.makedirs(output_dir, exist_ok=True)
    pos_mean = cleaned_df.groupby('Id')[['Pos X','Pos Y']].mean().rename(columns={'Pos X':'mx','Pos Y':'my'})
    teams = cleaned_df.groupby('Id')['Team'].agg(lambda s: s.replace('UNKNOWN', np.nan).dropna().iloc[-1] if len(s.replace('UNKNOWN', np.nan).dropna())>0 else 'UNKNOWN')
    per_player = pos_mean.join(teams)
    given_counts = passes_df.groupby('id_emisor').size().rename('passes_given') if not passes_df.empty else pd.Series(dtype=int)
    given_difficulty = passes_df.groupby('id_emisor')['difficulty'].sum().rename('sum_difficulty') if ('difficulty' in passes_df.columns and not passes_df.empty) else pd.Series(dtype=float)
    data = per_player.join(given_counts, how='left').join(given_difficulty, how='left').fillna({'passes_given':0,'sum_difficulty':0.0})
    edge_counts = passes_df.groupby(['id_emisor','id_receptor']).size() if not passes_df.empty else pd.Series(dtype=int)
    edge_diffs  = passes_df.groupby(['id_emisor','id_receptor'])['difficulty'].sum() if ('difficulty' in passes_df.columns and not passes_df.empty) else pd.Series(dtype=float)
    def _team_color(team):
        return '#000000' if team == 'equipo_negro' else ('#FFFFFF' if team == 'equipo_blanco' else '#888888')
    from matplotlib.patches import FancyArrowPatch
    # Count-based graph with edges
    fig, ax = plt.subplots(figsize=(8,6))
    for tid, row in data.iterrows():
        ax.scatter(row['mx'], row['my'], s=20 + 8*row['passes_given'], c=_team_color(row['Team']), edgecolors='k', zorder=3)
    if not edge_counts.empty:
        max_c = float(edge_counts.max()) if len(edge_counts)>0 else 1.0
        for (em, rc), c in edge_counts.items():
            if (em in per_player.index) and (rc in per_player.index):
                x1, y1 = per_player.loc[em, ['mx','my']]
                x2, y2 = per_player.loc[rc, ['mx','my']]
                lw = 0.5 + 3.5 * (c / max_c)
                arrow = FancyArrowPatch((x1,y1), (x2,y2), arrowstyle='->', mutation_scale=10, linewidth=lw, color='#444444', alpha=0.45, zorder=2)
                ax.add_patch(arrow)
    ax.set_title('Grafo de pases (grosor = cantidad)')
    ax.invert_yaxis(); ax.set_xlabel('X'); ax.set_ylabel('Y')
    path1 = os.path.join(output_dir, 'pass_map_count.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight'); plt.close(fig)
    # Difficulty-based graph with edges
    fig, ax = plt.subplots(figsize=(8,6))
    for tid, row in data.iterrows():
        ax.scatter(row['mx'], row['my'], s=20 + 120*row['sum_difficulty'], c=_team_color(row['Team']), edgecolors='k', zorder=3)
    if not edge_diffs.empty:
        max_d = float(edge_diffs.max()) if len(edge_diffs)>0 else 1.0
        for (em, rc), dsum in edge_diffs.items():
            if (em in per_player.index) and (rc in per_player.index):
                x1, y1 = per_player.loc[em, ['mx','my']]
                x2, y2 = per_player.loc[rc, ['mx','my']]
                lw = 0.5 + 3.5 * (dsum / max_d)
                arrow = FancyArrowPatch((x1,y1), (x2,y2), arrowstyle='->', mutation_scale=10, linewidth=lw, color='#8844cc', alpha=0.45, zorder=2)
                ax.add_patch(arrow)
    ax.set_title('Grafo de pases (grosor = suma dificultad)')
    ax.invert_yaxis(); ax.set_xlabel('X'); ax.set_ylabel('Y')
    path2 = os.path.join(output_dir, 'pass_map_difficulty.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight'); plt.close(fig)
    return path1, path2

def generate_summary_metrics(cleaned_df, passes_df, out_path='codes/data/metrics_summary.xlsx'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if 'Unique_Possession' not in cleaned_df.columns:
        cleaned_df['Unique_Possession'] = cleaned_df.get('Posesion', False)
    poss_frames = cleaned_df[cleaned_df['Unique_Possession'] == True]
    poss_one = poss_frames.sort_values(['Frame','Id']).groupby('Frame').head(1)
    team_poss = poss_one['Team'].value_counts().rename_axis('Team').to_frame('Frames')
    total_poss_frames = team_poss['Frames'].sum() if len(team_poss) else 0
    team_poss['Share'] = team_poss['Frames'] / total_poss_frames if total_poss_frames > 0 else 0
    if passes_df is None or passes_df.empty:
        passes_df = pd.DataFrame(columns=['id_emisor','id_receptor','team_emisor','team_receptor','difficulty'])
    per_player_given = passes_df.groupby(['id_emisor']).size().rename('passes_given')
    per_player_recv  = passes_df.groupby(['id_receptor']).size().rename('passes_received')
    per_player_diff  = passes_df.groupby(['id_emisor'])['difficulty'].agg(['mean','sum']).rename(columns={'mean':'avg_difficulty','sum':'sum_difficulty'}) if 'difficulty' in passes_df.columns and not passes_df.empty else pd.DataFrame(columns=['avg_difficulty','sum_difficulty'])
    player_team_map = cleaned_df.groupby('Id')['Team'].agg(lambda s: s.replace('UNKNOWN', np.nan).dropna().iloc[-1] if len(s.replace('UNKNOWN', np.nan).dropna())>0 else 'UNKNOWN')
    player_summary = pd.concat([per_player_given, per_player_recv, per_player_diff], axis=1).fillna(0)
    player_summary['Team'] = player_team_map
    top_passers = player_summary.sort_values(['Team','passes_given'], ascending=[True, False]).groupby('Team').head(1)
    player_poss_frames = poss_frames.groupby('Id').size().rename('possession_frames')
    player_poss_summary = pd.concat([player_poss_frames, player_team_map], axis=1).rename(columns={'Team':'Team'})
    top_possession = player_poss_summary.sort_values('possession_frames', ascending=False).groupby('Team').head(1)
    try:
        writer = pd.ExcelWriter(out_path)
    except Exception:
        writer = pd.ExcelWriter(out_path, engine='openpyxl')
    with writer as writer:
        team_poss.to_excel(writer, sheet_name='team_possession')
        player_summary.reset_index().rename(columns={'index':'Id'}).to_excel(writer, sheet_name='player_pass_stats', index=False)
        top_passers.reset_index().rename(columns={'index':'Id'}).to_excel(writer, sheet_name='top_passers', index=False)
        top_possession.reset_index().rename(columns={'index':'Id'}).to_excel(writer, sheet_name='top_possession', index=False)
    return out_path
def calculate_pass_difficulty(cleaned_df: pd.DataFrame, passes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada pase, calcula mÃ©tricas y una dificultad aproximada basada en:
      - Longitud del pase
      - Espacio despejado (mÃ­nima distancia de oponentes al segmento del pase al inicio)
      - Apertura del receptor (distancia a oponentes en el frame de recepciÃ³n)
    Devuelve passes_df con columnas agregadas.
    """
    if passes_df.empty:
        return passes_df

    df = cleaned_df.copy()
    # Asegurar tipos y columnas
    for col in ['Frame', 'Id', 'Pos X', 'Pos Y', 'Ball X', 'Ball Y', 'Team']:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida en cleaned_df: {col}")

    # Normalizadores empÃ­ricos (ajustables segÃºn tu escala de cancha)
    LEN_NORM = 2000.0        # ~20m si tus unidades ~cm
    SPACE_NORM = 300.0       # ~3m

    meta = {
        'length': [],
        'min_opp_to_line': [],
        'receiver_clearance': [],
        'opp_dominance_ratio': [],
        'difficulty': [],
    }

    for idx, row in passes_df.iterrows():
        id_em = row.get('id_emisor'); id_rc = row.get('id_receptor')
        team_em = row.get('team_emisor')
        f0 = int(row.get('Frame', row.get('frame_start', row.get('frame', 0))))  # compatibilidad si no hay frame guardado
        x1 = float(row.get('X_ball_inicio', row.get('Ball X', np.nan)))
        y1 = float(row.get('Y_ball_inicio', row.get('Ball Y', np.nan)))
        x2 = float(row.get('X_ball_Final', row.get('Ball X', np.nan)))
        y2 = float(row.get('Y_ball_final', row.get('Ball Y', np.nan)))

        length = hypot(x2 - x1, y2 - y1) if (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)) else 0.0

        # oponentes al inicio
        start_players = df[df['Frame'] == f0]
        opp_start = start_players[start_players['Team'] != team_em]
        opp_pts_start = opp_start[['Pos X', 'Pos Y']].to_numpy(dtype=np.float32) if len(opp_start) else np.zeros((0,2), np.float32)
        # distancia mÃ­nima de oponente al segmento del pase
        if len(opp_pts_start):
            dists = [ _point_segment_distance(px, py, x1, y1, x2, y2) for (px, py) in opp_pts_start ]
            min_opp_to_line = float(np.min(dists))
        else:
            min_opp_to_line = float('inf')

        # apertura receptor al final (si frame fin existe)
        # estimar frame de llegada: siguiente cambio de posesiÃ³n ya lo guardaste como frame_end en tu pipeline; si no, usa f0+1
        f1 = int(row.get('frame_end', f0 + 1))
        end_players = df[df['Frame'] == f1]
        opp_end = end_players[end_players['Team'] != team_em]
        rec_row = end_players[end_players['Id'] == id_rc]
        if len(rec_row) and len(opp_end):
            rx = float(rec_row.iloc[0]['Pos X']); ry = float(rec_row.iloc[0]['Pos Y'])
            receiver_clearance = _nearest_distance(opp_end[['Pos X','Pos Y']].to_numpy(dtype=np.float32), rx, ry)
        else:
            receiver_clearance = float('inf')

        # NormalizaciÃ³n y scoring (0 fÃ¡cil .. 1 difÃ­cil)
        length_norm = min(length / LEN_NORM, 1.0)
        clear_penalty = 1.0 - min((min_opp_to_line / SPACE_NORM) if np.isfinite(min_opp_to_line) else 1.0, 1.0)
        recv_penalty  = 1.0 - min((receiver_clearance / SPACE_NORM) if np.isfinite(receiver_clearance) else 1.0, 1.0)
        difficulty = 0.4*length_norm + 0.3*clear_penalty + 0.3*recv_penalty

        meta['length'].append(length)
        meta['min_opp_to_line'].append(min_opp_to_line)
        meta['receiver_clearance'].append(receiver_clearance)
        meta['difficulty'].append(float(np.clip(difficulty, 0.0, 1.0)))

    for k, v in meta.items():
        passes_df[k] = v
    return passes_df

def fix_early_unknown_possession_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Reasigna posesiones tempranas UNKNOWN al primer equipo no-UNKNOWN posterior."""
    if 'Unique_Possession' not in df.columns:
        df['Unique_Possession'] = df.get('Posesion', False)
    poss_frames = df[df['Unique_Possession'] == True].copy()
    if len(poss_frames) == 0:
        return df
    poss_one = poss_frames.sort_values(['Frame','Id']).groupby('Frame').head(1)
    later_known = poss_one[poss_one['Team'] != 'UNKNOWN']
    if len(later_known) == 0:
        return df
    first_known_row = later_known.sort_values('Frame').iloc[0]
    first_team = first_known_row['Team']
    first_frame = int(first_known_row['Frame'])
    mask = (df['Unique_Possession'] == True) & (df['Team'] == 'UNKNOWN') & (df['Frame'] <= first_frame)
    df.loc[mask, 'Team'] = first_team
    return df

def process_file(file_path, cleaned_output_path, output_possession_path, output_passes_path, output_team_passes_path):
    print("Iniciando procesamiento del archivo...")
    # Limpieza del archivo
    cleaned_df = clean_ball_trajectory(file_path, cleaned_output_path)

    # Calcular posesiÃ³n
    cleaned_df = calculate_possession(cleaned_df, radius=57)
    cleaned_df = validate_persistence(cleaned_df, frame_threshold=4)
    cleaned_df = resolve_disputes(cleaned_df, dispute_frames=2)
    # Reasignar UNKNOWN tempranos a primer equipo conocido
    cleaned_df = fix_early_unknown_possession_teams(cleaned_df)

    # Generar archivo de posesiÃ³n
    generate_possession_excel(cleaned_df, output_possession_path)

    # Detectar pases
    passes_df = detect_passes(cleaned_df)

    # Normalizar estructura si no hay pases (evita fallos posteriores)
    expected_cols = ['id_emisor','id_receptor','team_emisor','team_receptor','Frame','frame_end','X_ball_inicio','Y_ball_inicio','X_ball_Final','Y_ball_final']
    if passes_df is None or passes_df.empty or any(c not in passes_df.columns for c in expected_cols[:4]):
        print("No se detectaron pases o faltan columnas esperadas; creando archivo vacÃ­o con columnas estÃ¡ndar.")
        passes_df = pd.DataFrame(columns=expected_cols)
    else:
        # Asignar equipos desconocidos
        passes_df = assign_unknown_teams(cleaned_df, passes_df)
        # Guardar archivo de pases con dificultad aproximada (Voronoi-like)
        try:
            passes_df = calculate_pass_difficulty_v2(cleaned_df, passes_df)
        except NameError:
            # fallback a versiÃ³n base si no existe la v2
            passes_df = calculate_pass_difficulty(cleaned_df, passes_df)
    passes_df.to_excel(output_passes_path, index=False)
    print(f"Archivo de pases guardado en: {output_passes_path}")

    # Filtrar pases correctos
    correct_passes = passes_df[passes_df['team_emisor'] == passes_df['team_receptor']]

    # Dividir por equipos
    team_white_passes = correct_passes[correct_passes['team_emisor'] == 'equipo_blanco']
    team_black_passes = correct_passes[correct_passes['team_emisor'] == 'equipo_negro']

    # Guardar pases por equipo
    team_white_passes.to_excel(output_team_passes_path.replace("{team}", "equipo_blanco"), index=False)
    team_black_passes.to_excel(output_team_passes_path.replace("{team}", "equipo_negro"), index=False)
    # Mapas y mÃ©tricas finales
    try:
        generate_pass_maps(cleaned_df, correct_passes, output_dir='codes/outputs')
        generate_summary_metrics(cleaned_df, correct_passes, out_path='codes/data/metrics_summary.xlsx')
    except Exception as e:
        print('Advertencia: no se pudieron generar mapas/mÃ©tricas finales:', e)
    print("Procesamiento completado.")


