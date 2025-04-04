import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función de limpieza de la trayectoria de la pelota
def clean_ball_trajectory(file_path, output_path, distance_threshold=900):
    print("Iniciando limpieza de posiciones de la pelota...")
    df = pd.read_excel(file_path)

    if 'Ball X' in df.columns and 'Ball Y' in df.columns:
        df = df.dropna(subset=['Ball X', 'Ball Y'])
    else:
        raise ValueError("Las columnas 'Ball X' o 'Ball Y' no están en el archivo Excel.")

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

# Funciones modulares para procesar posesión y pases
def calculate_possession(df, radius):
    print("Calculando posesión inicial...")
    df['Distance_to_Ball'] = np.sqrt((df['Pos X'] - df['Ball X'])**2 + (df['Pos Y'] - df['Ball Y'])**2)
    df['Initial_Possession'] = df['Distance_to_Ball'] <= radius
    print("Cálculo de posesión inicial completado.")
    return df

def validate_persistence(df, frame_threshold):
    print("Validando persistencia de posesión...")
    df['Validated_Possession'] = False
    grouped = df.groupby('Id')
    for player_id, group in grouped:
        possession_frames = group['Initial_Possession'].rolling(window=int(frame_threshold), min_periods=1).sum()
        df.loc[group.index, 'Validated_Possession'] = possession_frames >= frame_threshold
    print("Validación de persistencia completada.")
    return df

def resolve_disputes(df, dispute_frames):
    print("Resolviendo disputas de posesión...")
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
    print("Resolución de disputas completada.")
    return df

def generate_possession_excel(df, output_path):
    print("Generando archivo de posesión...")
    df['Posesion'] = df['Unique_Possession']
    df[['Frame', 'Id', 'Pos X', 'Pos Y', 'Ball X', 'Ball Y', 'Team', 'Posesion']].to_excel(output_path, index=False)
    print(f"Archivo de posesión generado en: {output_path}")

def detect_passes(df):
    print("Detectando pases...")
    possession_df = df[df['Posesion']].sort_values(by=['Frame', 'Id'])
    passes = []
    ball_positions = []
    previous_player = None
    previous_team = None

    for index, row in possession_df.iterrows():
        current_player = row['Id']
        current_team = row['Team']
        if previous_player is not None and previous_player != current_player:
            passes.append((previous_player, current_player, previous_team, current_team))
            ball_positions.append((row['Frame'], row['Ball X'], row['Ball Y']))
        previous_player = current_player
        previous_team = current_team

    passes_data = []
    for i in range(len(passes) - 1):
        id_emisor, id_receptor, team_emisor, team_receptor = passes[i]
        frame_start, x_start, y_start = ball_positions[i]
        frame_end, x_end, y_end = ball_positions[i + 1]
        passes_data.append({
            'id_emisor': id_emisor,
            'id_receptor': id_receptor,
            'team_emisor': team_emisor,
            'team_receptor': team_receptor,
            'X_ball_inicio': x_start,
            'Y_ball_inicio': y_start,
            'X_ball_Final': x_end,
            'Y_ball_final': y_end
        })

    print("Detección de pases completada.")
    return pd.DataFrame(passes_data)

def assign_unknown_teams(df, passes_df):
    print("Asignando equipos a jugadores UNKNOWN...")
    team_mapping = df[['Id', 'Team']].drop_duplicates().sort_values(by=['Id', 'Team'])
    team_mapping = team_mapping[team_mapping['Team'] != 'UNKNOWN']
    team_mapping = team_mapping.drop_duplicates(subset=['Id'], keep='last')
    team_dict = dict(zip(team_mapping['Id'], team_mapping['Team']))
    passes_df['team_emisor'] = passes_df['id_emisor'].map(team_dict).fillna(passes_df['team_emisor'])
    passes_df['team_receptor'] = passes_df['id_receptor'].map(team_dict).fillna(passes_df['team_receptor'])
    print("Asignación de equipos completada.")
    return passes_df

def process_file(file_path, cleaned_output_path, output_possession_path, output_passes_path, output_team_passes_path):
    print("Iniciando procesamiento del archivo...")
    # Limpieza del archivo
    cleaned_df = clean_ball_trajectory(file_path, cleaned_output_path)

    # Calcular posesión
    cleaned_df = calculate_possession(cleaned_df, radius=57)
    cleaned_df = validate_persistence(cleaned_df, frame_threshold=4)
    cleaned_df = resolve_disputes(cleaned_df, dispute_frames=2)

    # Generar archivo de posesión
    generate_possession_excel(cleaned_df, output_possession_path)

    # Detectar pases
    passes_df = detect_passes(cleaned_df)

    # Asignar equipos desconocidos
    passes_df = assign_unknown_teams(cleaned_df, passes_df)

    # Guardar archivo de pases general
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
    print("Procesamiento completado.")

# Ejecución del flujo completo
process_file(
    file_path='Posiciones-jugadores-balon.xlsx',
    cleaned_output_path='codes/cleaned_output_path.xlsx',
    output_possession_path='codes/output_possession_path.xlsx',
    output_passes_path='codes/output_passes_path.xlsx',
    output_team_passes_path='codes/passes_by_{team}.xlsx'
)

