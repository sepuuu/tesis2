import sys
sys.path.append('codes')
from postprocess.postprocess import process_file
try:
    process_file(
        file_path='codes/data/Posiciones-jugadores-balon-multicam.xlsx',
        cleaned_output_path='codes/data/limpieza.xlsx',
        output_possession_path='codes/data/posesion.xlsx',
        output_passes_path='codes/data/pases.xlsx',
        output_team_passes_path='codes/data/passes_by_{team}.xlsx'
    )
    print('OK: postprocess finished')
except Exception as e:
    import traceback
    print('ERR:', e)
    traceback.print_exc()
