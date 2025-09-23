from .read_logs import process_log
from .smoother_params import get_ball_position_series_params, get_robot_position_series_params, get_robot_heading_series_params
from .smooth_data import Smoother

# Lista dos seus arquivos de log (sem a extensão .log)
data_set_files = ['treino_1', 'treino_2', 'treino_3', 'treino_4', 'teste_1']

# Lista CORRIGIDA dos nomes dos arquivos de saída processados
processed_data_files = ['proc_treino_1', 'proc_treino_2', 'proc_treino_3', 'proc_treino_4', 'proc_teste_1']

print("---- Lendo arquivos de log brutos ----")
for file in data_set_files:
    print(file)
    process_log(file)

print('---- Processando parâmetros do Kalman (Ball) ----')
get_ball_position_series_params()
print('---- Processando parâmetros do Kalman (Robot Position) ----')
get_robot_position_series_params()
print('---- Processando parâmetros do Kalman (Robot Heading) ----')
get_robot_heading_series_params()

print("---- Suavizando os dados de todas as trajetórias ----")
smoother = Smoother()
for (source, dest) in zip(data_set_files, processed_data_files):
    print(f"Suavizando {source} -> {dest}")
    smoother.smooth_data(source, dest)

print("---- Processamento de dados concluído! ----")