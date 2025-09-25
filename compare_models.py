import os
from dataset.load_dataset import LoadDataSet
from ai_model.predictor import RobotOnlyPredictor, BallRobotPredictor
from ai_model.losses import TestLoss
from comparison_tests import MLPComparison, KalmanFilterComparison
import sys
import numpy as np

sys.path.append(os.path.abspath('dataset'))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

test_files = ['dataset/proc_teste_2']


def compare_models(look_back, look_forth, output_dims, robot_model_name, ball_model_name):
    loader = LoadDataSet(look_back, look_forth)
    robot_x, ball_x, ball_mask, y = loader.load_data(test_files, for_test=True)
    loader.convert_to_real(y)

    print(f'--- Resultados para o modelo de robô {look_back} -> {look_forth}')
    seq_predictor = RobotOnlyPredictor(look_back, look_back, look_forth, output_dims, use_tf_function=True, forcing=False)
    seq_predictor(robot_x[0:1]) 
    seq_predictor.load_model(robot_model_name)
    res = seq_predictor.predict(robot_x, batch_size=512)
    y_pred_conv = loader.convert_batch(robot_x, res)
    test_loss = TestLoss()
    test_loss(y[:, 0:look_forth, 0:2], y_pred_conv[:, 0:look_forth])
    test_loss.print_error()

    print(f'--- Resultados para o modelo com bola {look_back} -> {look_forth}')
    seq_predictor = BallRobotPredictor(look_back, look_back, look_forth, output_dims, use_tf_function=True, forcing=False)
    seq_predictor([robot_x[0:1], ball_x[0:1], ball_mask[0:1]])
    seq_predictor.load_model(ball_model_name)
    res = seq_predictor.predict([robot_x, ball_x, ball_mask], batch_size=512)
    y_pred_conv = loader.convert_batch(robot_x, res)
    test_loss = TestLoss()
    test_loss(y[:, :, 0:2], y_pred_conv)
    test_loss.print_error()


compare_models(30, 15, 2, 'robot_30_15_t', 'ball_30_15_t')
compare_models(60, 30, 2, 'robot_30_60_t', 'ball_30_60_t')

print('--- Resultados para o modelo MLP 30 -> 15')
# Passamos input_dims=7 porque sabemos que os dados têm 7 features
mlp_comparison_model = MLPComparison(30, 15, 2, input_dims=7)
mlp_comparison_model.test_model(test_files, 'mlp_comp')

print('--- Resultados para o modelo MLP 60 -> 30')
mlp_comparison_model = MLPComparison(60, 30, 2, input_dims=7)
mlp_comparison_model.test_model(test_files, 'mlp_comp_2')

print("---- Resultados do Filtro de Kalman ----")
test_file_base_name = os.path.basename(test_files[0]).replace('proc_', '').replace('.pkl', '')

# CORRIGIDO: Passamos o caminho completo para o ficheiro de parâmetros
kf_comp = KalmanFilterComparison(30, 15, test_file_base_name, 'dataset/position_series_params')
kf_comp.perform_test()

kf_comp_2 = KalmanFilterComparison(60, 30, test_file_base_name, 'dataset/position_series_params')
kf_comp_2.perform_test()