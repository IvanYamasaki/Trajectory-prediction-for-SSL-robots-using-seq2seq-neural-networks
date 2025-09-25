import os
import pickle
import numpy as np
from pykalman import KalmanFilter
from dataset.kalman_smoother import KalmanSmoother
from ai_model.losses import TestLoss, SequenceLoss
import tensorflow as tf
from dataset.load_dataset import LoadDataSet
import matplotlib.pyplot as plt


class KalmanFilterComparison:

    def __init__(self, look_back, look_forth, data_file, params_file):
        self.look_back = look_back
        self.look_forth = look_forth

        # CORRIGIDO: Abre o ficheiro .pkl de dados brutos da pasta 'dataset'
        robots_f = open('dataset/' + data_file + '.pkl', 'rb')
        self.robots_t = pickle.load(robots_f)

        # CORRIGIDO: O ficheiro de parâmetros já tem o caminho completo
        filter_params_f = open(params_file + '.pkl', 'rb')
        params = pickle.load(filter_params_f)
        self.transition_matrix = params.A
        self.observation_matrix = params.C

        self.observation_covariance = np.linalg.inv(np.matmul(params.V_neg_sqrt, params.V_neg_sqrt))
        self.transition_covariance = np.linalg.inv(np.matmul(params.W_neg_sqrt, params.W_neg_sqrt))

        self.smoother = KalmanSmoother()
        # CORRIGIDO: O ficheiro de parâmetros já tem o caminho completo
        self.smoother.load_params(params_file) 
        self.true = []
        self.predicted = []
        robots_f.close()
        filter_params_f.close()

    def get_future(self, a_matrix, last_pos):
        res = []
        for i in range(self.look_forth):
            pos = np.inner(a_matrix, last_pos)
            last_pos = pos
            res.append([pos[0], pos[2]])
        return res

    def process_robots(self, robots):
        for k in range(0, len(robots)):
            for robot_id, series in robots[k].items():
                if len(series['x']) > (self.look_back + self.look_forth + 1):
                    x_hat, _, _ = self.smoother.smooth(series['x'], series['y'], series['mask'])
                    x_sm = x_hat[:, 0]
                    y_sm = x_hat[:, 2]
                    ism = [series['x'][0], 0, series['y'][0], 0]
                    kf = KalmanFilter(transition_matrices=self.transition_matrix,
                                      observation_matrices=self.observation_matrix,
                                      initial_state_mean=ism,
                                      observation_covariance=self.observation_covariance,
                                      transition_covariance=self.transition_covariance)
                    initial = np.array((series['x'][0:self.look_back], series['y'][0:self.look_back])).T
                    means, cov = kf.filter(initial)

                    self.true.append(np.array((x_sm[self.look_back:(self.look_back + self.look_forth)],
                                               y_sm[self.look_back:(self.look_back + self.look_forth)])).T)
                    self.predicted.append(np.array(self.get_future(kf.transition_matrices, means[-1])))

                    means, cov = means[-1], cov[-1]

                    for i in range(self.look_back + 1, len(series['x']) - self.look_forth - 1):
                        self.true.append(np.array((x_sm[(i + 1):(i + 1 + self.look_forth)],
                                                   y_sm[(i + 1):(i + 1 + self.look_forth)])).T)

                        means, cov = kf.filter_update(means, cov,
                                                      np.array((series['x'][i], series['y'][i])))
                        self.predicted.append(np.array(self.get_future(kf.transition_matrices, means)))
                    break

    def perform_test(self):
        self.process_robots(self.robots_t['blue'])
        self.process_robots(self.robots_t['yellow'])

        true = np.array(self.true)
        predicted = np.array(self.predicted)
        loss = TestLoss()
        loss(true, predicted)
        print(f"Look back: {self.look_back} | Look forth: {self.look_forth}")
        loss.print_error()


class MLPBatchLogs(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MLPBatchLogs, self).__init__()
        self.batch_logs = []
        self.val_logs = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_logs.append(logs['loss'])

    def on_test_end(self, logs=None):
        self.val_logs.append(logs['loss'])


class MLPComparison:
    model = None

    def __init__(self, look_back, look_forth, output_dims, input_dims=7, use_cuda=True):
        self.look_back = look_back
        self.output_dims = output_dims
        self.look_forth = look_forth
        self.input_dims = input_dims
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' if use_cuda else '-1'
        self.loader = LoadDataSet(look_back, look_forth)

    def create_model(self):
        data_input = tf.keras.Input(shape=(self.look_back, self.input_dims))
        x = tf.keras.layers.Dense(128, activation='relu')(data_input)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.output_dims * self.look_forth)(x)
        x = tf.keras.layers.Reshape((self.look_forth, self.output_dims))(x)
        return tf.keras.Model(inputs=data_input, outputs=x)

    def train_model(self, file_path: list, model_name):
        if self.model is None:
            self.model = self.create_model()
        robot_x, _, _, y = self.loader.load_data(file_path)
        
        actual_dims = robot_x.shape[2]
        if actual_dims != self.input_dims:
            print(f"A dimensão da entrada mudou. A recriar o modelo MLP para aceitar {actual_dims} features.")
            self.input_dims = actual_dims
            self.model = self.create_model()

        batch_logs = MLPBatchLogs()
        self.model.compile(optimizer=tf.optimizers.Adam(), loss=SequenceLoss(), run_eagerly=False)
        self.model.fit(robot_x, y, epochs=3, batch_size=512, callbacks=[batch_logs], validation_split=0.1)
        self.model.save(model_name + '.keras')

    def test_model(self, file_path: list, model_name):
        try:
            self.model = tf.keras.models.load_model(
                model_name + '.keras', 
                custom_objects={'SequenceLoss': SequenceLoss}
            )
        except IOError:
            print(f"Erro: Não foi possível encontrar o ficheiro do modelo treinado '{model_name}.keras'.")
            print("Por favor, execute 'train_models.py' primeiro para treinar e guardar todos os modelos.")
            return

        robot_x, _, _, y = self.loader.load_data(file_path, for_test=True)
        response = self.model.predict(robot_x)
        y_pred_conv = self.loader.convert_batch(robot_x, response)
        self.loader.convert_to_real(y)

        test_loss = TestLoss()
        test_loss(y[:, :, 0:2], y_pred_conv)
        test_loss.print_error()