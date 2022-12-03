import os
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import ReLU, PReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler

from model import Model
from util import Util

import shap


# tensorflowの警告抑制
import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.safe_load(file)
FIGURE_DIR_NAME = yml['SETTING']['FIGURE_DIR_NAME']

# 各foldの学習過程を保存する配列
hists = []

class ModelNN(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット・スケーリング
        self.scaler = StandardScaler().fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        # tr_y = np_utils.to_categorical(tr_y)

        validation = va_x is not None
        if validation:
            va_x = self.scaler.transform(va_x)
            # va_y = np_utils.to_categorical(va_y)

        # パラメータ
        num_classes = self.params["num_classes"]
        input_dropout = self.params['input_dropout']
        hidden_layers = int(self.params['hidden_layers'])
        hidden_units = int(self.params['hidden_units'])
        hidden_activation = self.params['hidden_activation']
        hidden_dropout = self.params['hidden_dropout']
        batch_norm = self.params['batch_norm']
        output_activation = self.params["output_activation"]
        optimizer_type = self.params['optimizer']['type']
        optimizer_lr = self.params['optimizer']['lr']
        loss = self.params['loss']
        metrics = self.params['metrics']
        batch_size = int(self.params['batch_size'])

        # モデルの構築
        self.model = Sequential()

        # 入力層
        self.model.add(Dropout(input_dropout, input_shape=(tr_x.shape[1],)))

        # 中間層
        for i in range(hidden_layers):
            self.model.add(Dense(hidden_units))
            if batch_norm == 'before_act':
                self.model.add(BatchNormalization())
            if hidden_activation == 'prelu':
                self.model.add(PReLU())
            elif hidden_activation == 'relu':
                self.model.add(ReLU())
            else:
                raise NotImplementedError
            self.model.add(Dropout(hidden_dropout))

        # 出力層
        self.model.add(Dense(num_classes, activation=output_activation))

        # オプティマイザ
        if optimizer_type == 'sgd':
            optimizer = SGD(lr=optimizer_lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif optimizer_type == 'adam':
            optimizer = Adam(lr=optimizer_lr, beta_1=0.9, beta_2=0.999, decay=0.)
        else:
            raise NotImplementedError

        # 目的関数、評価指標などの設定
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        # エポック数、アーリーストッピング、学習の実行
        # あまりepochを大きくすると、小さい学習率のときに終わらないことがあるので注意
        nb_epoch = 1000
        patience = 100 
        if validation:
            early_stopping = EarlyStopping(monitor='val_loss',patience=patience, restore_best_weights=True)
            history = self.model.fit(
                tr_x, tr_y,
                epochs=nb_epoch,
                batch_size=batch_size, 
                verbose=0,
                validation_data=(va_x, va_y),
                callbacks=[early_stopping]
                )
        else:
            self.model.fit(
                tr_x, tr_y, 
                nb_epoch=nb_epoch, 
                batch_size=batch_size, 
                verbose=0
                )
        

        hists.append(pd.DataFrame(history.history))

        
    def predict(self, te_x):
        te_x = self.scaler.transform(te_x)
        pred = self.model.predict_proba(te_x)
        return pred

     # shapを計算するver、うまくいかない
    def predict_and_shap(self, te_x, shap_sampling):
        explainer = shap.KernelExplainer(self.model.predict_proba, te_x[:shap_sampling])
        shap_values = explainer.shap_values(te_x[:shap_sampling])
        te_x = self.scaler.transform(te_x)
        pred = self.model.predict_proba(te_x)
        return pred, shap_values

    def save_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(path, f'{self.run_fold_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(path, f'{self.run_fold_name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Util.load(scaler_path)

    @classmethod
    def plot_learning_curve(self, run_name):
        """学習曲線を描く
        """
        print(len(hists))
        # # 学習過程を取得
        # hist_df = hists[0]

        # # 出力
        # fig = plt.figure(figsize = (12,8))
        # plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        # plt.tight_layout()

        # # 棒グラフを出力
        # ax1 = fig.add_subplot(111)
        # ax1.set_title('Learning curve')
        # ax1.set_xlabel('epoch')
        # ax1.plot(hist_df["loss"], label="trian loss")
        # ax1.plot(hist_df["val_loss"], label="valid loss")

        # # 凡例を表示（グラフ左上、ax2をax1のやや下に持っていく）
        # ax1.legend()

        # # グリッド表示(ax1のみ)
        # ax1.grid(True)

        fig, axes = plt.subplots(2, 2, figsize=(12,8))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        plt.tight_layout()
        plt.title('Learning curve')

        for i, ax in enumerate(axes.ravel()):
            ax.plot(hists[i]["loss"], label="trian loss")
            ax.plot(hists[i]["val_loss"], label="valid loss")
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.legend()
            ax.grid(True)

        plt.savefig(FIGURE_DIR_NAME + run_name + '_curve.png', dpi=300, bbox_inches="tight")
        plt.close()
    

    
    
