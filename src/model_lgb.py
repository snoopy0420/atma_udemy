import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# 定数の読み込み
CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.safe_load(file)
FIGURE_DIR_NAME = yml['SETTING']['DIR_FIGURE']
DIR_HOME = yml['SETTING']['DIR_HOME']
DIR_MODEL = yml['SETTING']['DIR_MODEL']
DIR_FIGURE = yml['SETTING']['DIR_FIGURE']

# 自作モジュールの読み込み
sys.path.append(DIR_HOME)
from src.model import Model
from src.util import Util, Metric


class ModelLGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        """モデルの学習
        """
        # データセットの作成
        dtrain = lgb.Dataset(tr_x, tr_y)
        dvalid = lgb.Dataset(va_x, va_y)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_boost_round')
        early_stopping_rounds = params.pop('early_stopping_rounds')
        verbose = params.pop('verbose')
        period = params.pop('period')

        # 学習
        evals_result = {}
        self.model = lgb.train(
                            params,
                            dtrain,
                            num_boost_round=num_round,
                            valid_sets=(dtrain, dvalid),
                            valid_names=("train", "eval"),
                            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose),
                                       lgb.log_evaluation(period=period),
                                       lgb.record_evaluation(evals_result)],
                            feval=self.custum_eval, # カスタム評価関数
                            # fobj=ModelLGB.custum_loss, # カスタム目的関数
                            )

        # 学習曲線を保存
        self.plot_learning_curve(evals_result)

    def predict(self, te_x):
        """予測（shapを計算しないver）
        """
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)


    def save_model(self):
        """モデルを保存
        """
        model_path = os.path.join(DIR_MODEL, self.run_fold_name, f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)


    def load_model(self):
        """モデルの読み込み
        """
        model_path = os.path.join(DIR_MODEL, self.run_fold_name, f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)


    @staticmethod
    def custum_eval(preds: np.ndarray, dtrain: lgb.Dataset):
        """カスタム評価関数（mape)
        """
        labels = dtrain.get_label()

        eval_result = roc_auc_score(labels, preds)

        return "AUC", eval_result, True
    

    def plot_learning_curve(self, evals_result):
        """学習過程の可視化
        """
        fig, ax = plt.subplots(figsize=(12,8))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        plt.tight_layout()
        plt.title('Learning curve')

        ax.plot(evals_result['train']["AUC"][10:], label="train")
        ax.plot(evals_result['eval']["AUC"][10:], label="valid")
        ax.set_xlabel('epoch')
        ax.set_ylabel("AUC")
        ax.legend()
        ax.grid(True)

        save_path = os.path.join(DIR_FIGURE, f'{self.run_fold_name}_lcurve.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def get_feature_importance(self):
        """特徴量の重要度を取得
        """
        return self.model.feature_importance(importance_type='gain')

############################################################################


    def predict_and_shap(self, te_x, shap_sampling):
        """予測（shapを計算するver うまくいかない）
        """
        fold_importance = shap.TreeExplainer(self.model).shap_values(te_x[:shap_sampling])
        valid_prediticion = self.model.predict(te_x, num_iteration=self.model.best_iteration)
        return valid_prediticion, fold_importance
    
    @staticmethod
    def custum_loss(preds: np.ndarray, dtrain: lgb.Dataset):
        """カスタム目的関数（fair loss)
        """
        # 残差を取得
        x = preds - dtrain.get_label()
        # Fair関数のパラメータ
        c = 1.0
        # 勾配の式の分母
        den = abs(x) + c
        # 勾配
        grad = c * x / den
        # 二階微分値
        hess = c * c / den ** 2

        return grad, hess


    
