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


class model_LGBM_multimodel(Model):

    def __init__(self, run_fold_name: str, params) -> None:
        super().__init__(run_fold_name, params)
        # カラム
        self.key_cols = self.params.pop("key_cols") # list
        self.target_col = self.params.pop("target_col") # str
        self.remove_cols = self.params.pop("remove_cols") # list
        # オブジェクト
        self.models = []
        self.feat_cols = None
        self.base_dir = os.path.join(DIR_MODEL, self.run_fold_name)
        self.term_max = 23 # 予測対象の時間範囲

    def create_dataset(self, data, term, feat_cols, is_test=False):
        """データセットの作成"""
        tr_x, tr_y, va_x, va_y = [], [], [], []
        max_date = data['datetime'].max()
        va_start_date = max_date.replace(day=1, hour=0)
        # 訓練データと検証データに分割
        tr = data[data["datetime"]<va_start_date]
        va = data[data["datetime"]>=va_start_date]
        # vaの2014-09-01以前のデータをpredict=2にする
        va.loc[va["datetime"]<"2014-09-01", "predict"] = 2
        for station_id in tr["station_id"].unique():
            tr_ = tr[tr["station_id"]==station_id]
            va_ = va[va["station_id"]==station_id]
            # 正解データの作成
            target_col_term = f"{self.target_col}_term{term}"
            tr_.sort_values("datetime", inplace=True) 
            va_.sort_values("datetime", inplace=True)
            tr_[target_col_term] = tr_[self.target_col].shift(-term)
            va_[target_col_term] = va_[self.target_col].shift(-term)
            # va期間のうちpredict==2のデータを検証データに使う
            predict_col_term = f"predict_term{term}"
            va_[predict_col_term] = va_["predict"].shift(-term)
            va_ = va_[va_[predict_col_term]==2]
            va_.drop(columns=[predict_col_term], inplace=True)
            # 00:00のデータを抽出
            tr_ = tr_[tr_["datetime"].dt.hour==0]
            va_ = va_[va_["datetime"].dt.hour==0]
            # 正解データが欠損している行を削除
            tr_ = tr_.dropna(subset=[target_col_term])
            va_ = va_.dropna(subset=[target_col_term])
            # x,yに分割
            tr_x.append(tr_[feat_cols])
            tr_y.append(tr_[target_col_term])
            va_x.append(va_[feat_cols])
            va_y.append(va_[target_col_term])
        tr_x, tr_y, va_x, va_y = pd.concat(tr_x, axis=0), pd.concat(tr_y, axis=0), pd.concat(va_x, axis=0), pd.concat(va_y, axis=0)

        return tr_x, tr_y, va_x, va_y
    
    def train(self, data):
        """モデルの学習"""
        # 特徴量
        feat_cols = [col for col in data.columns if col not in self.key_cols + self.remove_cols]

        # 1~23期モデルを学習
        for term in range(1, self.term_max+1):

            # データセットの作成
            tr_x, tr_y, va_x, va_y = self.create_dataset(data, term, feat_cols)
            dtrain = lgb.Dataset(tr_x, tr_y)
            dvalid = lgb.Dataset(va_x, va_y)

            # ハイパーパラメータ
            params = self.params.copy()
            num_round = params.pop('num_boost_round')
            early_stopping_rounds = params.pop('early_stopping_rounds')
            verbose = params.pop('verbose')
            period = params.pop('period')

            # 学習
            evals_result = {}
            model = lgb.train(
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

            self.models.append(model)

            # 学習曲線を保存
            self.plot_learning_curve(evals_result)

    def predict_term(self, te_x, term):
        """予測"""
        
        model = self.models[term - 1]

        return self.models[term - 1].predict(te_x)


    def plot_learning_curve(self, evals_result, term):
        """学習曲線を保存"""
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(evals_result['train']['l2'], label='train')
        plt.plot(evals_result['eval']['l2'], label='eval')
        plt.title(f'Term {term} Learning Curve')
        plt.xlabel('Iterations')
        plt.ylabel('L2 Loss')
        plt.legend()
        plt.savefig(os.path.join(self.base_dir, f'learning_curve_term_{term}.png'))
        plt.close()

    


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