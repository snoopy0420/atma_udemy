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

import gc

class model_LGBM_multimodel(Model):

    def __init__(self, run_fold_name: str, params, logger) -> None:
        super().__init__(run_fold_name, params, logger)
        # カラム
        self.key_cols = self.params.pop("key_cols") # list
        self.target_col = self.params.pop("target_col") # str
        self.remove_cols = self.params.pop("remove_cols") # list
        # オブジェクト
        self.models = []
        self.feat_cols = None
        self.base_dir = os.path.join(DIR_MODEL, self.run_fold_name)
        self.term_max = 23 # 予測対象の時間範囲
        os.makedirs(self.base_dir, exist_ok=True)

    def create_dataset(self, data, term, feat_cols, is_test=False):
        """データセットの作成
        termを指定してterm期先のterget_colから正解データを作成
        train,validにデータセットを分割

        Args:
            data(pd.DataFrame): 学習データ[key_cols, target_col, predict, 特徴量]
            term(int): 予測期間
            feat_cols(list): 特徴量
            is_test(bool): テストデータの場合
        Returns:
            tr_x: 学習データの特徴量
            tr_y: 学習データの目的変数
            va_x: バリデーションデータの特徴量
            va_y: バリデーションデータの目的変数
        """
        tr_x, tr_y, va_x, va_y = [], [], [], []
        va_start_date = data['datetime'].max().replace(day=1, hour=0)
        # 訓練データと検証データに分割
        tr = data[data["datetime"]<va_start_date].copy()
        va = data[data["datetime"]>=va_start_date].copy()
        # vaの2014-09-01以前のデータをpredict=2にする
        va.loc[va["datetime"]<"2014-09-01", "predict"] = 2
        for station_id in tr["station_id"].unique():
            tr_ = tr[tr["station_id"]==station_id].copy()
            va_ = va[va["station_id"]==station_id].copy()
            # 正解データの作成
            target_col_term = f"{self.target_col}_term{term}"
            tr_ = tr_.sort_values("datetime")
            va_ = va_.sort_values("datetime")
            tr_[target_col_term] = tr_[self.target_col].shift(-term)
            va_[target_col_term] = va_[self.target_col].shift(-term)
            # vaのうち正解データのpredict==2となるデータを検証データに使う
            predict_col_term = f"predict_term{term}"
            va_[predict_col_term] = va_["predict"].shift(-term)
            va_ = va_[va_[predict_col_term]==2]
            va_ = va_.drop(columns=[predict_col_term])
            # 00:00のデータを抽出
            tr_ = tr_[tr_["datetime"].dt.hour==0]
            va_ = va_[va_["datetime"].dt.hour==0]
            # 正解データが欠損している行を削除
            tr_ = tr_.dropna(subset=[target_col_term])
            va_ = va_.dropna(subset=[target_col_term])
            # x,yに分割
            tr_x.append(tr_[feat_cols])
            tr_y.append(tr_[[target_col_term]])
            va_x.append(va_[feat_cols])
            va_y.append(va_[[target_col_term]])
        tr_x, tr_y, va_x, va_y = pd.concat(tr_x, axis=0), pd.concat(tr_y, axis=0), pd.concat(va_x, axis=0), pd.concat(va_y, axis=0)

        return tr_x, tr_y, va_x, va_y
    
    def train(self, data):
        """モデルの学習
        Args:
            data(pd.DataFrame): 学習データ[key_cols, target_col, predict, 特徴量]
        """
        # 特徴量
        self.feat_cols = [col for col in data.columns if col not in self.key_cols + self.remove_cols]

        # 1~23期モデルを学習
        evals_results = []
        for term in range(1, self.term_max+1):

            # データセットの作成
            tr_x, tr_y, va_x, va_y = self.create_dataset(data, term, self.feat_cols)
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
                num_round,
                valid_sets=(dtrain, dvalid),
                valid_names=("train", "eval"),
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose),
                           lgb.log_evaluation(period=period),
                           lgb.record_evaluation(evals_result)],
                # feval=self.custum_eval, # カスタム評価関数
                # fobj=ModelLGB.custum_loss, # カスタム目的関数
            )

            self.models.append(model)
            evals_results.append(evals_result)

        # 学習曲線を保存
        self.plot_learning_curve(evals_results)


    def predict_term(self, te, term):
        """term先の予測
        """
        model = self.models[term-1]
        te_x = te[self.feat_cols]
        pred = model.predict(te_x, num_iteration=model.best_iteration)

        return pred
    
    def predict(self, te):
        """予測
        Args:
            te: 予測対象の00:00のデータ [key_cols, target_col, 特徴量]
        Returns:
            df_te_pred: 予測対象の1~23期の予測結果 [key_cols, target_col]
        """
        # staion_id単位に1~23期の予測
        list_df_pred = []
        for station_id in te["station_id"].unique():
            te_ = te[te["station_id"]==station_id].copy()
            for term in range(1, self.term_max+1):
                pred = self.predict_term(te_, term)
                df_pred = te_[self.key_cols].copy()
                df_pred["datetime"] = df_pred["datetime"] + pd.Timedelta(hours=term)
                df_pred[self.target_col] = pred
                list_df_pred.append(df_pred)
        df_te_pred = pd.concat(list_df_pred, axis=0)
        return df_te_pred.sort_values(self.key_cols)


    def save_model(self) -> None:
        """
        モデルを保存する
        """
        os.makedirs(self.base_dir, exist_ok=True)
        path_model = os.path.join(self.base_dir, 'models.pkl')
        path_feat_cols = os.path.join(self.base_dir, 'feat_cols.pkl')
        Util.dump(self.models, path_model)
        Util.dump(self.feat_cols, path_feat_cols)

    def load_model(self) -> None:
        """
        モデルを読み込む
        """
        path_model = os.path.join(self.base_dir, 'models.pkl')
        path_feat_cols = os.path.join(self.base_dir, 'feat_cols.pkl')
        self.models = Util.load(path_model)
        self.feat_cols = Util.load(path_feat_cols)


    def plot_learning_curve(self, evals_results):
        """23期分の学習曲線を保存
        """
        # 23期分の学習曲線をaxを分けて描画し保存する
        fig, ax = plt.subplots(4, 6, figsize=(24, 16))
        for term in range(1, self.term_max+1):
            ax_ = ax[(term-1)//6][(term-1)%6]
            ax_.plot(evals_results[term-1]['train']['l1'], label='train')
            ax_.plot(evals_results[term-1]['eval']['l1'], label='eval')
            ax_.set_title(f'Term {term} Learning Curve')
            ax_.set_xlabel('Iterations')
            ax_.set_ylabel('L1 Loss')
            ax_.legend()
        save_path = os.path.join(self.base_dir, 'learning_curve.png')
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    
    def get_feature_importance(self):
        """termごとの特徴量の重要度を取得
        return:
            df_importance: termごとの特徴量の重要度 [feature, importance]
        """
        list_df_importance = []
        for term in range(1, self.term_max+1):
            model = self.models[term-1]
            df_importance = pd.DataFrame()
            df_importance["feature"] = model.feature_name()
            df_importance["importance"] = model.feature_importance(importance_type='gain')
            list_df_importance.append(df_importance)
        return list_df_importance