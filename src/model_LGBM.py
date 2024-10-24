import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import gc


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

    def create_dataset(self, data, term, feat_cols):
        """データセットの作成
        termを指定してterm期先のterget_colから正解データを作成
        train,validにデータセットを分割

        Args:
            data(pd.DataFrame): 学習データ[key_cols, target_col, predict, 特徴量]
            term(int): 予測期間
            feat_cols(list): 特徴量
        Returns:
            tr_x: 学習データの特徴量
            tr_y: 学習データの目的変数
            va_x: バリデーションデータの特徴量
            va_y: バリデーションデータの目的変数
        """        
        # 訓練データと検証データに分割
        va_start_date = data['datetime'].max().replace(day=1, hour=0) # 最新月の1日
        tr = data[data["datetime"] < va_start_date].copy()
        va = data[data["datetime"] >= va_start_date].copy()
        
        # vaの2014-09-01以前のデータをpredict=2にする
        va.loc[va["datetime"] < "2014-09-01", "predict"] = 2

        # 正解データのシフト処理
        target_col_term = f"{self.target_col}_term{term}"
        tr[target_col_term] = tr.groupby("station_id")[self.target_col].shift(-term)
        va[target_col_term] = va.groupby("station_id")[self.target_col].shift(-term)

        # vaのうち、正解データのpredict == 2 のデータを検証データに使用
        predict_col_term = f"predict_term{term}"
        va[predict_col_term] = va.groupby("station_id")["predict"].shift(-term)
        va = va[va[predict_col_term] == 2]
        va = va.drop(columns=[predict_col_term])

        # 00:00のデータのみを抽出
        tr = tr[tr["datetime"].dt.hour == 0]
        va = va[va["datetime"].dt.hour == 0]

        # 正解データが欠損している行を削除
        tr = tr.dropna(subset=[target_col_term])
        va = va.dropna(subset=[target_col_term])

        # 特徴量とターゲットを分割
        tr_x = tr[feat_cols]
        tr_y = tr[[target_col_term]]
        va_x = va[feat_cols]
        va_y = va[[target_col_term]]

        # メモリ解放
        del tr, va
        gc.collect()

        return tr_x, tr_y, va_x, va_y
    
    def train(self, data):
        """モデルの学習
        Args:
            data(pd.DataFrame): 学習データ[key_cols, target_col, predict, 特徴量]
        """
        # 特徴量
        self.feat_cols = [col for col in data.columns if col not in self.key_cols+[self.target_col]+self.remove_cols]

        # 1~23期モデルを学習
        evals_results = []
        for term in range(1, self.term_max+1):
            # データセットの作成
            tr_x, tr_y, va_x, va_y = self.create_dataset(data, term, self.feat_cols)
            print(f"term: {term}, tr_x: {tr_x.shape}, tr_y: {tr_y.shape}, va_x: {va_x.shape}, va_y: {va_y.shape}")
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

    def predict_all(self, data):
        """与えられたデータの全ての0時断面に対して予測値を生成する
        """
        X = data[data["datetime"].dt.hour==0].copy()
        if self.target_col in X.columns:
            X = X.dropna(subset=[self.target_col])
        df_pred = self.predict(X)
        return df_pred


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
            ax_.plot(evals_results[term-1]['train'][self.params.get("metric")], label='train')
            ax_.plot(evals_results[term-1]['eval'][self.params.get("metric")], label='eval')
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
    
##############################################################################

class model_LGBM_multi_feat(Model):

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

    def create_dataset(self, data, term):
        """データセットの作成
        termを指定してterm期先のterget_colから正解データを作成
        train,validにデータセットを分割

        Args:
            data(pd.DataFrame): 学習データ[key_cols, target_col, predict, 特徴量]
            term(int): 予測期間
            feat_cols(list): 特徴量
        Returns:
            tr_x: 学習データの特徴量
            tr_y: 学習データの目的変数
            va_x: バリデーションデータの特徴量
            va_y: バリデーションデータの目的変数
        """        
        va_start_date = data['datetime'].max().replace(day=1, hour=0) # 最新月の1日
        list_date = data["datetime"].dt.date.unique()

        # termのデータ
        df_term = Util.load_feature(f"df_timefeat_{self.target_col}_hour{term}")
        tr_va = df_term[df_term["datetime"].dt.date.isin(list_date)]
        tr = tr_va[tr_va["datetime"] < va_start_date].copy()
        va = tr_va[tr_va["datetime"] >= va_start_date].copy()
        
        # vaの2014-09-01以前のデータをpredict=2にする
        va.loc[va["datetime"] < "2014-09-01", "predict"] = 2

        # vaのうち、正解データのpredict == 2 のデータを検証データに使用
        va = va[va["predict"] == 2]
        # va = va.drop(columns=["predict"])

        # 正解データが欠損している行を削除
        tr = tr.dropna(subset=[self.target_col])
        va = va.dropna(subset=[self.target_col])

        # 特徴量とターゲットを分割
        tr_x = tr.drop(columns=[self.target_col]+self.key_cols+self.remove_cols)
        tr_y = tr[[self.target_col]]
        va_x = va.drop(columns=[self.target_col]+self.key_cols+self.remove_cols)
        va_y = va[[self.target_col]]

        # メモリ解放
        del tr, va
        gc.collect()

        return tr_x, tr_y, va_x, va_y
    
    def train(self, data):
        """モデルの学習
        Args:
            data(pd.DataFrame): 学習データ[key_cols, target_col, predict, 特徴量]
        """
        print(f"max use train date : {data['datetime'].max()}")
        # 1~23期モデルを学習
        evals_results = []
        for term in range(1, self.term_max+1):
            # データセットの作成
            tr_x, tr_y, va_x, va_y = self.create_dataset(data, term)
            print(f"term: {term}, tr_x: {tr_x.shape}, tr_y: {tr_y.shape}, va_x: {va_x.shape}, va_y: {va_y.shape}")
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

    
    def predict(self, te):
        """予測
        Args:
            te: 予測対象の00:00のデータ [key_cols, target_col, 特徴量]
        Returns:
            df_te_pred: 予測対象の1~23期の予測結果 [key_cols, target_col]
        """
        # staion_id単位に1~23期の予測
        # i_fold = te["datetime"].min().replace(day=1, hour=0)
        list_te_date = te["datetime"].dt.date.unique()

        list_df_pred = []
        for term in range(1, self.term_max+1):
            te_term = Util.load_feature(f"df_timefeat_{self.target_col}_hour{term}")
            te_term = te_term[te_term["datetime"].dt.date.isin(list_te_date)]
            model = self.models[term-1]
            te_x = te_term[model.feature_name()]
            pred = model.predict(te_x, num_iteration=model.best_iteration)
            df_pred = te_term[self.key_cols].copy()
            df_pred[self.target_col] = pred
            list_df_pred.append(df_pred)

        return pd.concat(list_df_pred, axis=0).sort_values(self.key_cols)
        

    def predict_all(self, data):
        """与えられたデータの全ての0時断面に対して予測値を生成する
        Args:
            te: 予測対象のデータ [key_cols, target_col, 特徴量]
        Returns:
            df_te_pred: 予測対象の1~23期の予測結果 [key_cols, target_col]
        """
        return self.predict(data)


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
            ax_.plot(evals_results[term-1]['train'][self.params.get("metric")], label='train')
            ax_.plot(evals_results[term-1]['eval'][self.params.get("metric")], label='eval')
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