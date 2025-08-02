import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import gc
from sklearn.metrics import mean_absolute_error
import optuna
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 自作モジュールの読み込み
sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.model import Model
from src.util import Util, Metric


class model_LGBM(Model):

    def __init__(self, run_fold_name: str, params, out_dir_name, logger) -> None:
        super().__init__(run_fold_name, params, logger)
        # カラム
        self.key_cols = self.params.pop("key_cols") # list
        self.target_col = self.params.pop("target_col") # str
        self.remove_cols = self.params.pop("remove_cols") # list
        # self.base_data_name = self.params.pop("base_data_name") # 不要
        # オブジェクト
        self.model = None
        self.feat_cols = None
        self.base_dir = os.path.join(out_dir_name, self.run_fold_name)
        os.makedirs(self.base_dir, exist_ok=True)


    def train(self, tr, va):
        """モデルの学習
        Args:
            data(pd.DataFrame): 学習データ[key_cols, target_col, predict, 特徴量]
        """

        # データセットの作成
        self.feat_cols = tr.columns.difference([self.target_col]+self.key_cols+self.remove_cols).tolist()
        tr_x, tr_y, va_x, va_y = tr[self.feat_cols], tr[self.target_col], va[self.feat_cols], va[self.target_col]
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
        self.model = lgb.train(
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

        # 学習曲線を保存
        self.plot_learning_curve(evals_result)

    
    def predict(self, te):
        """予測
        """
        df_te_pred = te[self.key_cols].copy()
        pred = self.model.predict(te[self.feat_cols], num_iteration=self.model.best_iteration)
        df_te_pred[self.target_col] = pred

        return df_te_pred.sort_values(self.key_cols)

    def save_model(self) -> None:
        """
        モデルを保存する
        """
        path_model = os.path.join(self.base_dir, 'model.pkl')
        path_feat_cols = os.path.join(self.base_dir, 'feat_cols.pkl')
        Util.dump(self.model, path_model)
        Util.dump(self.feat_cols, path_feat_cols)

    def load_model(self) -> None:
        """
        モデルを読み込む
        """
        path_model = os.path.join(self.base_dir, 'model.pkl')
        path_feat_cols = os.path.join(self.base_dir, 'feat_cols.pkl')
        self.model = Util.load(path_model)
        self.feat_cols = Util.load(path_feat_cols)


    def plot_learning_curve(self, evals_result):
        """23期分の学習曲線を保存
        """
        fig, ax = plt.subplots(1, 1, figsize=(24, 16))
        ax_ = ax
        ax_.plot(evals_result['train'][self.params.get("metric")], label='train')
        ax_.plot(evals_result['eval'][self.params.get("metric")], label='eval')
        ax_.set_title(f'Learning Curve')
        ax_.set_xlabel('Iterations')
        ax_.set_ylabel('metric')
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
        df_feature_importance = pd.DataFrame()
        df_feature_importance["feature"] = self.model.feature_name()
        df_feature_importance["importance"] = self.model.feature_importance(importance_type='gain')

        return df_feature_importance
    
##############################################################################

class model_LGBM_multi_feat(Model):

    def __init__(self, run_fold_name: str, params, logger) -> None:
        super().__init__(run_fold_name, params, logger)
        # カラム
        self.key_cols = self.params.pop("key_cols") # list
        self.target_col = self.params.pop("target_col") # str
        self.remove_cols = self.params.pop("remove_cols") # list
        self.base_data_name = self.params.pop("base_data_name") # str
        self.params_term = self.params.pop("params_term", None) 
        # オブジェクト
        self.models = []
        self.feat_cols = None
        self.base_dir = os.path.join(DIR_MODEL, self.run_fold_name)
        self.term_max = 23 # 予測対象の時間範囲
        os.makedirs(self.base_dir, exist_ok=True)

    def create_dataset(self, data, term, return_tr_va=False):
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
        list_date = data["datetime"].dt.date.unique()
        va_start_date = data['datetime'].max().replace(day=1, hour=0) # 最新月の1日

        # termのデータの読み込み
        df_term = Util.load_feature(f"{self.base_data_name}{term}")

        # 訓練データと検証データに分割
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

        if return_tr_va:
            return tr, va

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
            if term <= 6:
                self.models.append(None)
                evals_results.append(None)
                continue
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
            if self.params_term is not None:
                params.update(self.params_term[term-1])

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

            # termのデータの読み込み
            te_term = Util.load_feature(f"{self.base_data_name}{term}")
            te_term = te_term[te_term["datetime"].dt.date.isin(list_te_date)]
            model = self.models[term-1]
            if model is None:
                pred = 0
            else:
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
        for term in range(7, self.term_max+1):
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
        for term in range(7, self.term_max+1):
            model = self.models[term-1]
            df_importance = pd.DataFrame()
            df_importance["feature"] = model.feature_name()
            df_importance["importance"] = model.feature_importance(importance_type='gain')
            list_df_importance.append(df_importance)
        return list_df_importance
    
    def get_tuned_params(self, data, n_trials=10):
        """ハイパーパラメータのチューニング
        """
        def objective(trial):
            params = self.params.copy()
            num_round = params.pop('num_boost_round')
            early_stopping_rounds = params.pop('early_stopping_rounds')
            verbose = params.pop('verbose')
            period = params.pop('period')
            params['num_leaves'] = trial.suggest_int('num_leaves', 2, 256)
            params['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-8, 1.0)
            params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 5, 100)
            params['min_child_weight'] = trial.suggest_int('min_child_weight', 5, 100)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 10.0)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 10.0)
            # params['max_bin'] = trial.suggest_int('max_bin', 128, 512)
            params['min_split_gain'] = trial.suggest_float('min_split_gain', 0.0, 1.0)
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            params['subsample_freq'] = trial.suggest_int('subsample_freq', 1, 10)
            params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.5, 1.0)
            params["feature_pre_filter"] = False
            model = lgb.train(
                params,
                dtrain,
                num_round,
                valid_sets=(dtrain, dvalid),
                valid_names=("train", "eval"),
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=-1),
                           lgb.log_evaluation(period=period)],
            )
            va_pred = model.predict(va_x)
            score = mean_absolute_error(va_y, va_pred)

            return score
        
        self.params_term = []
        for term in tqdm(range(1, self.term_max+1)):
            # データセットの作成
            if term <= 6:
                self.params_term.append({})
            else:
                tr, va = self.create_dataset(data, term, return_tr_va=True)
                tr_va = pd.concat([tr, va], axis=0)
                # ランダムサンプリング
                tr, va = train_test_split(tr_va, test_size=0.2, random_state=0)
                # 特徴量とターゲットを分割
                tr_x = tr.drop(columns=[self.target_col]+self.key_cols+self.remove_cols)
                tr_y = tr[[self.target_col]]
                va_x = va.drop(columns=[self.target_col]+self.key_cols+self.remove_cols)
                va_y = va[[self.target_col]]
                dtrain = lgb.Dataset(tr_x, tr_y)
                dvalid = lgb.Dataset(va_x, va_y)

                pruner = optuna.pruners.HyperbandPruner()
                study = optuna.create_study(direction='minimize', pruner=pruner)
                study.optimize(objective, n_trials=n_trials)
                self.logger.info(f"term: {term}, best_score: {study.best_value}, best_params: {study.best_params}")
                self.params_term.append(study.best_params)

        return self.params_term
    


    