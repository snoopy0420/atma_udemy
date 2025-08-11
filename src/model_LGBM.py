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
from sklearn.inspection import permutation_importance
from lightgbm import LGBMClassifier

# 自作モジュールの読み込み
sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.model import Model
from src.util import Util, Metric


class model_LGBM(Model):

    def __init__(self, run_fold_name: str, params, out_dir_name, logger) -> None:
        super().__init__(run_fold_name, params, logger)
        # run params
        self.key_cols = self.params.pop("key_cols") # list
        self.target_col = self.params.pop("target_col") # str
        self.remove_cols = self.params.pop("remove_cols") # list
        self.tune = self.params.pop("tune") # bool
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

        # パラメータチューニング
        if self.tune[0]:
            self.tune_params(tr, va, n_trials=self.tune[1])

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

    # def get_permutation_importance(self, va):
    #     """Permutation Importanceを計算
    #     Args:
    #         te: 予測対象のデータ [key_cols, target_col, 特徴量]
    #     Returns:
    #         df_importance: 特徴量の重要度 [feature, importance]
    #     """
    #     cls_model = LGBMClassifier()
    #     cls_model._Booster = self.model
    #     cls_model._n_features = va[self.feat_cols].shape[1]
    #     cls_model.fitted_ = True

    #     print(va.shape)
    #     print(va[self.feat_cols].shape)
    #     print(va['target'].shape)
    #     result = permutation_importance(cls_model, va[self.feat_cols].values, va['target'].values.ravel(), n_repeats=30, random_state=42, scoring='roc_auc')

    #     df_importance = pd.DataFrame({
    #         'feature': self.feat_cols,
    #         'importance': result.importances_mean
    #     })
        
    #     return df_importance.sort_values('importance', ascending=False)

    def tune_params(self, tr, va, n_trials=10):
        """ハイパーパラメータのチューニング
        """
        def objective(trial):
            params = self.params.copy()
            num_round = params.pop('num_boost_round')
            early_stopping_rounds = params.pop('early_stopping_rounds')
            _ = params.pop('verbose')
            period = params.pop('period')
            params['max_depth'] = trial.suggest_int("max_depth", -1, 15)
            params['num_leaves'] = trial.suggest_int("num_leaves", 2, 128)
            params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.5, 1.0)
            params['bagging_freq'] = trial.suggest_int("bagging_freq", 0, 10)
            params['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 0.1)
            params['bagging_fraction'] = trial.suggest_float("bagging_fraction", 0.5, 1.0)
            params['colsample_bytree'] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            params['colsample_bynode'] = trial.suggest_float("colsample_bynode", 0.5, 1.0)
            params['lambda_l1'] = trial.suggest_float("lambda_l1", 0.0, 10.0)
            params['lambda_l2'] = trial.suggest_float("lambda_l2", 0.0, 10.0)
            params['min_data_in_leaf'] = trial.suggest_int("min_data_in_leaf", 10, 100)
            params["feature_pre_filter"] = False

            model = lgb.train(
                params,
                dtrain,
                num_round,
                valid_sets=(dtrain, dvalid),
                valid_names=("train", "eval"),
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=-1),
                           lgb.log_evaluation(period=period)],
                # feval=self.custum_eval, # カスタム評価関数
                # fobj=ModelLGB.custum_loss, # カスタム目的関数
            )
            va_pred = model.predict(va_x)
            score = Metric.my_metric(va_y, va_pred)

            return score
        
        tr_x = tr[self.feat_cols]
        tr_y = tr[[self.target_col]]
        va_x = va[self.feat_cols]
        va_y = va[[self.target_col]]
        dtrain = lgb.Dataset(tr_x, tr_y)
        dvalid = lgb.Dataset(va_x, va_y)

        pruner = optuna.pruners.HyperbandPruner()
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(objective, n_trials=n_trials)
        self.logger.info(f"best_score: {study.best_value}, best_params: {study.best_params}")
        # self.paramsを更新
        self.params.update(study.best_params)    

    