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

class model_ASMBL_demand_supply_intervention(Model):

    def __init__(self, run_fold_name: str, params, logger) -> None:
        super().__init__(run_fold_name, params, logger)
        # カラム
        self.key_cols = self.params.pop("key_cols") # list
        self.target_col = self.params.pop("target_col") # str
        self.remove_cols = self.params.pop("remove_cols") # list
        # models
        self.demand_model_cls = self.params.pop("demand_model_cls")
        self.supply_model_cls = self.params.pop("supply_model_cls")
        self.intervention_model_cls = self.params.pop("intervention_model_cls")
        self.demand_run_name = self.params.pop("demand_run_name")
        self.supply_run_name = self.params.pop("supply_run_name")
        self.intervention_run_name = self.params.pop("intervention_run_name")
        # オブジェクト
        self.model = None
        self.feat_cols = None
        self.base_dir = os.path.join(DIR_MODEL, self.run_fold_name)
        os.makedirs(self.base_dir, exist_ok=True)

    def create_dataset(self, data):
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
        # datasetの作成
        X = data[data["datetime"].dt.hour==0].copy()
        if self.target_col in X.columns:
            X = X.dropna(subset=[self.target_col])
        # モデル読み込み
        demand_model, supply_model, intervention_model = self._load_models()
        # 需要・供給・介入の予測
        # demand
        df_demand_pred = demand_model.predict(X)
        # supply
        df_supply_pred = supply_model.predict(X)
        # intervention        
        df_intervention_pred = intervention_model.predict(X)
        
        # 0時時点のデータを付与
        data_00 = data[data["datetime"].dt.hour==0][["station_id", "datetime", "bikes_available"]].copy()
        data_00["date"] = data_00["datetime"].dt.date
        data_00 = data_00.rename(columns={"bikes_available": "bikes_available_00"})
        data_00 = data_00[["station_id", "date", "bikes_available_00"]]
        df_pred = df_demand_pred.copy()
        df_pred = pd.merge(df_pred, df_supply_pred, on=["station_id","datetime"], how="left")
        df_pred = pd.merge(df_pred, df_intervention_pred, on=["station_id","datetime"], how="left")
        df_pred["date"] = df_pred["datetime"].dt.date
        df_pred = pd.merge(df_pred, data_00, on=["station_id", "date"], how="left")
        # 欠損値の削除
        df_pred = df_pred.dropna()

        return df_pred.sort_values(self.key_cols)
    
    def _load_models(self):
        """
        モデルを読み込む
        """
        i_fold = self.run_fold_name.split("_")[-1]
        demand_run_fold_name = f"{self.demand_run_name}_{i_fold}"
        supply_run_fold_name = f"{self.supply_run_name}_{i_fold}"
        intervention_run_fold_name = f"{self.intervention_run_name}_{i_fold}"
        with open(os.path.join(DIR_MODEL, self.demand_run_name, "params.yaml"), encoding="utf-8") as file:
            demand_params = yaml.safe_load(file)
        with open(os.path.join(DIR_MODEL, self.supply_run_name, "params.yaml"), encoding="utf-8") as file:
            supply_params = yaml.safe_load(file)
        with open(os.path.join(DIR_MODEL, self.intervention_run_name, "params.yaml"), encoding="utf-8") as file:
            intervention_params = yaml.safe_load(file)
        demand_model = self.demand_model_cls(demand_run_fold_name, demand_params, self.logger)
        demand_model.load_model()
        supply_model = self.supply_model_cls(supply_run_fold_name, supply_params, self.logger)
        supply_model.load_model()
        intervention_model = self.intervention_model_cls(intervention_run_fold_name, intervention_params, self.logger)
        intervention_model.load_model()

        return demand_model, supply_model, intervention_model

        
    def train(self, data):
        """
        Args:
            data(pd.DataFrame): 学習データ[key_cols, target_col, predict, 特徴量]
        """
        df_target = data[self.key_cols+[self.target_col]].copy()
        df_train = self.create_dataset(data)
        df_train_target = pd.merge(df_train, df_target, on=self.key_cols, how="inner")
        va_start_date = df_train_target['datetime'].max().replace(day=1, hour=0) # 最新月の1日
        tr = df_train_target[df_train_target["datetime"] < va_start_date].copy()
        va = df_train_target[df_train_target["datetime"] >= va_start_date].copy()
        tr_x = tr.drop(columns=self.key_cols+[self.target_col])
        tr_y = tr[self.target_col]
        va_x = va.drop(columns=self.key_cols+[self.target_col])
        va_y = va[self.target_col]
        print(tr_x.shape, tr_y.shape, va_x.shape, va_y.shape) 
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
        Args:
            te: 予測対象の00:00のデータ [key_cols, target_col, 特徴量]
        Returns:
            df_te_pred: 予測対象の1~23期の予測結果 [key_cols, target_col]
        """
        df_pred = self.create_dataset(te)
        feat_cols = self.model.feature_name()
        te_x = df_pred[feat_cols]
        pred = self.model.predict(te_x, num_iteration=self.model.best_iteration)
        df_pred[self.target_col] = pred
        return df_pred[self.key_cols+[self.target_col]].sort_values(self.key_cols)


    def save_model(self) -> None:
        """
        モデルを保存する
        """
        os.makedirs(self.base_dir, exist_ok=True)
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
        """学習過程の可視化
        """
        fig, ax = plt.subplots(figsize=(12,8))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        plt.tight_layout()
        plt.title('Learning curve')

        ax.plot(evals_result['train']["l1"], label="train")
        ax.plot(evals_result['eval']["l1"], label="valid")
        ax.set_xlabel('epoch')
        ax.set_ylabel("AUC")
        ax.legend()
        ax.grid(True)

        save_path = os.path.join(self.base_dir, 'learning_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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