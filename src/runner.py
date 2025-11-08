import os
import sys
import math
import yaml
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Union, Optional
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

# 自作モジュールの読み込み
sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.model import Model
from src.util import Util, Metric

from matplotlib import rcParams
# 日本語フォントを設定
rcParams['font.family'] = 'Meiryo'  # または 'MS Gothic'

class Runner:
    """学習・予測・評価・パラメータチューニングを担うクラス
    """

    # コンストラクタ
    def __init__(self,
                 run_name: str, # runの名前
                 model_cls: Callable[[str, dict], Model], #モデルのクラス
                 params: dict, # ハイパーパラメータ
                 df_train: pd.DataFrame, df_test: pd.DataFrame, 
                 run_setting: dict, cv_setting: dict,
                 logger,
                 memo,
                 ): 
        
        self.run_name = run_name
        self.model_cls = model_cls
        self.params = params
        self.logger = logger
        self.memo = memo
        # params
        self.key_cols = params.get("key_cols") # list
        self.target_col = params.get("target_col") # str
        # run_setting
        self.after_predict_process = run_setting.get("after_predict_process", None)
        self.after_split_process = run_setting.get("after_split_process", None)  # データセットの分割後に行う処理
        # cv_setting
        self.group_col = cv_setting.get("group_col")
        self.n_splits = cv_setting.get("n_splits", 5)
        self.validator = StratifiedGroupKFold(n_splits=self.n_splits,
                                              shuffle=cv_setting.get("shuffle", True), 
                                              random_state=cv_setting.get("random_state", 42))
        # データのセット
        self.df_train = df_train
        self.df_test = df_test
        self.out_dir_name = os.path.join(DIR_MODEL, run_name)


    def metric(self, true, pred):
        """評価指標の計算
        """
        score = Metric.my_metric(true, pred)
        return score


    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # run名、i_fold、モデルのクラス名からモデルを作成する
        run_fold_name = f'{self.run_name}_fold-{i_fold}'
        model = self.model_cls(run_fold_name, self.params.copy(), self.out_dir_name, self.logger)
        return model
    

    def create_train_valid_dateset(self, i_fold):
        """
        foldを指定して訓練・検証データを準備する
        """
        groups = self.df_train[self.group_col].values
        y = self.df_train[self.target_col].values
        tr_idx, va_idx = list(self.validator.split(self.df_train, y, groups))[i_fold]
        tr = self.df_train.iloc[tr_idx].reset_index(drop=True)
        va = self.df_train.iloc[va_idx].reset_index(drop=True)

        # データセットの分割後に行う処理
        if self.after_split_process:
            tr, va = self.after_split_process(tr, va)

        return tr, va

    def train_fold(self, i_fold, metrics=None) -> Tuple[Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """foldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）, metrics: 評価に用いる関数
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """

        # データセットの準備
        tr, va = self.create_train_valid_dateset(i_fold)
        
        # 学習を行う
        model = self.build_model(i_fold)
        model.train(tr, va)

        return model


    def run_train_cv(self) -> None:
        """CVでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        # ログ
        self.logger.info(f'{self.run_name} - start training cv')

        # fold毎の学習：train_foldをn_splits回繰り返す
        for i_fold in range(self.n_splits):

            self.logger.info(f'{self.run_name} fold {i_fold} - start training')
            # 学習を行う
            model = self.train_fold(i_fold)
            # モデルを保存する
            model.save_model()
            self.logger.info(f'{self.run_name} fold {i_fold} - end training')

        # パラメータの保存
        try:
            path_output = os.path.join(self.out_dir_name, f'params.yaml')
            Util.jump_json(self.params, path_output)
        except:
            self.logger.info("パラメータは保存しません")

        self.logger.info(f'{self.run_name} - end training cv')


    def metric_fold(self, i_fold):
        """
        foldを指定して評価を行う
        """

        # データセットの準備
        _, va = self.create_train_valid_dateset(i_fold)

        # 学習済みモデル
        model = self.build_model(i_fold)
        model.load_model()
        
        # 検証データの予測
        df_va_true = va[self.key_cols + [self.target_col]]
        df_va_pred = model.predict(va)

        # 後処理
        if self.after_predict_process is not None:
            df_va_pred = self.after_predict_process(df_va_pred, self.target_col)
        
        # バリデーションデータの評価
        va_score = self.metric(df_va_true[self.target_col].values, df_va_pred[self.target_col].values)

        return va_score, df_va_pred


    def run_metric_cv(self):
        """
        CVでの評価を行う
        """
        self.logger.info(f'{self.run_name} - start metric cv')

        scores = [] # 各foldのscoreを保存
        preds = [] # 各foldの予測値を保存

        # fold毎の検証データの予測・評価
        for i_fold in tqdm(range(self.n_splits)):
            # 評価を行う
            score, df_va_pred = self.metric_fold(i_fold)
            # 結果を保持する
            scores.append(score)
            preds.append(df_va_pred)
        
        df_va_preds = pd.concat(preds, axis=0)
        df_va_preds_true = pd.merge(
            df_va_preds, self.df_train[self.key_cols + [self.target_col]], 
            on=self.key_cols, how='left', suffixes=('_pred', '_true')
        )
        score_all = self.metric(
            df_va_preds_true[self.target_col + '_true'].values, 
            df_va_preds_true[self.target_col + '_pred'].values
        )

        # 評価結果の保存
        self.logger.result(f"memo: {self.memo}")
        self.logger.result_scores(self.run_name, scores)
        self.logger.result(f"all: {score_all}, mean: {np.mean(scores)}, std: {np.std(scores)}")
        self.logger.info(f"mean: {np.mean(scores)}, std: {np.std(scores)}")
        # 予測結果の保存
        path_output = os.path.join(self.out_dir_name, f'va_pred.pkl')
        Util.dump_df_pickle(df_va_preds, path_output)
        self.logger.info(f'output predict : {path_output}')

        self.logger.info(f'{self.run_name} - end metric cv')


    def predict_fold(self, i_fold: Union[int, str]):
        """foldを指定して予測を行う"""

        # 学習済みモデル
        model = self.build_model(i_fold)
        model.load_model()

        # 予測
        df_te_pred = model.predict(self.df_test)

        # 後処理
        if self.after_predict_process is not None:
            df_te_pred = self.after_predict_process(df_te_pred, self.target_col)

        return df_te_pred.sort_values(self.key_cols) 


    def run_predict_cv(self) -> None:
        """CVでテストデータの予測を行う
        """
        self.logger.info(f'{self.run_name} - start prediction cv')
        te_preds = []

        # fold毎のテストデータの予測
        df_te_preds = self.df_test.copy()[self.key_cols]
        for i_fold in tqdm(range(self.n_splits)):
            df_te_pred = self.predict_fold(i_fold)
            te_preds.append(df_te_pred[self.target_col].values)

        # 予測の平均値を出力する
        df_te_preds = df_te_preds.sort_values(self.key_cols)
        df_te_preds[self.target_col] = np.mean(te_preds, axis=0)

        # 予測結果の保存
        path_output = os.path.join(self.out_dir_name, f'te_pred.pkl')
        Util.dump_df_pickle(df_te_preds, path_output)
        self.logger.info(f'output predict : {path_output}')

        self.logger.info(f'{self.run_name} - end prediction cv')


    def get_feature_importance_fold(self, i_fold):

        # 学習済みモデル
        model = self.build_model(i_fold)
        model.load_model()

        # 予測
        df_feature_importance = model.get_feature_importance()
        
        # データセットの準備
        # _, va = self.create_train_valid_dateset(i_fold)
        # df_feature_importance = model.get_permutation_importance(va)

        return df_feature_importance


    def plot_feature_importance_cv(self) -> None:
        """CVで学習した各foldのモデルの平均により、特徴量の重要度を取得する
        """
        self.logger.info(f'{self.run_name} - start plot feature importance cv')

        # 各foldの特徴量の重要度を取得
        list_df_feature_importance = []
        for i_fold in range(self.n_splits):
            list_df_feature_importance.append(self.get_feature_importance_fold(i_fold))

        list_importance = []
        for df_feature_importance in list_df_feature_importance:
            importance = df_feature_importance["importance"].values
            list_importance.append(importance)

        df_feature_importance_mean = pd.DataFrame({
            "feature": list_df_feature_importance[0]["feature"],
            "mean": np.mean(list_importance, axis=0),
            "std": np.std(list_importance, axis=0)
        })

        df = df_feature_importance_mean
        # 変動係数を算出
        df['coef_of_var'] = df['std'] / df['mean']
        df['coef_of_var'] = df['coef_of_var'].fillna(0)
        df = df.sort_values('mean', ascending=True)

        # 100以上は表示しない
        df = df[-100:]

        fig = plt.figure(figsize = (40, 30))
        ax1 = fig.add_subplot(1, 1, 1)

        # 棒グラフを出力
        ax1.set_title(f'feature importance gain')
        ax1.set_xlabel('feature importance mean & std')
        ax1.barh(df["feature"], df['mean'], label='mean',  align="center", alpha=0.6)
        ax1.barh(df["feature"], df['std'], label='std',  align="center", alpha=0.6)

        # 折れ線グラフを出力
        ax2 = ax1.twiny()
        ax2.plot(df['coef_of_var'], df["feature"], linewidth=1, color="crimson", marker="o", markersize=8, label='coef_of_var')
        ax2.set_xlabel('Coefficient of variation')

        # 凡例を表示（グラフ左上、ax2をax1のやや下に持っていく）
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=12)
        ax2.legend(bbox_to_anchor=(1, 0.97), loc='upper right', borderaxespad=0.5, fontsize=12)

        # グリッド表示(ax1のみ)
        ax1.grid(True)
        ax2.grid(False)
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        plt.tight_layout()

        # 図を保存
        path_output = os.path.join(self.out_dir_name, f'fi_gain.png')
        plt.savefig(path_output, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f'{self.run_name} - end plot feature importance cv')


