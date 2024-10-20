import os
import sys
import math
import yaml
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Union, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from datetime import datetime


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.safe_load(file)
FIGURE_DIR_NAME = yml['SETTING']['DIR_FIGURE']
DIR_HOME = yml['SETTING']['DIR_HOME']
DIR_MODEL = yml['SETTING']['DIR_MODEL']
DIR_FIGURE = yml['SETTING']['DIR_FIGURE']
TARGET_COL = yml['SETTING']['TARGET_COL']
REMOVED_COL = yml['SETTING']['REMOVE_COLS']
DIR_INTERIM = yml['SETTING']['DIR_INTERIM']

sys.path.append(DIR_HOME)
from src.model import Model
from src.util import Util


class TimeseriesModelRunner:
    """学習・予測・評価・パラメータチューニングを担うクラス
    """

    # コンストラクタ
    def __init__(self,
                 run_name: str, # runの名前
                 model_cls: Callable[[str, dict], Model], #モデルのクラス
                 params: dict, # ハイパーパラメータ
                 df_main: pd.DataFrame, # 学習データ
                 run_setting: dict,
                 cv_setting: dict,
                 logger,
                 memo,
                 ): 
        self.memo = memo
        self.logger = logger
        self.run_name = run_name
        self.model_cls = model_cls
        self.params = params
        # run_setting
        self.after_pred_func = run_setting.get("after_pred_func")
        # cv_setting
        self.target_col = cv_setting.get("target_col") # str
        self.key_cols = cv_setting.get("key_cols") # list
        self.initial_fold_date = cv_setting.get("initial_fold_date")
        # データのセット
        self.df_main = df_main
        self.out_dir_name = os.path.join(DIR_MODEL, run_name)


    def metric(self, va_true, va_pred):
        """評価指標の計算
        """
        score = mean_absolute_error(va_true, va_pred)
        # score = math.sqrt(mean_squared_error(va_true, va_pred))
        
        return score


    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # run名、i_fold、モデルのクラス名からモデルを作成する
        i_fold_str = i_fold.strftime('%Y-%m-%d')
        run_fold_name = f'{self.run_name}_fold-{i_fold_str}'
        model = self.model_cls(run_fold_name, self.params.copy(), self.logger)
        return model
    
    
    def after_split_process(self, tr, va):
        """データセットの分割後に行う処理
        """
        # target encoding
        # if self.target_encoder is not None:
        #     tr_ = self.target_encoder.fit_transform(tr)
        #     va_ = self.target_encoder.transform(va)
        #     tr = pd.merge(tr, tr_, on=self.key, how='left')
        #     va = pd.merge(va, va_, on=self.key, how='left')

        return tr, va
    

    def craete_train_valid_dateset(self, i_fold):
        """
        foldを指定して訓練・検証データを準備する
        """
        tr = self.df_main[self.df_main['datetime'] < i_fold]
        tr_va_te = self.df_main[self.df_main['datetime'] < i_fold+pd.DateOffset(months=1)]
        va_te = tr_va_te[tr_va_te['datetime'] >= i_fold]
        list_va_date = va_te[va_te["predict"]==2]["datetime"].apply(lambda x: x.date()).unique()
        list_te_date = va_te[va_te["predict"]==1]["datetime"].apply(lambda x: x.date()).unique()
        # predictを削除
        tr = tr.drop("predict", axis=1)
        tr_va_te = tr_va_te.drop("predict", axis=1)

        return tr, tr_va_te, list_va_date, list_te_date

    
    def train_fold(self, i_fold, metrics=None) -> Tuple[Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """foldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）, metrics: 評価に用いる関数
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """

        # データセットの準備
        tr, _, _, _ = self.craete_train_valid_dateset(i_fold)
        
        # 学習を行う
        model = self.build_model(i_fold)
        model.train(tr)

        return model

    def metric_fold(self, i_fold):
        """
        foldを指定して評価を行う
        """

        # データセットの準備
        _, tr_va_te, list_va_date, _  = self.craete_train_valid_dateset(i_fold)

        # 学習済みモデル
        model = self.build_model(i_fold)
        model.load_model()
        
        # 検証データの予測
        list_va_true = []
        list_va_pred = []
        # 日付毎に予測
        for va_date in list_va_date:
            va_datetime = pd.to_datetime(va_date)
            va = tr_va_te[tr_va_te["datetime"]<=va_datetime.replace(hour=0)]
            va_true = tr_va_te[(tr_va_te["datetime"]>=va_datetime.replace(hour=1))&(tr_va_te["datetime"]<=va_datetime.replace(hour=23))]
            va_true = va_true[self.key_cols + [self.target_col]]
            va_pred = model.predict(va)
            list_va_true.append(va_true)
            list_va_pred.append(va_pred)

        df_va_true = pd.concat(list_va_true, axis=0).sort_values(self.key_cols)
        df_va_pred = pd.concat(list_va_pred, axis=0).sort_values(self.key_cols)

        # 後処理
        df_va_pred = self.after_pred_func(df_va_pred, self.target_col)

        # 欠損値補完前の目的変数がNaNの行を削除
        target_data = pd.read_pickle(os.path.join(DIR_INTERIM, "df_target_all.pkl"))[self.key_cols + [self.target_col]]
        target_data = target_data.rename(columns={self.target_col: "target"})
        df_va_true = pd.merge(df_va_true, target_data, on=self.key_cols, how="inner")
        df_va_pred = pd.merge(df_va_pred, target_data, on=self.key_cols, how="inner")
        df_va_true = df_va_true.dropna(subset=["target"])
        df_va_pred = df_va_pred.dropna(subset=["target"])
        df_va_pred = df_va_pred.drop(columns=["target"])
        df_va_true = df_va_true.drop(columns=["target"])
        
        # バリデーションデータの評価
        va_score = self.metric(df_va_true[self.target_col].values, df_va_pred[self.target_col].values)

        return va_score, df_va_pred
    
    def predict_fold(self, i_fold: Union[int, str]):
        """foldを指定して予測を行う"""

        # データセットの準備
        _, tr_va_te, _, list_te_date  = self.craete_train_valid_dateset(i_fold)

        # 学習済みモデル
        model = self.build_model(i_fold)
        model.load_model()

        # 予測
        list_te_pred = []
        for te_date in list_te_date:
            te_datetime = pd.to_datetime(te_date)
            te = tr_va_te[tr_va_te["datetime"]<=te_datetime.replace(hour=0)]
            te_pred = model.predict(te)
            list_te_pred.append(te_pred)
        df_te_pred = pd.concat(list_te_pred, axis=0).sort_values(self.key_cols)

        # 後処理
        df_te_pred = self.after_pred_func(df_te_pred, self.target_col)

        return df_te_pred.sort_values(self.key_cols) 
    

    def get_cv_folds(self):
        """CVのfoldを返却
        """
        return pd.date_range(self.initial_fold_date, periods=12, freq='MS')


    def run_train_cv(self) -> None:
        """CVでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        12ヶ月分の学習と評価を行う
        12ヶ月
        """
        # ログ
        self.logger.info(f'{self.run_name} - start training cv')

        # fold毎の学習：train_foldをn_splits回繰り返す
        for i_fold in self.get_cv_folds():

            self.logger.info(f'{self.run_name} fold {i_fold.date()} - start training')
            # 学習を行う
            model = self.train_fold(i_fold)
            # モデルを保存する
            model.save_model()
            self.logger.info(f'{self.run_name} fold {i_fold.date()} - end training')

        # パラメータの保存
        path_output = os.path.join(self.out_dir_name, f'params.yaml')
        Util.jump_json(self.params, path_output)

        self.logger.info(f'{self.run_name} - end training cv')


    def run_metric_cv(self):
        """
        CVでの評価を行う
        """
        self.logger.info(f'{self.run_name} - start metric cv')

        scores = [] # 各foldのscoreを保存
        preds = [] # 各foldの予測値を保存

        # fold毎の検証データの予測・評価
        for i_fold in tqdm(self.get_cv_folds()):
            # 評価を行う
            score, df_va_pred = self.metric_fold(i_fold)
            # 結果を保持する
            scores.append(score)
            preds.append(df_va_pred)
        
        df_va_preds = pd.concat(preds, axis=0)

        # 評価結果の保存
        self.logger.result(f"memo: {self.memo}")
        self.logger.result_scores(self.run_name, scores)
        self.logger.result(f"mean: {np.mean(scores)}, std: {np.std(scores)}")
        self.logger.info(f"mean: {np.mean(scores)}, std: {np.std(scores)}")
        # 予測結果の保存
        path_output = os.path.join(self.out_dir_name, f'va_pred.pkl')
        Util.dump_df_pickle(df_va_preds, path_output)
        self.logger.info(f'output predict : {path_output}')

        self.logger.info(f'{self.run_name} - end metric cv')


    def run_predict_cv(self) -> None:
        """CVでテストデータの予測を行う
        """
        self.logger.info(f'{self.run_name} - start prediction cv')
        te_preds = []

        # fold毎のテストデータの予測
        for i_fold in tqdm(self.get_cv_folds()):
            pred = self.predict_fold(i_fold)
            te_preds.append(pred)
        df_te_preds = pd.concat(te_preds, axis=0)
        
        # 予測結果の保存
        path_output = os.path.join(self.out_dir_name, f'te_pred.pkl')
        Util.dump_df_pickle(df_te_preds, path_output)
        self.logger.info(f'output predict : {path_output}')

        self.logger.info(f'{self.run_name} - end prediction cv')


###################################################################################
 
    
class MLModelRunner(TimeseriesModelRunner):
    """学習・予測・評価・パラメータチューニングを担うクラス
    """

    # コンストラクタ
    def __init__(self,
                 run_name: str, # runの名前
                 model_cls: Callable[[str, dict], Model], #モデルのクラス
                 params: dict, # ハイパーパラメータ
                 df_main: pd.DataFrame, # 学習データ
                 run_setting: dict,
                 cv_setting: dict,
                 logger,
                 memo,
                 ): 
        super().__init__(run_name, model_cls, params, df_main, run_setting, cv_setting, logger, memo)
        # self.calc_shap = run_setting.get('calc_shap')
        # self.save_train_pred = run_setting.get('save_train_pred')
        # self.tune_params = run_setting.get('tune_params')
        # self.target_encoder = run_setting.get("target_encoder")
        # if self.calc_shap:
        #     self.shap_values = np.zeros(self.train_x.shape)

    
    
    def after_split_process(self, tr, va, te):
        """データセットの分割後に行う処理
        """
        # target encoding
        # if self.target_encoder is not None:
        #     tr_ = self.target_encoder.fit_transform(tr)
        #     va_ = self.target_encoder.transform(va)
        #     tr = pd.merge(tr, tr_, on=self.key, how='left')
        #     va = pd.merge(va, va_, on=self.key, how='left')

        return tr, va, te
    
    def create_train_valid_dateset(self, i_fold: Union[int, str]):
        """foldを指定して訓練・検証データを準備する
        return:
            tr: i_fold以前のデータ
            va: i_fild~i_fold+1期の検証対象日の0時~23時のデータ
            te: i_fild~i_fold+1期のテスト対象日の0時~23時のデータ
        """
        # データセットの準備
        # ex) i_fold: 2014-09
        # 0時のデータについて1期先のpredictの値を代入する
        data = self.df_main.copy()
        for station_id in data["station_id"].unique():
            df_station = data[data["station_id"] == station_id].copy()
            df_station["predict_term1"] = df_station["predict"].shift(-1)
            df_station["predict"] = df_station.apply(lambda x: x["predict_term1"] if x["datetime"].hour == 0 else x["predict"], axis=1)    
            df_station.drop(columns=["predict_term1"], inplace=True)
            data[data["station_id"] == station_id] = df_station
        # 学習データ・バリデーションデータ、テストデータに分割
        tr = data[data['datetime'] < i_fold]
        tr_va_te = data[data['datetime'] < i_fold+pd.DateOffset(months=1)]
        va_te = tr_va_te[tr_va_te['datetime'] >= i_fold]       
        va = va_te[va_te["predict"]==2]
        te = va_te[va_te["predict"]==1]
        # データセットの分割後に行う処理
        tr, va, te = self.after_split_process(tr, va, te)

        return tr, va, te
        

    def train_fold(self, i_fold, metrics=None):
        """foldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）, metrics: 評価に用いる関数
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """

        # データセットの準備
        tr, _, _ = self.create_train_valid_dateset(i_fold)

        # パラメータチューニングを行う
        # tr_tr_x, tr_tr_y, tr_va_x, tr_va_y = split(tr_x, tr_y, va_x, va_y)
        # if self.tune_params:
        #     self.tune_param(tr_tr_x, tr_tr_y, tr_va_x, tr_va_y)
        
        # 学習を行う
        model = self.build_model(i_fold)
        model.train(tr)

        return model

    def metric_fold(self, i_fold):
        """foldを指定して評価を行う"""

        # データセットの準備
        _, va, _  = self.create_train_valid_dateset(i_fold)

        print(va.shape)

        # 予測値
        va_0 = va[va["datetime"].dt.hour==0]
        model = self.build_model(i_fold)
        model.load_model()
        df_va_pred = model.predict(va_0)

        # 後処理
        df_va_pred = self.after_pred_func(df_va_pred, self.target_col)

        # 正解データの作成
        df_va_true = va[va["datetime"].dt.hour!=0][self.key_cols + [self.target_col]]

        # バリデーションデータの評価
        df_va_pred = df_va_pred.sort_values(self.key_cols)
        df_va_true = df_va_true.sort_values(self.key_cols)

        # 欠損値補完前の目的変数がNaNの行を削除
        target_data = pd.read_pickle(os.path.join(DIR_INTERIM, "df_target_all.pkl"))[self.key_cols + [self.target_col]]
        target_data = target_data.rename(columns={self.target_col: "target"})
        df_va_true = pd.merge(df_va_true, target_data, on=self.key_cols, how="inner")
        df_va_pred = pd.merge(df_va_pred, target_data, on=self.key_cols, how="inner")
        df_va_true = df_va_true.dropna(subset=["target"])
        df_va_pred = df_va_pred.dropna(subset=["target"])
        df_va_pred = df_va_pred.drop(columns=["target"])
        df_va_true = df_va_true.drop(columns=["target"])

        score = self.metric(df_va_true[self.target_col].values, df_va_pred[self.target_col].values)

        return score, df_va_pred
    
    def predict_fold(self, i_fold):
        """foldを指定して予測を行う"""

        # データセットの準備
        _, _, te = self.create_train_valid_dateset(i_fold)

        # 予測値
        te_x = te[te["datetime"].dt.hour==0]
        model = self.build_model(i_fold)
        model.load_model()
        df_te_pred = model.predict(te_x)

        # 後処理
        df_te_pred = self.after_pred_func(df_te_pred, self.target_col)

        return df_te_pred.sort_values(self.key_cols) 


    def plot_feature_importance_term(self, list_list_df_importance, term, ax1) -> None:      
        """cvのterm毎の特徴量の重要度をプロットする
        """ 
        # 各foldの指定したtermの特徴量の重要度を取得
        list_importance = []
        for list_df_importance in list_list_df_importance:
            importance = list_df_importance[term-1]["importance"].values
            list_importance.append(importance)

        df_importance = pd.DataFrame({
            "feature": list_list_df_importance[0][term-1]["feature"],
            "mean": np.mean(list_importance, axis=0),
            "std": np.std(list_importance, axis=0)
        })

        df = df_importance
        # 変動係数を算出
        df['coef_of_var'] = df['std'] / df['mean']
        df['coef_of_var'] = df['coef_of_var'].fillna(0)
        df = df.sort_values('mean', ascending=True)

        # 棒グラフを出力
        ax1.set_title(f'feature importance gain term {term}')
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

        


    def plot_feature_importance_cv(self) -> None:
        """CVで学習した各foldのモデルの平均により、特徴量の重要度を取得する
        """
        self.logger.info(f'{self.run_name} - start plot feature importance cv')

        # 各foldの特徴量の重要度を取得
        list_list_df_importance = []
        for i_fold in self.get_cv_folds():
            model = self.build_model(i_fold)
            model.load_model()
            list_list_df_importance.append(model.get_feature_importance())

        fig = plt.figure(figsize = (100, 30))
        for term in range(1, 24):
            ax = fig.add_subplot(1, 24, term)
            self.plot_feature_importance_term(list_list_df_importance, term, ax)
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        plt.tight_layout()
        # 図を保存
        path_output = os.path.join(self.out_dir_name, f'fi_gain.png')
        plt.savefig(path_output, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f'{self.run_name} - end plot feature importance cv')




####### model utils ##################################################################
    

    def tune_param(self, tr_x, tr_y, va_x, va_y):
        """パラメータチューニングを行う
        """
        # optunaによるパラメータ探索の実行
        # パラメータ探索の範囲
        def objective(trial):
            params = self.params.copy()
            params['num_leaves'] = trial.suggest_int('num_leaves', 2, 256)
            params['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-8, 1.0)
            params['subsample'] = trial.suggest_float('subsample', 1e-8, 1.0)
            params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 2, 256)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-8, 1.0)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-8, 1.0)
            model = self.model_cls(self.run_name, params)
            model.train(tr_x, tr_y, va_x, va_y)
            va_pred = model.predict(va_x)
            score = self.metrics(va_y, va_pred)
            return score
        self.logger.info(f'{self.run_name} - start tuning')
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        for key, value in study.best_params.items():
            self.params[key] = value
        self.logger.info(f'{self.run_name} - end tuning')
