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


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.safe_load(file)
FIGURE_DIR_NAME = yml['SETTING']['DIR_FIGURE']
DIR_HOME = yml['SETTING']['DIR_HOME']
DIR_MODEL = yml['SETTING']['DIR_MODEL']
DIR_FIGURE = yml['SETTING']['DIR_FIGURE']
TARGET_COL = yml['SETTING']['TARGET_COL']
REMOVED_COL = yml['SETTING']['REMOVE_COLS']

sys.path.append(DIR_HOME)
from src.model import Model
from src.util import Util, Metric, Validation


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
        model = self.model_cls(run_fold_name, self.params, self.logger)
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
        # predict=1のデータについて過去24時間分のデータを取得
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

    def metric_fold(self, i_fold, metrics=None):
        """
        foldを指定して評価を行う
        """

        # データセットの準備
        _, tr_va_te, list_va_date, _  = self.craete_train_valid_dateset(i_fold)

        # 学習済みモデル
        model = self.build_model(i_fold)
        model.load_model()

        # 検証・テスト期間の予測
        # list_va_te_date = tr_va_te["datetime"].apply(lambda x: x.date()).unique()
        # list_va_te_true = []
        # list_va_te_pred = []
        # for va_te_date in list_va_te_date:
        #     va_te_datetime = pd.to_datetime(va_te_date)
        #     va_te = tr_va_te[tr_va_te["datetime"]<=va_te_datetime.replace(hours=0)]
        #     va_te_true = tr_va_te[(tr_va_te["datetime"]>=va_te_datetime.replace(hours=1))&(tr_va_te["datetime"]<=va_te_datetime.replace(hour=23))]
        #     va_te_true = va_te_true[self.key_cols + [self.target_col]]
        #     va_te_pred = model.predict(va_te)
        #     list_va_te_true.append(va_te_true)  
        #     list_va_te_pred.append(va_te_pred)
        
        # # 検証データの予測
        list_va_true = []
        list_va_pred = []
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

        # テストデータの予測
        list_te_pred = []
        for te_date in list_te_date:
            te_datetime = pd.to_datetime(te_date)
            te = tr_va_te[tr_va_te["datetime"]<=te_datetime.replace(hour=0)]
            te_pred = model.predict(te)
            list_te_pred.append(te_pred)

        df_te_pred = pd.concat(list_te_pred, axis=0).sort_values(self.key_cols)

        return df_te_pred
    

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




####### model utils ##################################################################


    def shap_feature_importance(self) -> None:
        """計算したshap値を可視化して保存する
        """
        all_columns = self.train_x.columns.values.tolist() + [self.target]
        ma_shap = pd.DataFrame(sorted(zip(abs(self.shap_values).mean(axis=0), all_columns), reverse=True),
                        columns=['Mean Abs Shapley', 'Feature']).set_index('Feature')
        ma_shap = ma_shap.sort_values('Mean Abs Shapley', ascending=True)

        # 可視化
        fig = plt.figure(figsize = (8,30))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        ax = fig.add_subplot(1,1,1)
        ax.set_title('shap value')
        ax.barh(ma_shap.index, ma_shap['Mean Abs Shapley'] , label='Mean Abs Shapley',  align="center", alpha=0.8)
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=0, fontsize=10)
        ax.legend(loc = 'upper left')
        plt.savefig(FIGURE_DIR_NAME + self.run_name + '_shap.png', dpi=300, bbox_inches="tight")
        plt.close()



    def get_feature_name(self):
        """ 学習に使用した特徴量を返却
        """
        return self.train_x.columns.values.tolist()

    def get_params(self):
        """ 学習に使用したハイパーパラメータを返却
        """
        return self.params
    

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

###################################################################################
 
    
class MLModelRunner:
    """学習・予測・評価・パラメータチューニングを担うクラス
    """

    # コンストラクタ
    def __init__(self,
                 run_name: str, # runの名前
                 model_cls: Callable[[str, dict], Model], #モデルのクラス
                 params: dict, # ハイパーパラメータ
                 df_train: pd.DataFrame, # 学習データ
                 df_test: pd.DataFrame, # テストデータ
                 run_setting: dict,
                 logger,
                 memo,
                 ): 
        
        self.memo = memo
        self.logger = logger
        self.run_name = run_name
        self.model_cls = model_cls
        self.params = params
        self.key = run_setting.get('key')
        self.calc_shap = run_setting.get('calc_shap')
        self.save_train_pred = run_setting.get('save_train_pred')
        self.tune_params = run_setting.get('tune_params')
        self.target_encoder = run_setting.get("target_encoder")
        # カラム
        self.target_col = run_setting.get("target_col") # str
        self.key_cols = run_setting.get("key_cols") # list
        # データのセット
        self.df_train = df_train
        self.df_test = df_test
        self.out_dir_name = DIR_MODEL

        if self.calc_shap:
            self.shap_values = np.zeros(self.train_x.shape)

        self.initial_date = "2014-09-01"


    def metric(self, va, va_pred):
        """評価指標の計算
        """
        score = mean_absolute_error(va[self.target_col], va_pred)
        return score


    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # run名、i_fold、モデルのクラス名からモデルを作成する
        run_fold_name = f'{self.run_name}-fold{i_fold}'
        model = self.model_cls(run_fold_name, self.params)
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
    

    def craete_train_valid_dateset(self, i_fold: Union[int, str]):
        """foldを指定して訓練・検証データを準備する
        """
        # データセットの準備
        # 学習データ・バリデーションデータ、テストデータに分割
        tr = self.df_train[self.df_train['datetime'] < i_fold]
        te = self.df_train[self.df_train['datetime'] == i_fold]
        va = te[te["predict"]==2]
        te = te[te["predict"]==1]

        # データセットの分割後に行う処理
        tr, va, te = self.after_split_process(tr, va, te)

        return tr, va, te
        

    
    def train_fold(self, i_fold: Union[int, str], metrics=None) -> Tuple[Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """foldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）, metrics: 評価に用いる関数
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """

        # データセットの準備
        tr, va, te = self.crete_train_valid_dateset(i_fold)

        # パラメータチューニングを行う
        # tr_tr_x, tr_tr_y, tr_va_x, tr_va_y = split(tr_x, tr_y, va_x, va_y)
        # if self.tune_params:
        #     self.tune_param(tr_tr_x, tr_tr_y, tr_va_x, tr_va_y)
        
        # 学習を行う
        model = self.build_model(i_fold)
        model.train(tr, va)

        # モデル、インデックス、予測値、評価を返す
        return model

    def metric_fold(self, i_fold: Union[int, str], metrics=None):
        """foldを指定して評価を行う"""

        # データセットの準備
        _, va, _  = self.crete_train_valid_dateset(i_fold)

        # 予測値
        model = self.build_model(i_fold)
        model.load_model()
        va_pred = model.predict(va)

        # バリデーションデータの評価
        score = self.metrics(va, va_pred)

        return score
    
    def predict_fold(self, i_fold: Union[int, str]):
        """foldを指定して予測を行う"""

        # データセットの準備
        tr, va, te = self.crete_train_valid_dateset(i_fold)

        # 予測値
        model = self.build_model(i_fold)
        model.load_model()
        pred = model.predict(te)

        return pred


    def run_train_cv(self) -> None:
        """CVでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        12ヶ月分の学習と評価を行う
        12ヶ月
        """
        # ログ
        self.logger.info(f'{self.run_name} - start training cv')

        # fold毎の学習：train_foldをn_splits回繰り返す
        for i_fold in pd.date_range(self.initial_date, periods=12, freq='M'):

            # 学習を行う
            self.logger.info(f'{self.run_name} fold {i_fold} - start training')
            model = self.train_fold(i_fold)
            self.logger.info(f'{self.run_name} fold {i_fold} - end training')

            # モデルを保存する
            model.save_model()

        self.logger.info(f'{self.run_name} - end training cv')


    def run_metric_cv(self):
        # ログ
        self.logger.info(f'{self.run_name} - start metric cv')

        scores = [] # 各foldのscoreを保存

        # fold毎の学習：train_foldをn_splits回繰り返す
        for i_fold in pd.date_range(self.initial_date, periods=12, freq='M'):

            score = self.metric_fold(i_fold)

            # 結果を保持する
            scores.append(score)

        # 評価結果の保存
        self.logger.result(f"memo: {self.memo}")
        self.logger.result_scores(self.run_name, scores)
        self.logger.result(f"mean: {np.mean(scores)}, std: {np.std(scores)}")

        self.logger.info(f'{self.run_name} - end metric cv')


    def run_predict_cv(self) -> None:
        """CVで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """
        self.logger.info(f'{self.run_name} - start prediction cv')
        preds = []

        for i_fold in pd.date_range(self.initial_date, periods=12, freq='M'):
            self.logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            pred = self.predict_fold(i_fold)
            preds.append(pred)
            self.logger.info(f'{self.run_name} - end prediction fold:{i_fold}')
        df_preds = pd.concat(preds, axis=0)
        
        # 予測結果の保存
        path_output = os.path.join(self.out_dir_name, f'{self.run_name}-pred.pkl')
        Util.dump_df_pickle(df_preds, path_output)
        self.logger.info(f'output predict : {path_output}')

        self.logger.info(f'{self.run_name} - end prediction cv')


    def plot_feature_importance_cv(self) -> None:
        """CVで学習した各foldのモデルの平均により、特徴量の重要度を取得する
        """
        list_feat_imp = []
        for i_fold in range(self.n_splits):
            model = self.build_model(i_fold)
            model.load_model()
            list_feat_imp.append(model.get_feature_importance())
        df_feat_imp = pd.concat([pd.Series(feat_imp) for feat_imp in list_feat_imp], axis=1)

        # 各foldの平均を算出
        # 各foldの標準偏差を算出
        df_feat_imp_ = pd.DataFrame({
            'feature': self.df_test.drop(columns=REMOVED_COL).columns,
            'mean': df_feat_imp.mean(axis=1),
            'std': df_feat_imp.std(axis=1)
        }).sort_values('mean')

        df = df_feat_imp_
        # 変動係数を算出
        df['coef_of_var'] = df['std'] / df['mean']
        df['coef_of_var'] = df['coef_of_var'].fillna(0)
        df = df.sort_values('mean', ascending=True)

        # 出力
        fig, ax1 = plt.subplots(figsize = (10, 30))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        plt.tight_layout()

        # 棒グラフを出力
        ax1.set_title('feature importance gain')
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

        # 図を保存
        path_output = os.path.join(DIR_FIGURE, f'{self.run_name}_fi_gain.png')
        plt.savefig(path_output, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"output feature importance : {path_output}")




####### model utils ##################################################################


    def shap_feature_importance(self) -> None:
        """計算したshap値を可視化して保存する
        """
        all_columns = self.train_x.columns.values.tolist() + [self.target]
        ma_shap = pd.DataFrame(sorted(zip(abs(self.shap_values).mean(axis=0), all_columns), reverse=True),
                        columns=['Mean Abs Shapley', 'Feature']).set_index('Feature')
        ma_shap = ma_shap.sort_values('Mean Abs Shapley', ascending=True)

        # 可視化
        fig = plt.figure(figsize = (8,30))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        ax = fig.add_subplot(1,1,1)
        ax.set_title('shap value')
        ax.barh(ma_shap.index, ma_shap['Mean Abs Shapley'] , label='Mean Abs Shapley',  align="center", alpha=0.8)
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=0, fontsize=10)
        ax.legend(loc = 'upper left')
        plt.savefig(FIGURE_DIR_NAME + self.run_name + '_shap.png', dpi=300, bbox_inches="tight")
        plt.close()



    def get_feature_name(self):
        """ 学習に使用した特徴量を返却
        """
        return self.train_x.columns.values.tolist()

    def get_params(self):
        """ 学習に使用したハイパーパラメータを返却
        """
        return self.params
    

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
