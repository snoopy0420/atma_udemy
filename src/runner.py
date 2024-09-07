import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os
import shap
import yaml
from tqdm import tqdm, tqdm_notebook
# from sklearn.metrics import mean_absolute_error
from typing import Callable, List, Optional, Tuple, Union
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import optuna

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


# 定数
shap_sampling = 10000

class Runner:
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
        self.metrics = roc_auc_score
        self.run_name = run_name
        self.model_cls = model_cls
        self.params = params
        self.key = run_setting.get('key')
        self.calc_shap = run_setting.get('calc_shap')
        self.save_train_pred = run_setting.get('save_train_pred')
        self.tune_params = run_setting.get('tune_params')
        self.target_encoder = run_setting.get("target_encoder")
        # CVの設定
        self.n_splits = 5
        self.random_state = 2024
        self.shuffle = True
        # データのセット
        self.df_train = df_train
        self.df_test = df_test
        self.out_dir_name = DIR_MODEL

        if self.calc_shap:
            self.shap_values = np.zeros(self.train_x.shape)
        # if self.hopt != False:
        #     # self.paramsを上書きする
        #     self.run_hopt()
        #     self.shap_values = np.zeros(self.train_x.shape)


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
        if self.target_encoder is not None:
            tr_ = self.target_encoder.fit_transform(tr)
            va_ = self.target_encoder.transform(va)
            tr = pd.merge(tr, tr_, on=self.key, how='left')
            va = pd.merge(va, va_, on=self.key, how='left')

        return tr, va

    
    def train_fold(self, i_fold: Union[int, str], metrics=None) -> Tuple[Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """foldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）, metrics: 評価に用いる関数
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """

        # データセットの準備
        # 学習データ・バリデーションデータのindexを取得
        tr_idx, va_idx = Validation.load_index_k_fold(i_fold, self.df_train, self.n_splits, self.shuffle, self.random_state)
        # 学習データ・バリデーションデータをセットする
        tr = self.df_train.iloc[tr_idx]
        va = self.df_train.iloc[va_idx]

        # データセットの分割後に行う処理
        tr, va = self.after_split_process(tr, va)
        
        # データセットの分割
        tr_x = tr.drop(columns=[TARGET_COL]+REMOVED_COL)
        tr_y = tr[TARGET_COL]
        va_x = va.drop(columns=[TARGET_COL]+REMOVED_COL)
        va_y = va[TARGET_COL]

        # パラメータチューニングを行う
        if self.tune_params:
            self.tune_param(tr_x, tr_y, va_x, va_y)
        
        # 学習を行う
        model = self.build_model(i_fold)
        model.train(tr_x, tr_y, va_x, va_y)

        # バリデーションデータの予測
        if self.calc_shap:
            va_pred, self.shap_values[va_idx[:shap_sampling]] = model.predict_and_shap(va_x, shap_sampling)
        else:
            va_pred = model.predict(va_x)

        # バリデーションデータの評価
        score = self.metrics(va_y, va_pred)

        # モデル、インデックス、予測値、評価を返す
        return model, va_idx, va_pred, score

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
        


    def run_train_cv(self) -> None:
        """CVでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        # ログ
        self.logger.info(f'{self.run_name} - start training cv')

        scores = [] # 各foldのscoreを保存
        va_idxes = [] # 各foldのvalidationデータのindexを保存
        va_preds = [] # 各foldのバリデーションデータに対する予測値を保存

        # fold毎の学習：train_foldをn_splits回繰り返す
        for i_fold in range(self.n_splits):

            # 学習を行う
            self.logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            self.logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model()

            # 結果を保持する
            scores.append(score)
            va_idxes.append(va_idx)
            va_preds.append(va_pred)

        self.logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}') # スコアを出力

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        va_preds = np.concatenate(va_preds, axis=0)
        va_preds = va_preds[order]

        # 学習データでの予測結果の保存
        if self.save_train_pred:
            Util.dump_df_pickle(pd.DataFrame(va_preds), self.out_dir_name + f'{self.run_name}-train_preds.pkl')

        # 評価結果の保存
        self.logger.result(f"memo: {self.memo}")
        self.logger.result_scores(self.run_name, scores)
        self.logger.result(f"mean: {np.mean(scores)}, std: {np.std(scores)}")

        # shap feature importanceデータの保存
        if self.calc_shap:
            self.shap_feature_importance() # shap値の計算・可視化


    def run_predict_cv(self) -> None:
        """CVで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """
        self.logger.info(f'{self.run_name} - start prediction cv')
        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_splits):
            self.logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(self.df_test.drop(columns=REMOVED_COL))
            preds.append(pred)
            self.logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を算出する
        pred_avg = np.mean(preds, axis=0)
        df_pred = pd.concat([self.df_test[self.key].reset_index(drop=True),
                             pd.DataFrame({"pred":pred_avg}).reset_index(drop=True)], axis=1)

        # 予測結果の保存
        path_output = os.path.join(self.out_dir_name, f'{self.run_name}-pred.pkl')
        Util.dump_df_pickle(df_pred, path_output)

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
