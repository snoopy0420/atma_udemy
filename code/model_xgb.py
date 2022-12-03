import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import shap
import xgboost as xgb
from model import Model
from util import Util

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.safe_load(file)
FIGURE_DIR_NAME = yml['SETTING']['FIGURE_DIR_NAME']

# 各foldのモデルを保存する配列
model_array = []
evals_array = []

class ModelXGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        validation = va_x is not None        
        if validation:
            dvalid = xgb.DMatrix(va_x, label=va_y)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')
        verbose = params.pop('verbose')

        # 学習
        if validation:
            evals_result = {}
            early_stopping_rounds = params.pop('early_stopping_rounds')
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = xgb.train(
                params,
                dtrain,
                num_round,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose,
                evals_result=evals_result,
                )
            model_array.append(self.model)
            evals_array.append(evals_result)

        else:
            watchlist = [(dtrain, 'train')]
            self.model = xgb.train(
                params,
                dtrain, 
                num_round, 
                evals=watchlist
                )
            model_array.append(self.model)

    # shapを計算しないver
    def predict(self, te_x):
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)

     # shapを計算するver
    def predict_and_shap(self, te_x, shap_sampling):
        fold_importance = shap.TreeExplainer(self.model).shap_values(te_x[:shap_sampling])
        dtest = xgb.DMatrix(te_x)
        valid_prediticion = self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)
        return valid_prediticion, fold_importance


    def save_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)


    def load_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)


    @classmethod
    def plot_learning_curve(self, run_name):
        """学習過程の可視化、foldが４以上の時のみ
        """
        eval_metiric = "mae"
        print(evals_array[0]) # eval_metiricを確認

        fig, axes = plt.subplots(2, 2, figsize=(12,8))
        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.title('Learning curve')
        for i, ax in enumerate(axes.ravel()):
            ax.plot(evals_array[i]['train'][eval_metiric][10:], label="train")
            ax.plot(evals_array[i]['eval'][eval_metiric][10:], label="valid")
            ax.set_xlabel('epoch')
            ax.set_ylabel(eval_metiric)
            ax.legend()
            ax.grid(True)

        plt.savefig(FIGURE_DIR_NAME + run_name + '_lcurve.png', dpi=300, bbox_inches="tight")
        plt.close()
    


    @classmethod
    def calc_feature_importance(self, dir_name, run_name, features):
        """feature importanceの計算
        """
        # 各foldのfeature importanceを取り出し,dfにまとめる
        val_gain = model_array[0].get_score(importance_type='total_gain')
        val_gain = pd.DataFrame(val_gain.values(), index=val_gain.keys())
        for m in model_array[1:]:
            s = m.get_score(importance_type='total_gain')
            df = pd.DataFrame(s.values(), index=s.keys())
            val_gain = pd.merge(val_gain, df, how='outer', left_index=True, right_index=True)

        # 各foldの平均を算出
        importance_df_mean = pd.DataFrame(val_gain.mean(axis=1), columns=['importance']).fillna(0)
        # 各foldの標準偏差を算出
        importance_df_std = pd.DataFrame(val_gain.std(axis=1), columns=['importance']).fillna(0)

        # マージ
        df = pd.merge(importance_df_mean, importance_df_std, left_index=True, right_index=True ,suffixes=['_mean', '_std'])

        # 変動係数を算出
        df['coef_of_var'] = df['importance_std'] / df['importance_mean']
        df['coef_of_var'] = df['coef_of_var'].fillna(0)
        df = df.sort_values('importance_mean', ascending=True)

        # 出力
        fig, ax1 = plt.subplots(figsize = (10,30))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
        plt.tight_layout()

        # 棒グラフを出力
        ax1.set_title('feature importance gain')
        ax1.set_xlabel('feature importance mean & std')
        ax1.barh(df.index, df['importance_mean'], label='importance_mean',  align="center", alpha=0.6)
        ax1.barh(df.index, df['importance_std'], label='importance_std',  align="center", alpha=0.6)

        # 折れ線グラフを出力
        ax2 = ax1.twiny()
        ax2.plot(df['coef_of_var'], df.index, linewidth=1, color="crimson", marker="o", markersize=8, label='coef_of_var')
        ax2.set_xlabel('Coefficient of variation')

        # 凡例を表示（グラフ左上、ax2をax1のやや下に持っていく）
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=12)
        ax2.legend(bbox_to_anchor=(1, 0.93), loc='upper right', borderaxespad=0.5, fontsize=12)

        # グリッド表示(ax1のみ)
        ax1.grid(True)
        ax2.grid(False)

        plt.savefig(FIGURE_DIR_NAME + run_name + '_fi_gain.png', dpi=300, bbox_inches="tight")
        plt.close()

