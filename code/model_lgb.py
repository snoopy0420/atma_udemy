import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import shap
import lightgbm as lgb
from model import Model
from util import Util

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.safe_load(file)
FIGURE_DIR_NAME = yml['SETTING']['FIGURE_DIR_NAME']

# 各foldのモデルを保存する配列
model_array = []
evals_array = []

class ModelLGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        dtrain = lgb.Dataset(tr_x, tr_y)
        validation = va_x is not None
        if validation:
            dvalid = lgb.Dataset(va_x, va_y)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_boost_round')
        verbose_eval = params.pop('verbose_eval')

        # 学習
        if validation:
            evals_result = {}
            early_stopping_rounds = params.pop('early_stopping_rounds')
            self.model = lgb.train(
                                params,
                                dtrain,
                                num_boost_round=num_round,
                                valid_sets=(dtrain, dvalid),
                                valid_names=("train", "eval"),
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=verbose_eval,
                                evals_result=evals_result,
                                )
            model_array.append(self.model)
            evals_array.append(evals_result)

        else:
            self.model = lgb.train(
                                params,
                                dtrain,
                                # num_boost_round=num_round
                                )
            model_array.append(self.model)


    # shapを計算しないver
    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)


    # shapを計算するver うまくいかない
    def predict_and_shap(self, te_x, shap_sampling):
        fold_importance = shap.TreeExplainer(self.model).shap_values(te_x[:shap_sampling])
        valid_prediticion = self.model.predict(te_x, num_iteration=self.model.best_iteration)
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
        eval_metiric = "l1"
        print(evals_array[0])

        # 学習過程の可視化、foldが４以上の時のみ
        fig, axes = plt.subplots(2, 2, figsize=(12,8))
        plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
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
        """feature importanceの計算,図の保存
        """
        val_gain = model_array[0].feature_importance(importance_type='gain') # gainの計算
        val_gain = pd.Series(val_gain)

        for m in model_array[1:]:
            s = pd.Series(m.feature_importance(importance_type='gain'))
            val_gain = pd.concat([val_gain, s], axis=1)

        # 各foldの平均を算出
        val_mean = val_gain.mean(axis=1)
        val_mean = val_mean.values
        importance_df_mean = pd.DataFrame(val_mean, index=features, columns=['importance']).sort_values('importance')

        # 各foldの標準偏差を算出
        val_std = val_gain.std(axis=1)
        val_std = val_std.values
        importance_df_std = pd.DataFrame(val_std, index=features, columns=['importance']).sort_values('importance')

        # マージ
        df = pd.merge(importance_df_mean, importance_df_std, left_index=True, right_index=True ,suffixes=['_mean', '_std'])

        # 変動係数を算出
        df['coef_of_var'] = df['importance_std'] / df['importance_mean']
        df['coef_of_var'] = df['coef_of_var'].fillna(0)
        df = df.sort_values('importance_mean', ascending=True)

        # 出力
        fig, ax1 = plt.subplots(figsize = (10, 30))
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
