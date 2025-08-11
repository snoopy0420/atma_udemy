import datetime
import logging
import sys,os
import json
import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

sys.path.append(os.path.abspath('..'))
from configs.config import *

# tensorflowとloggingのcollisionに対応
# try:
#     import absl.logging
#     # https://github.com/abseil/abseil-py/issues/99
#     logging.root.removeHandler(absl.logging._absl_handler)
#     # https://github.com/abseil/abseil-py/issues/102
#     absl.logging._warn_preinit_stderr = False
# except Exception:
#     pass

class Util:

    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
    
    @classmethod
    def jump_json(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(value, f, indent=4)

    @classmethod
    def dump_df_pickle(cls, df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_pickle(path)

    @classmethod
    def load_df_pickle(cls, path):
        return pd.read_pickle(path)
    
    @classmethod
    def load_feature(cls, file_name):
        file_name = file_name + ".pkl"
        return pd.read_pickle(os.path.join(DIR_FEATURE, file_name))


# ログ関連
class Logger:

    def __init__(self, path):
        self.general_logger = logging.getLogger(os.path.join(path, 'general'))
        self.result_logger = logging.getLogger(os.path.join(path, 'result'))
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(path, 'general.log'))
        file_result_handler = logging.FileHandler(os.path.join(path, 'result.log'))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    # 時刻をつけてコンソールとログに出力
    def info(self, message):
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    # 計算結果をコンソールと計算結果用ログに出力
    def result_scores(self, run_name, scores):
        dic = dict()
        dic['run_name'] = run_name
        dic['score_mean'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    def now_string(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])


# submission関連
class Submission:

    @classmethod
    def create_submission(cls, run_name, dir_name, preds):
        """submissionファイルの作成
        """
        logger = Logger(dir_name)
        logger.info(f'{run_name} - start create submission')

        submission = pd.read_csv(os.path.join(DIR_INPUT, FILE_SAMPLE_SUBMISSION))
        submission.loc[:, TARGET_COL] = preds.iloc[:, 0]
        submission.to_csv(os.path.join(DIR_SUBMISSIONS, f'{run_name}_submission.csv'), index=False, header=True)

        logger.info(f'{run_name} - end create submission')


class Validation:

    ## クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す関数
    ## param  
    ## i_fold: foldの番号, return: foldに対応するレコードのインデックス

    @classmethod
    def load_index_k_fold(self, i_fold: int, train_x, n_splits=5, shuffle=True, random_state=54) -> np.array:
        """KFold
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        dummy_x = np.zeros(len(train_x))
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(kf.split(dummy_x))[i_fold]

    @classmethod
    def load_index_gk_fold(self, i_fold, train_x, cv_target_column, n_splits=5, shuffle=True, random_state=54) -> np.array:
        """GroupKFold
        """
        # cv_target_column列により分割する
        group_data = train_x[cv_target_column]
        unique_group_data = group_data.unique()

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        tr_group_idx, va_group_idx = list(kf.split(unique_group_data))[i_fold]
        # unique_group_dataをtrain/valid（学習に使うデータ、バリデーションデータ）に分割する
        tr_groups, va_groups = unique_group_data.iloc[tr_group_idx], unique_group_data.iloc[va_group_idx]

        # 各レコードのgroup_dataがtrain/validのどちらに属しているかによって分割する
        is_tr = group_data.isin(tr_groups)
        is_va = group_data.isin(va_groups)
        tr_x, va_x = train_x[is_tr], train_x[is_va]

        return np.array(tr_x.index), np.array(va_x.index)

    @classmethod
    def load_stratify_or_group_target(self) -> pd.Series:
        """
        groupKFoldで同じグループが異なる分割パターンに出現しないようにデータセットを分割したい対象カラムを取得する
        または、StratifiedKFoldで分布の比率を維持したいカラムを取得する
        :return: 分布の比率を維持したいデータの特徴量
        """
        df = pd.read_pickle(self.feature_dir_name + self.cv_target_column + '_train.pkl')
        return pd.Series(df[self.cv_target_column])

    @classmethod
    def load_index_sk_fold(self, i_fold: int) -> np.array:
        """StratifiedKFold
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        stratify_data = self.load_stratify_or_group_target() # 分布の比率を維持したいデータの対象
        dummy_x = np.zeros(len(stratify_data))
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return list(kf.split(dummy_x, stratify_data))[i_fold]

    @classmethod
    def load_index_custom_ts_fold(self, i_fold, train_x) -> np.array:
        """カスタム時系列バリデーション
        """

        tr_x = train_x[(train_x["year"]!=2021)|((train_x["year"]==2021)&(train_x["month"]<5-i_fold))]
        va_x = train_x[(train_x["year"]==2021)&(train_x["month"]==5-i_fold)]

        return np.array(tr_x.index), np.array(va_x.index)


class Metric:

    @classmethod
    def my_metric(self, y_true, y_pred):
        """
        今回の分析で使用する評価関数、コンペの評価指標に応じて変更する
        """
        # スコアを計算
        result = roc_auc_score(y_true, y_pred)
        return result

    