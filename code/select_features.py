import pandas as pd
import numpy as np
import sys,os
import csv
import yaml
from pathlib import Path
import warnings
from util import Logger

import xgboost as xgb
from runner import Runner
from util import load_index_k_fold

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.load(file)
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']  # 生成した特徴量の出力場所
TARGET = yml['SETTING']['TARGET'] # 目的変数
REMOVE_COLS = yml['SETTING']['REMOVE_COLS']


## trainとtestの特徴量選択する関数を定義
def select_by_xgb(train, test, n_splits=5, num_feat=50):
    """Xgboostによる特徴量選択"""

    train_x = train.drop(columns=REMOVE_COLS)
    train_y = train[TARGET]

    # xgbパラメータを設定する
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eta': 0.1,
        'gamma': 0.0,
        'alpha': 0.0,
        'lambda': 1.0,
        'min_child_weight': 1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 87,
    }

    # 各foldの重要度を算出する
    gain_df = pd.DataFrame(index=train_x.columns) # gainの格納場所
    for i_fold in range(n_splits):
        # 学習データと検証データに分割
        tr_idx, va_idx = load_index_k_fold(i_fold, train_x, n_splits=n_splits, random_state=42) # 分割indexを取得
        tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
        va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        # 学習
        model = xgb.train(
            params,
            dtrain,
            5000,
            evals=[(dtrain, 'train'), (dvalid, 'eval')],
            early_stopping_rounds=100,
            verbose_eval=False)
        # gainの取り出し、格納
        gain = model.get_score(importance_type='total_gain')
        _df = pd.DataFrame(gain.values(), index=gain.keys())
        gain_df = pd.merge(gain_df, _df, how="outer", left_index=True, right_index=True)

    # 各foldの平均を算出
    gain_df_mean = pd.DataFrame(gain_df.mean(axis=1), columns=['importance']).fillna(0)
    
    # 降順に並べ替えてindexを取得
    selected_feats = gain_df_mean.sort_values("importance", ascending=False).iloc[:num_feat].index

    train_select = train[selected_feats]
    test_select = test[selected_feats]

    return train_select, test_select


def main():
    
    # データの読み込み
    train = pd.read_pickle(FEATURE_DIR_NAME + 'train.pkl')
    test = pd.read_pickle(FEATURE_DIR_NAME + 'test.pkl')
    
    # 特徴量選択の実行
    train_select, test_select = select_by_xgb(train, test, n_splits=5, num_feat=50)

    print("train_select shape: ", train_select.shape)
    print("test_select shape: ", test_select.shape)
    
    # pickleファイルとして保存
    train_select.to_pickle(FEATURE_DIR_NAME + 'train_select.pkl')
    test_select.to_pickle(FEATURE_DIR_NAME + 'test_select.pkl')
    
    # 生成した特徴量のリスト
    features_list = list(train_select.columns)
    
    # 特徴量リストの保存
    with open(FEATURE_DIR_NAME + 'selected_features_list.txt', 'wt') as f:
        for i in range(len(features_list)):
            f.write('\'' + str(features_list[i]) + '\',\n')
    
    return 'main() Done!'

    
if __name__ == '__main__':

    main()