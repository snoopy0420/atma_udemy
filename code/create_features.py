import pandas as pd
import numpy as np
import sys,os
import csv
import yaml
from pathlib import Path
import warnings
from util import Logger

import itertools
from sklearn.preprocessing import LabelEncoder
import re
import datetime

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.safe_load(file)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']  # RAWデータ格納場所
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']  # 生成した特徴量の出力場所
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME'] # モデルの格納場所
if 'REMOVE_COLS' in yml['SETTING'].keys():
    REMOVE_COLS = yml['SETTING']['REMOVE_COLS']


#### preprocessing関数を定義 ##########################################################

# all_dfを受け取りall_dfを返す関数

def change_to_date(all_df):
    """datetimeに変換
    """
    all_df["date"] = pd.to_datetime(all_df["date"], format="%Y%m%d")
    if (all_df["date"][0]==20041106):
        for i in ("max_temp_time", "min_temp_time"):
            all_df[i] = pd.C(all_df[i], format="%Y/%m/%d %H:%M")
            
    return all_df


def merge_wea(all_df, wea_df):
    """卸売データと天候データをマージする
    """
    # 卸売データのエリア
    area_pairs = all_df["area"].unique()
    yasai_areas = set()
    for area_pair in area_pairs:
        areas = area_pair.split("_")
        yasai_areas = (yasai_areas | set(areas)) # 論理和
        
    # 天候データのエリア
    wea_areas = wea_df["area"].unique()

    # マッピングのための辞書を作成
    area_map = {}
    update_area_map = {
        '岩手':'盛岡','宮城':'仙台','静岡':'浜松','沖縄':'那覇','神奈川':'横浜','愛知':'名古屋','茨城':'水戸','北海道':'帯広','各地':'全国',
        '兵庫':'神戸','香川':'高松','埼玉':'熊谷','国内':'全国','山梨':'甲府','栃木':'宇都宮','群馬':'前橋','愛媛':'松山'
    }
    for yasai_area in yasai_areas:
        if (yasai_area in wea_areas):
            area_map[yasai_area] = yasai_area
        elif (yasai_area in update_area_map):
            area_map[yasai_area] = update_area_map[yasai_area]
        else:
            area_map[yasai_area] = "全国"

    # 卸売データのareaを置換
    all_df["area"] = all_df["area"].apply(lambda x: "_".join([area_map[i] for i in x.split("_")]))

    #＃ 天候データを処理
    # datetime型の平均の取り方がわからないので削除
    wea_df = wea_df.drop(columns=["max_temp_time","min_temp_time"])

    # wea_dfに全国を追加する
    agg_cols = [i for i in wea_df.columns if i not in ["area","date"]]
    tmp_df = wea_df.groupby(["date"])[agg_cols].agg(["mean"]).reset_index()

    new_cols = []
    for col1,col2 in tmp_df.columns:
        new_cols.append(col1)
    tmp_df.columns = new_cols

    tmp_df["area"] = "全国"
    tmp_df["date"] = wea_df[wea_df["area"]=="千葉"]["date"].values
    tmp_df = tmp_df[wea_df.columns]

    wea_df = pd.concat([wea_df, tmp_df])

    # 複数地点の平均を取る
    area_pairs = all_df["area"].unique()
    target_cols = [i for i in wea_df.columns if i not in("area","date")]
    date = wea_df[wea_df["area"]=="千葉"]["date"]
    area_pair_dfs = []
    for area_pair in area_pairs:
        areas = area_pair.split("_")
        # 全ての値が０のDFを作成
        base_tmp_df = pd.DataFrame(np.zeros(wea_df[wea_df["area"]=="千葉"][target_cols].shape), columns=target_cols)
        for area in areas:
            tmp_df = wea_df[wea_df["area"]==area].reset_index(drop=True)[target_cols]
            base_tmp_df = base_tmp_df.add(tmp_df)
        base_tmp_df /= len(areas)
        base_tmp_df["area"] = area_pair
        base_tmp_df["date"] = date.to_list()
        area_pair_dfs.append(base_tmp_df)

    wea_df = pd.concat(area_pair_dfs)
    
    all_df = pd.merge(all_df, wea_df, on=['date', 'area'], how='left')

    return all_df, wea_df


def add_may(wea_df):
    """wea_dfに5月を追加する関数、ラグ特徴量生成に使用
    """
    # wea_dfに2022/05も追加
    start = datetime.datetime.strptime("2022-05-01", "%Y-%m-%d") # 5月の日付を取得
    may_date = pd.date_range(start, periods=31)
    for area in wea_df["area"].unique():
        # areaとdate意外NANの5月のdfを作る
        maywea_df = pd.DataFrame(columns=wea_df.columns,
                                data={"date":may_date,
                                      "area":area}
                                )
        # dtypesをfloat64に戻す
        cols = [i for i in maywea_df.columns if i not in ("date","area")]
        maywea_df[cols] = maywea_df[cols].astype("float64")
        # wea_dfとconcat
        wea_df = pd.concat([wea_df,maywea_df])
    # area,dateでソート
    wea_df = wea_df.sort_values(["area","date"])
    return wea_df

def get_lag_feat(all_df, wea_df, nshift):
    """単純ラグ特徴量
    """

    # mode_price, amount
    for value in ["mode_price", "amount"]:
        df_wide = all_df.pivot(index="date",columns="kind",values=value)
        df_wide_lag = df_wide.shift(nshift)
        df_long_lag = df_wide_lag.stack().reset_index()
        df_long_lag.columns = ["date", "kind", "{}_{}prev".format(value,nshift)]
        
        all_df = pd.merge(all_df, df_long_lag, on=['date', 'kind'], how='left')
        
    # wether
    # 5月を追加
    wea_df = add_may(wea_df)

    cols = [i for i in wea_df.columns if i not in ("area","date")]
    for value in cols:
        df_wide = wea_df.pivot(index="date",columns="area",values=value)
        df_wide_lag = df_wide.shift(nshift)
        df_long_lag = df_wide_lag.stack().reset_index()
        df_long_lag.columns = ["date", "area", "{}_{}prev".format(value,nshift)]

        all_df = pd.merge(all_df, df_long_lag, on=['date', 'area'], how='left')

    return all_df


def get_moving_ave(all_df, wea_df, start, window_size):
    """移動平均ラグ特徴量
    """

    # all_df
    # 移動平均のラグ特徴量　start期前からwindiw_size分の移動平均を取る
    for value in ["mode_price", "amount"]:
        df_wide = all_df.pivot(index="date", columns="kind", values=value)
        df_wide_lag = df_wide.shift(start).rolling(window=window_size, min_periods=1).mean()
        df_long_lag = df_wide_lag.stack().reset_index()

        df_long_lag.columns = ["date", "kind", "{}_{}prev_mean".format(value, window_size)]
        all_df = pd.merge(all_df, df_long_lag, on=['date', 'kind'], how='left')

    # wether_df
    # 5月を追加
    # wea_df = add_may(wea_df)

    cols = [i for i in wea_df.columns if i not in ("area","date")]
    for value in cols:
        df_wide = wea_df.pivot(index="date",columns="area",values=value)
        df_wide_lag = df_wide.shift(start).rolling(window=window_size, min_periods=1).mean()
        df_long_lag = df_wide_lag.stack().reset_index()
        df_long_lag.columns = ["date", "area", "{}_{}prev_mean".format(value,window_size)]

        all_df = pd.merge(all_df, df_long_lag, on=['date', 'area'], how='left')
        
    return all_df


def get_time_feat(all_df):
    """年、月、日、曜日を特徴量に追加
    """
    all_df["weekday"] = all_df["date"].dt.weekday
    all_df["year"] = all_df["date"].dt.year
    all_df["month"] = all_df["date"].dt.month
    all_df["day"] = all_df["date"].dt.day

    return all_df



def get_labelencoding(all_df):
    """ラベルエンコーディング
    """
    cols = all_df.dtypes[all_df.dtypes=="object"].index
    for col in cols:
        all_df.loc[:, col] = all_df[col].fillna("NaN")
        le = LabelEncoder()
        all_df.loc[:, col] = le.fit_transform(all_df[col])

    return all_df




##### main関数を定義 ###########################################################################
def main():
    
    # データの読み込み
    train = pd.read_csv(RAW_DATA_DIR_NAME + 'train.csv')
    test = pd.read_csv(RAW_DATA_DIR_NAME + 'test.csv')
    wea = pd.read_csv(RAW_DATA_DIR_NAME + "weather.csv")
    df = pd.concat([train, test], ignore_index=True)
    
    # preprocessingの実行
    df = change_to_date(df)
    wea = change_to_date(wea)
    df,wea = merge_wea(df,wea)
    df = get_lag_feat(df,wea,31)
    df = get_lag_feat(df,wea,365)
    df = get_moving_ave(df,wea,1,31)
    df = get_time_feat(df)
    df = get_labelencoding(df)
    
    # trainとtestに分割
    train = df.iloc[:len(train), :]
    test = df.iloc[len(train):, :]

    print("train shape: ", train.shape)
    print("test shape: ", test.shape)
    
    # pickleファイルとして保存
    train.to_pickle(FEATURE_DIR_NAME + 'train.pkl')
    test.to_pickle(FEATURE_DIR_NAME + 'test.pkl')
#     logger.info(f'train shape: {train.shape}, test shape, {test.shape}')
    
    # 生成した特徴量のリスト
    features_list = list(df.columns)
    if 'REMOVE_COLS' in yml['SETTING'].keys():
        features_list = list(df.drop(columns=REMOVE_COLS).columns)  # 学習に不要なカラムは除外
    
    # 特徴量リストの保存
    # features_list = sorted(features_list)
    with open(FEATURE_DIR_NAME + 'features_list.txt', 'wt', encoding='utf-8') as f:
        for i in range(len(features_list)):
            f.write('\'' + str(features_list[i]) + '\',\n')
    
    return 'main() Done!'

    
if __name__ == '__main__':
    
#     global logger
#     logger = Logger(MODEL_DIR_NAME + "create_features" + "/")

    main()
