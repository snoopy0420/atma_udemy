import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import linregress
from pathlib import Path
from abc import ABCMeta, abstractmethod
from time import time
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.util import Logger, Util



def decorate(s: str, decoration=None):
    if decoration is None:
        decoration = '★' * 20

    return ' '.join([decoration, str(s), decoration])

class Timer:
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' ', verbose=0):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None
        self.verbose = verbose

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        if self.verbose is None:
            return
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)

class FeatureBase(metaclass=ABCMeta):

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        self.use_cache = use_cache
        self.name = self.__class__.__name__
        self.cache_dir = Path(DIR_FEATURE)
        self.logger = logger
        self.seve_cache = save_cache
        self.use_cols = None
        self.key_column = None
    
    # 共通のキー整形 & 重複チェック
    def enforce_key_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        for key in self.key_column:
            if key not in df.columns:
                raise KeyError(f"{self.name}: キーカラム '{key}' が存在しません")
        assert ~df[self.key_column].duplicated().any(), f"{self.name}: 主キー {self.key_column} に重複があります"
    
    @abstractmethod
    def _create_feature(self) -> pd.DataFrame:
        """
        特徴量生成の実装をサブクラスで定義する必要があります。
        :return: pd.DataFrame 生成された特徴量
        """
        raise NotImplementedError()

    # 特徴量生成処理
    def create_feature(self) -> pd.DataFrame:

        # クラス名.pkl
        file_name = os.path.join(self.cache_dir, f"{self.name}.pkl")

        # キャッシュを使う & ファイルがあるなら読み出し
        if os.path.isfile(str(file_name)) and self.use_cache:
            feature = pd.read_pickle(file_name)
            print(decorate(f"{self.name}の特徴量をキャッシュから読み込みました。", decoration='★'))

        # 変換処理を実行
        else:
            # train/testの区別なく変換処理を実行
            feature = self._create_feature()

            # 主キーチェック
            if self.key_column is not None:
                self.enforce_key_integrity(feature)

            # 保存する場合
            if self.seve_cache:
                feature.to_pickle(file_name)

        return feature


def one_hot_encode(df, col, drop_col=True):
    """
    特定の列に対してOne-Hotエンコーディングを適用します。
    
    :param df: pd.DataFrame 対象のDataFrame
    :param col: str エンコードする列名
    :return: pd.DataFrame エンコードされたDataFrame
    """
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

    # Fit and transform the specified column
    encoded = encoder.fit_transform(df[[col]])

    # Convert the encoded array to a DataFrame
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))

    # concat
    df_concat = pd.concat([df, encoded_df], axis=1)

    # drop 
    if drop_col:
        df_concat.drop(columns=col, inplace=True)

    return df_concat

def clean_feature_names(data):
    # 特徴量名を修正
    data.columns = data.columns.str.replace(r'[^\w]', '_', regex=True)
    return data




############ 継承クラス ##########################################################################

class Key(FeatureBase):
    """
    TrainFeatureクラスは、train.csvデータを処理し、特徴量を生成します。
    """
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号', 'category']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:
        """
        train.csvデータを読み込み、特徴量を生成します。

        Returns:
        pd.DataFrame: 生成された特徴量を含むDataFrame。
        """
        # train.csvデータを読み込む
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_train.pkl'))
        df_test = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_test.pkl'))
        df_Key = pd.concat([df_train, df_test], ignore_index=True)[self.key_column]

        return df_Key
    

class Target(FeatureBase):
    """
    Targetクラスは、ターゲットデータを処理し、特徴量を生成します。
    """
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号', 'category']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:
        """
        ターゲットデータを読み込み、特徴量を生成します。

        Returns:
        pd.DataFrame: 生成されたターゲットデータを含むDataFrame。
        """
        # ターゲットデータを読み込む
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_train.pkl'))

        # 必要なカラムを選択
        df_target = df_train[['社員番号', 'category', 'target']]

        # 主キーとターゲット列を含むDataFrameを返す
        return df_target

class CategoryFeature(FeatureBase):
    """
    TrainFeatureクラスは、train.csvデータを処理し、特徴量を生成します。
    """
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['category']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:
        """
        train.csvデータを読み込み、特徴量を生成します。

        Returns:
        pd.DataFrame: 生成された特徴量を含むDataFrame。
        """
        # train.csvデータを読み込む
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_train.pkl'))
        df_test = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_test.pkl'))
        df_all = pd.concat([df_train, df_test], ignore_index=True)

        df_category_feature = df_all.copy().drop_duplicates('category')[self.key_column]

        # One-hotエンコーディング
        # df_category_feature = one_hot_encode(df_category_feature, 'category', False)

        # labaelエンコーディング
        le = LabelEncoder()
        df_category_feature['le_category'] = le.fit_transform(df_category_feature['category'])

        # 主キーとターゲット列を含むDataFrameを返す
        return df_category_feature


class CareerFeature(FeatureBase):
    """
    CareerBlockクラスは、キャリアデータを処理し、特徴量を生成します。
    """
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:
        """
        キャリアデータを読み込み、特徴量を生成します。

        Returns:
        pd.DataFrame: 生成された特徴量を含むDataFrame。
        """
        # 前処理済みのキャリアデータを読み込む
        df_career = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_career.pkl"))

        df_career_feature = df_career.copy()

        # ポジティブな回答数
        df_career_feature['positive_responses'] = df_career_feature.iloc[:, 1:].apply(lambda row: (row>=4).sum(), axis=1)

        # ネガティブな回答数
        df_career_feature['negative_responses'] = df_career_feature.iloc[:, 1:].apply(lambda row: (row<=2).sum(), axis=1)

        # ポジティブな回答の割合
        df_career_feature['positive_ratio'] = df_career_feature['positive_responses'] / (df_career_feature.shape[1] - 1)

        return df_career_feature



   
class UdemyActivityFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 前処理済みのUdemy活動データを読み込む
        df_udemy = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_udemy_activity.pkl"))

        df_udemy_feature = df_udemy.copy()[self.key_column].drop_duplicates()

        # クイズ判定
        # df_udemy["is_quiz"] = df_udemy["レクチャーもしくはクイズ"]=="Quiz"

        # 基本統計量の集計
        df_udemy_activity_numerical = df_udemy.groupby(self.key_column).agg(
            # 推定完了率%
            mean_推定完了率=('推定完了率_', 'mean'),
            min_推定完了率=('推定完了率_', 'min'),
            max_推定完了率=('推定完了率_', 'max'),
            std_推定完了率=('推定完了率_', 'std'),
            count_推定完了率=('推定完了率_', 'count'),
            # 最終結果（クイズの場合）
            mean_最終結果=('最終結果_クイズの場合_', 'mean'),
            min_最終結果=('最終結果_クイズの場合_', 'min'),
            max_最終結果=('最終結果_クイズの場合_', 'max'),
            std_最終結果=('最終結果_クイズの場合_', 'std'),
            count_最終結果=('最終結果_クイズの場合_', 'count'),
            # マーク済み修了
            mean_マーク済み修了=('マーク済み修了', 'mean'),
            min_マーク済み修了=('マーク済み修了', 'min'),
            max_マーク済み修了=('マーク済み修了', 'max'),
            std_マーク済み修了=('マーク済み修了', 'std'),
            count_マーク済み修了=('マーク済み修了', 'count'),
            # 開始日
            min_開始日=('開始日', 'min'),
            max_開始日=('開始日', 'max'),
        )

        # 学習スパン（日数）
        df_udemy_activity_numerical["learning_span"] = (df_udemy_activity_numerical["max_開始日"] - df_udemy_activity_numerical["min_開始日"]).dt.days
        # 日付型を数値型に変換
        df_udemy_activity_numerical["min_開始日"] = df_udemy_activity_numerical["min_開始日"].apply(lambda x: float(datetime.strftime(x, format='%Y%m%d')))
        df_udemy_activity_numerical["max_開始日"] = df_udemy_activity_numerical["max_開始日"].apply(lambda x: float(datetime.strftime(x, format='%Y%m%d')))
        
        # コースカテゴリごとの回数を集計
        # 正規化の結果同じ値になったものを分別
        map_val = {val: f'{i}_{val}' for i, val in enumerate(df_udemy['コースカテゴリー'].unique())}
        df_udemy['コースカテゴリー'] = df_udemy['コースカテゴリー'].map(lambda x: map_val.get(x, np.nan))
        # 集計
        df_udemy_activity_course_category = df_udemy.pivot_table(
            index='社員番号',
            columns='コースカテゴリー',
            values='コースID',
            aggfunc='count',
            fill_value=None,
        ).reset_index()
        # カラム名を変更
        prefix = "ua_カテゴリ_"
        df_udemy_activity_course_category.columns = [col if col=='社員番号' else prefix + col for col in df_udemy_activity_course_category.columns]

        # レクチャーもしくはクイズごとの回数を集計
        df_udemy_activity_type = df_udemy.pivot_table(
            index='社員番号',
            columns='レクチャーもしくはクイズ',
            values='コースID',
            aggfunc='count',
            fill_value=None,
        ).reset_index()
        # カラム名を変更
        prefix = "ua_レクチャーorクイズ_"
        df_udemy_activity_type.columns = [col if col=='社員番号' else prefix + col for col in df_udemy_activity_type.columns]

        # # コースIDごとの回数を集計
        # df_udemy_activity_course_id = df_udemy.pivot_table(
        #     index='社員番号',
        #     columns='コースID',
        #     values='コースID',
        #     aggfunc='count',
        #     fill_value=0
        # ).reset_index()     
        # # カラム名を変更
        # prefix = "ua_コースID_"
        # df_udemy_activity_course_id.columns = [str(col[0]) if col[0]=='社員番号' else prefix + str(col[1]) for col in df_udemy_activity_course_id.columns]

        # # クイズスコアの集計
        # df_quiz = df_udemy[df_udemy["is_quiz"]].copy()
        # df_quiz_stats = df_quiz.groupby(self.key_column).agg(
        #     count_クイズ=('最終結果（クイズの場合）', 'count'),
        #     mean_クイズスコア=('最終結果（クイズの場合）', 'mean'),
        # ).reset_index()

        # カラム名を変更
        # df_udemy_activity_course_category.columns = [replace_special_characters(col) for col in df_udemy_activity_course_category.columns]
        # df_udemy_activity_type.columns = [replace_special_characters(col) for col in df_udemy_activity_type.columns]


        # マージ
        df_udemy_feature = df_udemy_feature.merge(df_udemy_activity_numerical, on=self.key_column, how='left')
        # df_udemy_feature = df_udemy_feature.merge(df_udemy_activity_course_category, on=self.key_column, how='left')
        df_udemy_feature = df_udemy_feature.merge(df_udemy_activity_type, on=self.key_column, how='left')

        # カラム名の修正
        df_udemy_feature = clean_feature_names(df_udemy_feature)

        return df_udemy_feature
    

class UdemyTimeseriesFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:
        """
        Udemyの時系列特徴量を生成します。

        Returns:
        pd.DataFrame: 生成された特徴量を含むDataFrame。
        """
        # 前処理済みのUdemy活動データを読み込む
        df_udemy = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_udemy_activity.pkl"))

        df_udemy_target = df_udemy.copy()
        df_udemy_target['開始年'] = df_udemy_target['開始日'].dt.to_period('Y')

        # 社員番号、開始年ごとの受講数を集計して増加数を計算
        df_udemy_target = df_udemy_target.groupby(['社員番号', '開始年']).agg(受講数=('コースID', 'count')).reset_index()
        # 社員番号、開始年のすべての組み合わせを作成
        id = df_udemy_target['社員番号'].unique()
        year = df_udemy_target['開始年'].unique()
        df_udemy_target_all = pd.MultiIndex.from_product([id, year], names=['社員番号', '開始年']).to_frame(index=False)
        df_udemy_target_all = df_udemy_target_all.merge(df_udemy_target, on=['社員番号', '開始年'], how='left')
        # lagを計算
        df_udemy_target_all.sort_values(['社員番号', '開始年'], inplace=True)
        # lag特徴量を生成
        lag=7
        for i in range(1, lag + 1):
            df_udemy_target_all[f'ua_受講数_{i}_age'] = df_udemy_target_all.groupby('社員番号')['受講数'].shift(i)

        # 最新行を抽出
        df_udemy_lag = df_udemy_target_all.groupby('社員番号').tail(1).reset_index(drop=True)

        # カラム整形
        lag_cols = [f'ua_受講数_{i}_age' for i in range(1, lag + 1)]
        df_udemy_lag = df_udemy_lag[['社員番号', '受講数'] + lag_cols]
        df_udemy_lag = df_udemy_lag.rename(columns={'受講数': 'ua_受講数_0_age'})

        # 前年との受講数の差分を計算
        for i in range(0, lag):
            df_udemy_lag[f'ua_受講数_{i}_age_ratio'] = df_udemy_lag[f'ua_受講数_{i}_age'] / df_udemy_lag[f'ua_受講数_{i+1}_age']

        # プラスとマイナスの受講数の差分を計算
        ratio_cols = [f'ua_受講数_{i}_age_ratio' for i in range(0, lag)]
        df_udemy_lag['ua_受講数_0_age_diff_plus'] = df_udemy_lag[ratio_cols].apply(lambda x: len(x[x > 1.0]), axis=1)
        df_udemy_lag['ua_受講数_0_age_diff_minus'] = df_udemy_lag[ratio_cols].apply(lambda x: len(x[x < 1.0]), axis=1)
        
        # # 線形トレンド（回帰直線の傾き）を算出
        # trends = []
        # for _, row in df_udemy_lag[lag_cols].iterrows():
        #     y = row.values
        #     x = np.arange(1, lag + 1)[::-1].reshape(-1, 1)
        #     if np.isnan(y).all():
        #         trends.append(np.nan)
        #     else:
        #         mask = ~np.isnan(y) # NaNを除外
        #         reg = LinearRegression().fit(x[mask], y[mask])
        #         trends.append(reg.coef_[0])
        # df_udemy_lag[f'ua_受講数_trend'] = trends

        return df_udemy_lag
    

def create_sparse_matrix(df: pd.DataFrame, user_col: str, action_col: str, value_col=None,) -> tuple[sp.csr_matrix, LabelEncoder, LabelEncoder]:
    # user_col と action_col を数値に変更する
    user_encoder = LabelEncoder()
    action_encoder = LabelEncoder()
    user_array = user_encoder.fit_transform(df[user_col].to_numpy().ravel())
    action_array = action_encoder.fit_transform(df[action_col].to_numpy().ravel())

    # 重みを指定する (value_colがNoneの場合は1を指定)
    data_array = df[value_col].to_numpy().ravel() if value_col is not None else np.ones(len(df))

    # スパース行列を作成する
    sparse_matrix = sp.csr_matrix(
        (data_array, (user_array, action_array)),
        shape=(len(user_encoder.classes_), len(action_encoder.classes_)),
    )
    return sparse_matrix, user_encoder, action_encoder

def generate_embeddings(df: pd.DataFrame, user_col: str, action_col: str, value_col=None, n_components: int = 8, prefix: str = "") -> pd.DataFrame:
    """
    スパース行列を作成し、SVDで次元削減を行い、埋め込みを生成する関数
    Args:
        df (pd.DataFrame): 入力データフレーム
        user_col (str): ユーザーを識別するカラム名
        action_col (str): アクションを識別するカラム名
        value_col (str, optional): 重みを指定するカラム名 (デフォルトはNone)
        n_components (int): SVDでの次元数
    Returns:
        pd.DataFrame: ユーザーごとの埋め込み特徴量を含むデータフレーム
    """
    # スパース行列を作成
    sparse_matrix, user_encoder, action_encoder = create_sparse_matrix(df, user_col, action_col, value_col)

    # SVDで次元削減
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    # ユーザー埋め込みを生成
    user_embeddings = svd.fit_transform(sparse_matrix)
    df_user_embeddings = pd.concat([
        pd.DataFrame({user_col: user_encoder.classes_}),
        pd.DataFrame(user_embeddings, columns=[f'{prefix}svd_{action_col}_{i}' for i in range(user_embeddings.shape[1])])
    ], axis=1)

    # アクション埋め込みを生成
    action_embeddings = svd.components_.T
    course_title_to_vec = {
        course: action_embeddings[idx]
        for course, idx in zip(action_encoder.classes_, range(len(action_encoder.classes_)))
    }

    # 各ユーザーごとのベクトル平均を計算
    def compute_mean_embedding(group):
        embeddings = [course_title_to_vec[title] for title in group[action_col] if title in course_title_to_vec]
        if embeddings:
            return pd.Series(np.mean(embeddings, axis=0))
        else:
            return pd.Series([np.nan] * n_components)

    df_mean_embeddings = df.groupby(user_col).apply(compute_mean_embedding).reset_index()
    df_mean_embeddings.columns = [user_col] + [f"{prefix}mean_svd_{action_col}_{i}" for i in range(n_components)]

    # 埋め込みデータをマージ
    df_embeddings = df[[user_col]].drop_duplicates().merge(df_user_embeddings, on=user_col, how='left')
    df_embeddings = df_embeddings.merge(df_mean_embeddings, on=user_col, how='left')

    return df_embeddings
    

class UdemyTitleEmbedding(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 前処理済みのUdemy活動データを読み込む
        df_udemy = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_udemy_activity.pkl"))

        # スパース行列を作成し、SVDで次元削減を行い、埋め込みを生成
        n_components = 8
        df_udemy_user_embeddings = generate_embeddings(df_udemy,
                                                     user_col='社員番号', 
                                                     action_col='コースタイトル', 
                                                     n_components=n_components)

        return df_udemy_user_embeddings

class UdemyIDEmbedding(FeatureBase):

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 前処理済みのUdemy活動データを読み込む
        df_udemy = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_udemy_activity.pkl"))

        # スパース行列を作成し、SVDで次元削減を行い、埋め込みを生成
        n_components = 8
        df_udemy_ID_embeddings_feature = generate_embeddings(df_udemy,
                                                             user_col='社員番号', 
                                                             action_col='コースID', 
                                                             n_components=n_components)

        return df_udemy_ID_embeddings_feature
    


def calculate_similarity_features(df_course, df_train, course_col="コースタイトル", train_col="category", model_name="hotchpotch/static-embedding-japanese", device="cpu"):
    """
    講座タイトルとカテゴリの類似度スコアを計算し、統計量を生成する関数。

    Args:
        df_course (pd.DataFrame): 講座データ
        df_train (pd.DataFrame): 学習データ
        course_col (str): 講座タイトルのカラム名
        train_col (str): カテゴリのカラム名
        model_name (str): 埋め込みモデルの名前
        device (str): モデルを実行するデバイス ("cpu" または "cuda")

    Returns:
        pd.DataFrame: 類似度スコアの統計量を含む特徴量データフレーム
    """
    # ユニークな講座タイトルとカテゴリを抽出
    unique_course_titles = df_course[course_col].unique().tolist()
    unique_train_cats = df_train[train_col].unique().tolist()

    # 埋め込みを取得
    model = SentenceTransformer(model_name, device=device)
    emb_course = model.encode(unique_course_titles, show_progress_bar=True)
    emb_train = model.encode(unique_train_cats, show_progress_bar=True)

    # 類似度行列 (カテゴリ x 講座タイトル)
    sim_matrix = cosine_similarity(emb_train, emb_course)
    df_sim = pd.DataFrame(sim_matrix, index=unique_train_cats, columns=unique_course_titles)

    # 社員ごとの受講履歴を取得
    # df_user_course = df_course[["社員番号", course_col]].copy()
    df_user_course = df_course[["社員番号", course_col]].drop_duplicates().copy()

    # 類似度スコアの統計量を計算
    df_similarity_features = df_train[["社員番号", train_col]].drop_duplicates().copy()
    sim_mean_list = []
    sim_max_list = []
    sim_min_list = []

    for _, row in df_similarity_features.iterrows():
        emp_id = row["社員番号"]
        train_cat = row[train_col]
        # その社員が受講した講座タイトル
        learned_titles = df_user_course[df_user_course["社員番号"] == emp_id][course_col].tolist()
        # カテゴリとの類似度を取得
        similarities = [df_sim.loc[train_cat, title] for title in learned_titles if title in df_sim.columns]
        # 類似度スコアの統計量を算出
        if similarities:
            sim_mean_list.append(np.mean(similarities))
            sim_max_list.append(np.max(similarities))
            sim_min_list.append(np.min(similarities))
        else:
            sim_mean_list.append(np.nan)
            sim_max_list.append(np.nan)
            sim_min_list.append(np.nan)

    # 結果をDataFrameに追加
    df_similarity_features[f"{course_col}_sim_mean"] = sim_mean_list
    df_similarity_features[f"{course_col}_sim_max"] = sim_max_list
    df_similarity_features[f"{course_col}_sim_min"] = sim_min_list

    return df_similarity_features


class UdemyCategorySimilarityFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号', 'category']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # データ読み込み
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_train.pkl"))
        df_test = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_test.pkl'))
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        df_udemy = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_udemy_activity.pkl"))

        # nan除外
        df_udemy = df_udemy[df_udemy["コースカテゴリー"].notnull()].copy()
        df_udemy = df_udemy[df_udemy["コースカテゴリー"] != "企業オリジナル講座"].copy()

        df_category_sim_feature = calculate_similarity_features(
            df_course=df_udemy,
            df_train=df_all,
            course_col="コースカテゴリー",
            train_col="category",
        )

        return df_category_sim_feature
    
class UdemyTitleSimilarityFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号', 'category']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # データ読み込み
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_train.pkl"))
        df_test = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_test.pkl"))
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        df_udemy = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_udemy_activity.pkl"))

        # nan除外
        df_udemy = df_udemy[df_udemy["コースタイトル"].notnull()].copy()
        df_udemy = df_udemy[df_udemy["コースタイトル"] != "企業オリジナル講座"].copy()

        df_title_sim_feature = calculate_similarity_features(
            df_course=df_udemy,
            df_train=df_all,
            course_col="コースタイトル",
            train_col="category",
        )

        return df_title_sim_feature
    
class DxSimilarityFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号', 'category']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # データ読み込み
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_train.pkl"))
        df_test = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_test.pkl'))
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        df_dx = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_dx.pkl"))

        # nan除外
        df_dx_cate = df_dx[df_dx["研修カテゴリ"].notnull()].copy()

        df_cate_sim_feature = calculate_similarity_features(
            df_course=df_dx_cate,
            df_train=df_all,
            course_col="研修カテゴリ",
            train_col="category",
        )

        df_dx_title = df_dx[df_dx["研修名"].notnull()].copy()

        df_title_sim_feature = calculate_similarity_features(
            df_course=df_dx_title,
            df_train=df_all,
            course_col="研修名",
            train_col="category",
        )

        df_sim_feature = df_cate_sim_feature.merge(df_title_sim_feature, on=self.key_column, how='left')

        return df_sim_feature
    
class HrSimilarityFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号', 'category']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # データ読み込み
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_train.pkl"))
        df_test = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_test.pkl'))
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        df_hr = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_hr.pkl"))

        # nan除外
        df_hr_cate = df_hr[df_hr["カテゴリ"].notnull()].copy()

        df_cate_sim_feature = calculate_similarity_features(
            df_course=df_hr_cate,
            df_train=df_all,
            course_col="カテゴリ",
            train_col="category",
        )

        df_hr_title = df_hr[df_hr["研修名"].notnull()].copy()

        df_title_sim_feature = calculate_similarity_features(
            df_course=df_hr_title,
            df_train=df_all,
            course_col="研修名",
            train_col="category",
        )

        df_sim_feature = df_cate_sim_feature.merge(df_title_sim_feature, on=self.key_column, how='left')

        return df_sim_feature



class DxFeature(FeatureBase):

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        df_dx = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_dx.pkl"))

        df_dx_feature = df_dx.copy()[self.key_column].drop_duplicates()

        # 基本集計値
        df_dx_numerical = df_dx.groupby(self.key_column).agg(
            dx_count=('研修名', 'count'),
            dx_unique_研修名=('研修名', 'nunique'),
            dx_unique_研修カテゴリ=('研修カテゴリ', 'nunique'),
        ).reset_index()

        # 研修カテゴリごとの参加回数を集計
        df_category_count = df_dx.pivot_table(
            index='社員番号',
            columns="研修カテゴリ",
            values="研修実施日",
            aggfunc="count",
            fill_value=None,
        ).reset_index()
        # カラム名の変更
        prefix = "dx_研修カテゴリ_"
        df_category_count.columns = [col if col=='社員番号' else prefix + col for col in df_category_count.columns]

        # 研修名ごとの参加回数を集計
        df_name_count = df_dx.pivot_table(
            index='社員番号',
            columns="研修名",
            values="研修実施日",
            aggfunc="count",
            fill_value=None,
        ).reset_index()
        # カラム名の変更
        prefix = "dx_研修名_"
        df_name_count.columns = [col if col=='社員番号' else prefix + col for col in df_name_count.columns]

        # 各社員の研修参加回数
        # df_dx_feature = df_dx.groupby(self.key_column).agg(
        #     count=('研修名', 'count'),
        #     unique_training_count=('研修名', 'nunique'),
        # ).reset_index()

        # 各社員のユニークな研修カテゴリ数
        # df_dx_feature['unique_training_categories'] = dx_data.groupby(self.key_column)['研修カテゴリ'].transform('nunique')

        # マージ
        df_dx_feature = df_dx_feature.merge(df_dx_numerical, on=self.key_column, how='left')
        df_dx_feature = df_dx_feature.merge(df_category_count, on=self.key_column, how='left')
        df_dx_feature = df_dx_feature.merge(df_name_count, on=self.key_column, how='left')

        # カラム名の修正
        df_dx_feature = clean_feature_names(df_dx_feature)

        return df_dx_feature
    
class DxCategoryEmbeddingFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 前処理済みのDXデータを読み込む
        df_dx = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_dx.pkl"))

        # スパース行列を作成し、SVDで次元削減を行い、埋め込みを生成
        n_components = 8
        df_dx_category_embeddings = generate_embeddings(df_dx,
                                                       user_col='社員番号', 
                                                       action_col='研修カテゴリ', 
                                                       n_components=n_components,
                                                       prefix='dx_')

        return df_dx_category_embeddings
    
class DxNameEmbeddingFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 前処理済みのDXデータを読み込む
        df_dx = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_dx.pkl"))

        # スパース行列を作成し、SVDで次元削減を行い、埋め込みを生成
        n_components = 8
        df_dx_name_embeddings = generate_embeddings(df_dx,
                                                   user_col='社員番号', 
                                                   action_col='研修名', 
                                                   n_components=n_components,
                                                   prefix='dx_')

        return df_dx_name_embeddings


class HrFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache, save_cache, logger=None)
        self.key_column = ['社員番号']

    def _create_feature(self) -> pd.DataFrame:

        df_hr = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_hr.pkl'))

        df_hr_feature = df_hr.copy()[self.key_column].drop_duplicates()

        # # 実施期間を算出（日数）
        # df_hr['研修日数'] = (df_hr['実施終了日'] - df_hr['実施開始日']).dt.days + 1
        # df_hr['研修日数'] = df_hr['研修日数'].fillna(1).clip(lower=1)

        # # 基礎集計値
        # df_hr_numetric = df_hr.groupby(self.key_column).agg(
        #     hr_count=('研修名', 'count'),
        #     hr_unique_研修名=('研修名', 'nunique'),
        #     hr_unique_カテゴリ=('カテゴリ', 'nunique'),
        #     min_実施開始日=("実施開始日", "min"),
        #     max_実施終了日=("実施終了日", "max"),
        #     sum_研修日数=("研修日数", "sum"),
        # ).reset_index()

        # # 活動期間（最終日 - 初日）
        # df_hr_numetric["hr_active_days"] = (df_hr_numetric["max_実施終了日"] - df_hr_numetric["min_実施開始日"]).dt.days
        
        # # 日付型を数値型に変換
        # df_hr_numetric["min_実施開始日"] = df_hr_numetric["min_実施開始日"].apply(lambda x: float(datetime.strftime(x, format='%Y%m%d')))
        # df_hr_numetric["max_実施終了日"] = df_hr_numetric["max_実施終了日"].apply(lambda x: float(datetime.strftime(x, format='%Y%m%d')))

        # 研修カテゴリごとの参加回数を集計
        df_category_count = df_hr.pivot_table(
            index='社員番号',
            columns="カテゴリ",
            values="実施開始日",
            aggfunc="count",
            fill_value=None,
        ).reset_index() 
        # カラム名の変更
        prefix = "hr_研修カテゴリ_"
        df_category_count.columns = [col if col=='社員番号' else prefix + col for col in df_category_count.columns]

        # 研修名ごとの参加回数を集計
        df_name_count = df_hr.pivot_table(
            index='社員番号',
            columns="研修名",
            values="実施開始日",
            aggfunc="count",
            fill_value=None,
        ).reset_index()
        # カラム名の変更
        prefix = "hr_研修名_"
        df_name_count.columns = [col if col=='社員番号' else prefix + col for col in df_name_count.columns]

        # # 実施期間を算出（日数）
        # df_hr['研修日数'] = (df_hr['実施終了日'] - df_hr['実施開始日']).dt.days + 1
        # df_hr['研修日数'] = df_hr['研修日数'].fillna(1).clip(lower=1)

        # # 特徴量作成
        # df_hr_feature = df_hr.groupby("社員番号").agg(
        #     n_hr_total=("研修名", "count"),
        #     n_hr_unique_program=("研修名", "nunique"),
        #     n_hr_unique_category=("カテゴリ", "nunique"),
        #     first_hr_date=("実施開始日", "min"),
        #     last_hr_date=("実施終了日", "max"),
        #     n_hr_days=("研修日数", "sum"),
        # ).reset_index()

        # # 活動期間（最終日 - 初日）
        # df_hr_feature["hr_active_days"] = (df_hr_feature["last_hr_date"] - df_hr_feature["first_hr_date"]).dt.days
        # df_hr_feature.drop(["first_hr_date", "last_hr_date"], axis=1, inplace=True)

        # マージ
        # df_hr_feature = df_hr_feature.merge(df_hr_numetric, on=self.key_column, how='left')
        df_hr_feature = df_hr_feature.merge(df_category_count, on=self.key_column, how='left')
        df_hr_feature = df_hr_feature.merge(df_name_count, on=self.key_column, how='left')

        return df_hr_feature
    
class HrCategoryEmbeddingFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 前処理済みのHRデータを読み込む
        df_hr = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_hr.pkl"))

        # スパース行列を作成し、SVDで次元削減を行い、埋め込みを生成
        n_components = 8
        df_hr_category_embeddings = generate_embeddings(df_hr,
                                                       user_col='社員番号', 
                                                       action_col='カテゴリ', 
                                                       n_components=n_components,
                                                       prefix='hr_')

        return df_hr_category_embeddings


class HrNameEmbeddingFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 前処理済みのHRデータを読み込む
        df_hr = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_hr.pkl"))

        # スパース行列を作成し、SVDで次元削減を行い、埋め込みを生成
        n_components = 8
        df_hr_name_embeddings = generate_embeddings(df_hr,
                                                   user_col='社員番号', 
                                                   action_col='研修名', 
                                                   n_components=n_components,
                                                   prefix='hr_')

        return df_hr_name_embeddings

class OvertimeWorkByMonthFeature(FeatureBase):

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 残業データを読み込む
        df_overtime = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_overtime_work_by_month.pkl"))

        # 基礎特徴量
        df_overtime_base_feature = df_overtime.groupby(self.key_column).agg(
            avg_overtime=('hours', 'mean'),
            median_overtime=('hours', 'median'),
            max_overtime=('hours', 'max'),
            min_overtime=('hours', 'min'),
            # total_overtime_hours=('hours', 'sum'),
            std_overtime=('hours', 'std'),
            count_overtime_months=('hours', 'count'),
        ).reset_index()


        return df_overtime_base_feature
    
    
class OvertimeWorkByMonthTimeseriesFeature(FeatureBase):

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 残業データを読み込む
        df_overtime = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_overtime_work_by_month.pkl"))

        # dateの欠損行はNullで埋める
        unique_employees = df_overtime['社員番号'].unique()
        unique_dates = df_overtime['date'].unique()
        all_combinations = pd.MultiIndex.from_product([unique_employees, unique_dates], names=['社員番号', 'date']).to_frame(index=False)
        df_overtime = pd.merge(all_combinations, df_overtime, on=['社員番号', 'date'], how='left')

        # lag特徴量
        def make_worker_hours_lag_features(df_overtime, lag=35):
            """
            社員別の過去労働時間（lag特徴量）を作成し、最新月の1行にまとめる。
            Returns:
                df_worker_lag: DataFrame
                    社員番号ごとの最新行 + lag特徴量（hours_0_age ～ hours_{lag}_age）
            """
            df = df_overtime.copy()
            df = df.sort_values(['社員番号', 'date']).reset_index(drop=True)

            # lag特徴量を生成
            for i in range(1, lag + 1):
                df[f'hours_{i}_age'] = df.groupby('社員番号')['hours'].shift(i)

            # 最新行を抽出
            df_worker_lag = df.groupby('社員番号').tail(1).reset_index(drop=True)

            # カラム整形
            lag_cols = [f'hours_{i}_age' for i in range(1, lag + 1)]
            df_worker_lag = df_worker_lag[['社員番号', 'date', 'hours'] + lag_cols]
            df_worker_lag = df_worker_lag.rename(columns={'hours': 'hours_0_age'})

            return df_worker_lag
        df_worker_lag = make_worker_hours_lag_features(df_overtime, lag=35)
        df_worker_lag.drop('date', axis=1, inplace=True)


        # 移動特徴量を生成
        # 各種統計特徴量を生成するウィンドウサイズのリスト
        windows = [3, 6, 12, 24, 36]
        current_hours = df_worker_lag['hours_0_age']
        for w in windows:
            # 直近 w ヶ月分の hours列（hours_0_age ～ hours_{w-1}_age）
            cols = [f'hours_{i}_age' for i in range(0, w)]

            # 移動統計量を計算
            df_worker_lag[f'hours_mean_{w}'] = df_worker_lag[cols].mean(axis=1)              # 平均
            df_worker_lag[f'hours_std_{w}'] = df_worker_lag[cols].std(axis=1)              # 標準偏差
            df_worker_lag[f'hours_max_{w}'] = df_worker_lag[cols].max(axis=1)              # 最大値
            df_worker_lag[f'hours_min_{w}'] = df_worker_lag[cols].min(axis=1)              # 最小値
            df_worker_lag[f'hours_diff_mean_{w}'] = current_hours - df_worker_lag[f'hours_mean_{w}']  # 今月と平均の差
            df_worker_lag[f'hours_range_{w}'] = df_worker_lag[f'hours_max_{w}'] - df_worker_lag[f'hours_min_{w}']  # 振れ幅
            df_worker_lag[f'hours_missing_count_{w}'] = df_worker_lag[cols].isna().sum(axis=1)  # 欠損数
            df_worker_lag[f'hours_zscore_{w}'] = (current_hours - df_worker_lag[f'hours_mean_{w}']) / (df_worker_lag[f'hours_std_{w}'] + 1e-6)  # z-score

            # 今月と wヶ月前との比較（差分）
            df_worker_lag[f'hours_diff_prev_{w}'] = current_hours - df_worker_lag[f'hours_{w-1}_age']
            # df_worker_lag[f'hours_rate_prev_{w}'] = current_hours / df_worker_lag[f'hours_{w-1}_age']

            # 線形トレンド（回帰直線の傾き）を算出
            trends = []
            for _, row in df_worker_lag[cols].iterrows():
                y = row.values
                x = np.arange(1, w + 1).reshape(-1, 1)
                # x = np.arange(1, w + 1)[::-1].reshape(-1, 1)　 # 逆順にすることで最古月が1になる
                if np.isnan(y).all():
                    trends.append(np.nan)
                else:
                    mask = ~np.isnan(y)
                    reg = LinearRegression().fit(x[mask], y[mask])
                    trends.append(reg.coef_[0])
            df_worker_lag[f'hours_trend_{w}'] = trends

            # 今月が過去平均より ±30% を超えているか
            ratio = current_hours / (df_worker_lag[f'hours_mean_{w}'] + 1e-6)
            df_worker_lag[f'hours_over_{w}_flag'] = (ratio > 1.3).astype(int)   # 今月が30%以上多い
            df_worker_lag[f'hours_under_{w}_flag'] = (ratio < 0.7).astype(int)  # 今月が30%以上少ない

            # 指数移動平均（最近の値をより重視した平均）
            df_worker_lag[f'hours_ewm_{w}'] = df_worker_lag[cols].T.ewm(span=3, axis=0).mean().iloc[-1]
        
        # ゴールデンクロス
        shorts = [3, 6]
        long = 36
        for short in shorts:
            gc_col_name = f"hours_gc_{short}_{long}"
            df_worker_lag[gc_col_name] = df_worker_lag[f"hours_mean_{short}"] - df_worker_lag[f"hours_mean_{long}"]

        # 前月との労働時間差が閾値を超えた回数
        spike_threshold = 20
        spike_count = []

        for _, row in df_worker_lag.iterrows():
            diffs = []
            for i in range(1, 35):
                col_now = f'hours_{i}_age'
                col_next = f'hours_{i+1}_age'
                if pd.notna(row[col_now]) and pd.notna(row[col_next]):
                    if abs(row[col_now] - row[col_next]) > spike_threshold:
                        diffs.append(1)
            spike_count.append(sum(diffs))

        df_worker_lag['hours_spike_count'] = spike_count


        # 過去6ヶ月の労働時間の増減方向
        def direction_mode(row, window=6):
            """
            過去 window ヶ月分の労働時間（hours_{i}_age）を比較し、
            増減方向の多数決に基づいてトレンド方向を判定する。

            Parameters:
                row: pd.Series
                window: int (比較対象の月数)
            Returns:
                int: -1 / 0 / 1（下降 / 中立 / 上昇）
            """
            directions = []
            for i in range(0, window - 1):
                a, b = row.get(f'hours_{i}_age'), row.get(f'hours_{i+1}_age')
                if pd.notna(a) and pd.notna(b):
                    # 増加なら +1、減少なら -1、変化なしなら 0
                    directions.append(np.sign(a - b))

            if not directions:
                return 0  # 比較できるペアがない場合は中立とする

            # 合計符号の sign → 全体傾向の方向
            return int(np.sign(sum(directions)))
        
        df_worker_lag['hours_trend_mode_6'] = df_worker_lag.apply(direction_mode, axis=1, args=(6,))

        # lag特徴量の削除
        cols_to_drop = [f'hours_{i}_age' for i in range(0, 36)]
        df_worker_lag.drop(columns=cols_to_drop, inplace=True)

        return df_worker_lag



class PositionHistoryFeature(FeatureBase):

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 役職履歴データを読み込む
        df_position_history = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_position_history.pkl"))

        df_position_history_feature = df_position_history.copy()[self.key_column].drop_duplicates()

        # # 基礎特徴量
        # df_position_history_base = df_position_history.groupby(self.key_column).agg(
        #     nunique_役職=('役職', 'nunique'),
        # ).reset_index()

        # # yearごとの役職
        # le = LabelEncoder()
        # df_position_history_ = df_position_history.copy()
        # df_position_history_['役職'] = le.fit_transform(df_position_history_['役職'])
        # for year in [22, 23, 24]:
        #     df_year_position = df_position_history_[df_position_history_['year'] == year][self.key_column + ['役職']]
        #     df_year_position.rename(columns={'役職': f'役職_{year}'}, inplace=True)
        #     df_position_history_base = df_position_history_base.merge(df_year_position, on=self.key_column, how='left')

        # 勤務区分ごとの年数を集計
        df_work_type_count = df_position_history.pivot_table(
            index='社員番号',
            columns="勤務区分",
            values="year",
            aggfunc="count",
            fill_value=None,
        ).reset_index()
        # カラム名の変更
        prefix = "ph_勤務区分_"
        df_work_type_count.columns = [col if col=='社員番号' else prefix + col for col in df_work_type_count.columns]

        # 役職ごとの年数を集計
        df_position_count = df_position_history.pivot_table(
            index='社員番号',
            columns="役職",
            values="year",
            aggfunc="count",
            fill_value=None,
        ).reset_index()
        # カラム名の変更
        prefix = "ph_役職_"
        df_position_count.columns = [col if col=='社員番号' else prefix + col for col in df_position_count.columns]

        # # 各社員の役職変更回数
        # df_position_history_feature = df_position_history.groupby(self.key_column).agg(
        #     position_change_count=('役職', 'nunique'),
        #     first_position=('役職', 'first'),
        #     last_position=('役職', 'last'),
        # ).reset_index()

        # # one-hotエンコーディング(OneHotEncoder)
        # df_position_history_feature = one_hot_encode(df_position_history_feature, 'first_position')
        # df_position_history_feature = one_hot_encode(df_position_history_feature, 'last_position')

        # マージ
        # df_position_history_feature = df_position_history_feature.merge(df_position_history_base, on=self.key_column, how='left')
        df_position_history_feature = df_position_history_feature.merge(df_work_type_count, on=self.key_column, how='left')
        df_position_history_feature = df_position_history_feature.merge(df_position_count, on=self.key_column, how='left')

        return df_position_history_feature




















