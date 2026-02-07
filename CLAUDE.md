# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

ATMA Cup 20 (Udemy) 機械学習コンペティションのソリューション。社員のUdemy学習活動・HR/DX研修・キャリアアンケート・残業データ・職位履歴をもとに、二値の目的変数を予測する（評価指標: ROC-AUC）。主キーは `['社員番号', 'category']`。

- Python 3.9
- 主要モデル: LightGBM
- コード内のコメント・カラム名・ログは日本語

## コマンド

```bash
# 依存関係のインストール
pip install -r requirements.txt

# ノートブックの実行
jupyter notebook notebooks/
```

ノートブックは `notebooks/` をカレントディレクトリとして実行する前提。`sys.path.append(os.path.abspath('..'))` により `configs/` と `src/` をインポートしている。

## 実行順序

1. `notebooks/preprocess.ipynb` — 生データ(CSV)をクレンジング → `data/interim/` にpickle保存
2. `notebooks/create_features.ipynb` — 特徴量を生成 → `data/features/` にpickle保存
3. `notebooks/exp_lgbm.ipynb`（または `exp_lgbm_unv.ipynb`）— LightGBMの学習・CV評価・提出ファイル作成

## アーキテクチャ

### データパイプライン

```
data/raw/input/*.csv → 前処理 → data/interim/*.pkl → 特徴量クラス → data/features/*.pkl
                                                                        ↓
                                                    Runner + model_LGBM → models/ + data/submission/
```

### 特徴量システム (`src/feature.py`)

全特徴量は `FeatureBase` を継承し、`_create_feature() -> pd.DataFrame` を実装する。基底クラスが提供する機能:
- **キャッシュ**: `use_cache`（読込）と `save_cache`（書込）で制御。`data/features/{クラス名}.pkl` に保存
- **主キー整合性チェック**: キーカラムの存在確認と重複チェック

ノートブックから `create_feature()` を呼び出して使用する。各特徴量クラスは `data/interim/` のpickleを読み込み、`['社員番号', 'category']` をキーとするDataFrameを返す。

特徴量クラス一覧: `Key`, `Target`, `CategoryFeature`, `CareerFeature`, `UdemyActivityFeature`, `UdemyTimeseriesFeature`, `UdemyTitleEmbedding`, `UdemyIDEmbedding`, `UdemyCategorySimilarityFeature`, `UdemyTitleSimilarityFeature`, `DxFeature`, `DxCategoryEmbeddingFeature`, `DxNameEmbeddingFeature`, `DxSimilarityFeature`, `HrFeature`, `HrCategoryEmbeddingFeature`, `HrNameEmbeddingFeature`, `HrSimilarityFeature`, `OvertimeWorkByMonthFeature`, `OvertimeWorkByMonthTimeseriesFeature`, `PositionHistoryFeature`

### モデルシステム

- `src/model.py` — 抽象基底クラス `Model`。`train`, `predict`, `save_model`, `load_model` を定義
- `src/model_LGBM.py` — `model_LGBM` 実装。key/target/removeカラムを自動除外して特徴量カラムを決定。`tune` パラメータによるOptunaハイパーパラメータチューニングに対応。モデルと特徴量カラムリストをpickle保存

### 学習パイプライン (`src/runner.py`)

`Runner` がCV学習ループ全体を管理:
- `StratifiedGroupKFold` を使用（`社員番号` でグループ化、targetで層化）
- `run_train_cv()` → fold毎にモデルを学習・保存
- `run_metric_cv()` → 全foldの評価を実行、スコア（mean, std, fold別）をログ出力
- `run_predict_cv()` → テストデータに対してfold予測の平均値を算出
- `plot_feature_importance_cv()` → gain重要度の上位100件を変動係数とともにプロット
- `after_split_process`（分割後のデータ変換）と `after_predict_process`（予測後の変換）フックに対応

### ユーティリティ (`src/util.py`)

- `Util` — シリアライズ（joblib dump/load, pickle, JSON）
- `Logger` — コンソール + ファイルへの二重出力（`general.log`, `result.log`）、スコアはLTSV形式
- `Metric.my_metric` — ROC-AUC（コンペの評価指標）
- `Submission.create_submission` — 予測値から提出用CSVを生成

### 設定 (`configs/config.py`)

パス定数（`DIR_*`）とファイル名（`FILE_NAME_*`）を定義。`os.path.abspath(os.path.dirname(os.path.abspath("")))` をホームディレクトリとして使用（`notebooks/` から実行時にプロジェクトルートに解決される）。

## 主要な規約

- 中間データはすべてpickle形式で保存
- 特徴量DataFrameは `['社員番号', 'category']` をキーとし、重複不可
- モデル成果物は `models/{run_name}/{run_name}_fold-{i}/` に保存
- ログは `logs/{run_name}/` に出力
- `sample_code/` には1位・3位の参考ソリューションを格納
