# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

ATMA Cup 20 (Udemy) 機械学習コンペティションのソリューション。社員のUdemy学習活動・HR/DX研修・キャリアアンケート・残業データ・職位履歴をもとに、二値の目的変数を予測する（評価指標: ROC-AUC）。主キーは `['社員番号', 'category']`。

データの詳細は./data_description.mdに記載。

## アーキテクチャ

### EDA

以下の観点は最低限確認する。

- 訓練セットとテストセットとのカラムの違い
- **質的変数**: 各カテゴリの重複度合い。
- **量的変数**: 訓練セットとテストセットとの分布の違い。

### Model-Runner-Notebookパターン

ロジックはsrc/配下にモジュールとして記述し実行・検証・可視化はnotebooks/配下でノートブックを作成し実行する。

- **Model抽象クラス**: `src/model.py` で `train()`, `predict()`, `save_model()`, `load_model()` を定義
- **具体的なモデル**: `src/model_resnet.py`, `src/model_arcface.py` などでModel継承
- **Runnerクラス**: `src/runner.py` でCV管理・学習・予測・評価を一元管理
- **Notebookで実行**: `notebooks/exp_*.ipynb` でパラメータ設定→Runner実行→分析

### 特徴量システム (`src/feature.py`)

全特徴量は `FeatureBase` を継承し、`_create_feature() -> pd.DataFrame` を実装する。基底クラスが提供する機能:
- **キャッシュ**: `use_cache`（読込）と `save_cache`（書込）で制御。`data/features/{クラス名}.pkl` に保存
- **主キー整合性チェック**: キーカラムの存在確認と重複チェック

ノートブックから `create_feature()` を呼び出して使用する。各特徴量クラスは `data/interim/` のpickleを読み込み、`['社員番号', 'category']` をキーとするDataFrameを返す。


### モデルシステム

- `src/model.py` — 抽象基底クラス `Model`。`train`, `predict`, `save_model`, `load_model` を定義

### 学習パイプライン (`src/runner.py`)

`Runner` がCV学習ループ全体を管理:
- `StratifiedGroupKFold` を使用（`社員番号` でグループ化、targetで層化）
- `run_train_cv()` → fold毎にモデルを学習・保存
- `run_metric_cv()` → 全foldの評価を実行、スコア（mean, std, fold別）をログ出力
- `run_predict_cv()` → テストデータに対してfold予測の平均値を算出
- `plot_feature_importance_cv()` → gain重要度の上位100件を変動係数とともにプロット
- `after_split_process`（分割後のデータ変換）と `after_predict_process`（予測後の変換）フックに対応

### ディレクトリ構成

```
.
├── README.md                     # リポジトリ全体の簡易概要
├── requirements.txt              # 依存パッケージ
├── configs/
│   └── config.py                 # 設定値（定数/パス）を集中管理
├── data/
│   ├── raw/                      # 生データ（配布物・外部取得データ）
│   │   └── input/                # コンペ配布の入力ファイル類
│   ├── interim/                  # 中間生成物（フィルタ後/正規化後など）
│   ├── features/                 # 特徴量生成結果（数値・埋め込み等）
│   ├── figures/                  # 可視化出力（PNG/HTML等）
│   └── submission/               # 提出用CSVの生成先
├── logs/                         # 実行ログ（日時付きファイル推奨）
├── models/                       # モデル/チェックポイント（必要時のみ）
├── notebooks/                    # 実験/実行用 Notebook（パラメータ調整・可視化・実行）
├── sample_code/                  # 参考サンプル（他人のコードなど）
├── src/                          # ロジックを記載したPythonモジュール群
├── tmp/                          # 検証などで利用するファイルやコード群
├── CLAUDE.md                   　# このリポジトリの開発/運用ガイド
└── data_discription.md           # データセットの詳細説明
```

## コード規約

### コメント

- docstringはGoogle styleで書く
- コメントは日本語で記述する
- コメントは#の数で階層化する。数が少ないほど上位階層となる。

### 変数名

- 以下のデータ型についてはprefixにデータ型を付与する。
    - df_: pandas Dataframe
    - pdf_: polars Dataframe
    - dict_: 辞書
    - list_: リスト
    - set_: セット



