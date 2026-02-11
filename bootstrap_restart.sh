# podのリスタート時に実行する
# ターミナルを再起動すると環境変数のこの変更が確認できる。
# windowをリロードするとソース管理も反映される。
# vscodeのフォルダもatma_udemyを開きカーネルを作成する。

# bashスクリプトの安全装置3点セット
set -euo pipefail

# uvのセットアップ
## インストール
curl -LsSf https://astral.sh/uv/install.sh | sh
## パスを通す
export PATH=/root/.local/bin:$PATH
## uvのキャッシュを/workspace配下にする
export UV_CACHE_DIR=/workspace/.cache/uv
mkdir -p /workspace/.cache/uv

# パッケージのインストール(不要？)
cd /workspace/atma_udemy/
uv sync

# gitのセットアップ
git config --global user.email "runpod@example.com"
git config --global user.name "runpod"
nbstripout --install
