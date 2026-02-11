# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

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

# uvプロジェクトの初期化
cd /workspace/{プロジェクト名}/
uv init
uv venv

# 開発用のライブラリをインストール
uv add --dev ipykernel nbstripout

# gitのセットアップ
git config --global user.email "runpod@example.com"
git config --global user.name "runpod"
nbstripout --install
