# 新しいpodの作成時に実行する
apt update 
apt install -y git curl

# uvのセットアップ
# uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh
# curl -LsSf https://astral.sh/uv/install.sh | sh
# cp ~/.local/bin/uv /workspace/.local/bin/uv

# パスを通す
source $HOME/.local/bin/env
# export PATH=/workspace/.local/bin:$PATH

## uvのキャッシュを/workspace配下にする
export UV_CACHE_DIR=/workspace/.cache/uv
mkdir -p /workspace/.cache/uv

# gitのセットアップ
git clone https://github.com/snoopy0420/atma_udemy.git
git config --global user.email "runpod@example.com"
git config --global user.name "runpod"

cd /workspace/atma_udemy/
