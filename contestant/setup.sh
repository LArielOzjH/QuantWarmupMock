#!/usr/bin/env bash
# 环境安装脚本（平台在首次运行前执行，结果会被缓存）
set -e

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r contestant/requirements.txt

# sglang 从本地 clone 安装（editable，修改源码即时生效）
# 提交时 sglang/ 目录与 contestant/ 一起打包进 .tar.gz
if [ -d "sglang" ]; then
    pip install -e "sglang/python/[all]"
else
    echo "WARNING: sglang/ directory not found, installing from PyPI as fallback"
    pip install sglang
fi
