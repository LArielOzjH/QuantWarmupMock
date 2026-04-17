#!/usr/bin/env bash
# 一键生成初赛提交包：96.zip
# 用法：bash pack.sh
set -e

tar czf submission.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='charts' \
    --exclude='doc' \
    --exclude='report' \
    --exclude='*.zip' \
    --exclude='*.tar.gz' \
    run.sh setup.sh contestant/

zip 96.zip submission.tar.gz
rm submission.tar.gz

echo "Done: 96.zip"
echo "提交邮件至 hackathon@ubiquant.com，主题：初赛代码-quan't"
