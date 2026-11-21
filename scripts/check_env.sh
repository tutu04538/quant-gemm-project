#!/usr/bin/env bash
set -u
for tool in nvidia-smi nvcc ncu nsys compute-sanitizer python; do
    if command -v "$tool" >/dev/null 2>&1; then
        printf '%-20s %s\n' "$tool" "$(command -v "$tool")"
    else
        printf '%-20s %s\n' "$tool" '未在 PATH 中找到'
    fi
done
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
fi
