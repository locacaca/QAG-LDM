#!/bin/bash
# 统计 logmel_l1_results.csv 中不同音轨的 logmel L1 结果
# 使用方法: ./scripts/stat_logmel.sh [CSV文件路径] [输出JSON路径(可选)]

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"

CSV_FILE="${1:-outputs_batch_src_extract_quality/logmel_l1_results.csv}"
OUTPUT_JSON="${2:-}"

if [ -z "$OUTPUT_JSON" ]; then
    python tools/stat_logmel_by_instrument.py --csv "${CSV_FILE}"
else
    python tools/stat_logmel_by_instrument.py --csv "${CSV_FILE}" --output "${OUTPUT_JSON}"
fi

