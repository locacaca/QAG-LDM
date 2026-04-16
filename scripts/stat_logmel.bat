@echo off
REM Windows 批处理脚本：统计 logmel_l1_results.csv 中不同音轨的 logmel L1 结果
REM 使用方法: scripts\stat_logmel.bat [CSV文件路径] [输出JSON路径(可选)]

setlocal

set CSV_FILE=%1
if "%CSV_FILE%"=="" set CSV_FILE=outputs_batch_src_extract_quality\logmel_l1_results.csv

set OUTPUT_JSON=%2

if "%OUTPUT_JSON%"=="" (
    python tools\stat_logmel_by_instrument.py --csv "%CSV_FILE%"
) else (
    python tools\stat_logmel_by_instrument.py --csv "%CSV_FILE%" --output "%OUTPUT_JSON%"
)

endlocal

