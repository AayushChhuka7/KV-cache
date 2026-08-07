@echo off
REM Run the full set of KV cache benchmark experiments on Windows.
setlocal

set OUT_DIR=results\%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%
set OUT_DIR=%OUT_DIR: =0%
mkdir "%OUT_DIR%" 2>nul

python scripts\run_all.py --output "%OUT_DIR%" %*

endlocal
