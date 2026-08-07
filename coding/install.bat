@echo off
REM Install dependencies for the KV cache benchmarking framework on Windows.

setlocal enabledelayedexpansion
set PYTHON=python

echo ==^> Creating virtual environment (.venv) if it does not exist
if not exist ".venv" (
  %PYTHON% -m venv .venv
)
call .venv\Scripts\activate.bat

echo ==^> Upgrading pip
python -m pip install --upgrade pip wheel setuptools

echo ==^> Installing requirements
pip install -r requirements.txt || echo Some optional packages may have failed.

echo ==^> Installing PyTorch (CPU build by default on Windows)
pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch

echo ==^> Installing transformers ecosystem
pip install transformers accelerate sentencepiece tqdm psutil pyyaml tabulate matplotlib pandas seaborn

echo ==^> vLLM and bitsandbytes are not officially supported on native Windows.
echo ==^> PagedAttention experiments will be skipped on Windows; the framework falls back to a CPU simulation.
echo ==^> For full PagedAttention support, please use WSL2 or Linux.

echo ==^> Installing the kvbench package in editable mode
pip install -e .

echo ==^> Done. Activate with:  .venv\Scripts\activate
endlocal
