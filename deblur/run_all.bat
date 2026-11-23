@echo off
cd /d "%~dp0"
set PY="C:\Program Files\Python312\python.exe"

echo === ours: train ===        & %PY% train.py --model ours            >> run_all_log.txt 2>&1 || goto :fail
echo === ours: predict ===      & %PY% train.py --model ours --predict  >> run_all_log.txt 2>&1 || goto :fail
echo === nafnet: train ===      & %PY% train.py --model nafnet          >> run_all_log.txt 2>&1 || goto :fail
echo === nafnet: predict ===    & %PY% train.py --model nafnet --predict >> run_all_log.txt 2>&1 || goto :fail
echo === all done ===
goto :eof

:fail
echo *** a step failed, see run_all_log.txt ***