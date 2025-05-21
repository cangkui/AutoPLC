@echo off
setlocal

:: 解释器路径
set PYTHON=..\asevenv\Scripts\python.exe

:: config 列表
set configs=default.yaml wo-apirec.yaml wo-debugger.yaml wo-planner.yaml wo-retrival.yaml

:: benchmark 数据集列表
set benchmarks=competition lgf githubcase agents4plc

:: 遍历 configs 和 benchmarks
for %%c in (%configs%) do (
    for %%b in (%benchmarks%) do (
        echo Running config=%%c on benchmark=%%b
        %PYTHON% ..\src\autoplc\main.py --config %%c --benchmark %%b
    )
)

echo All experiments finished.
pause
