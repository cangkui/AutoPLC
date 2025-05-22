@echo off
setlocal

:: Paths
set PYTHON_PATH=..\asevenv\Scripts\python.exe
set MAIN_PATH=..\src\autoplc\main.py

:: Config files
set CONFIGS=default wo-apirec wo-debugger wo-planner wo-retrival

:: Datasets
set DATASETS=competition lgf githubcase agents4plc

:: Loop over each config
for %%C in (%CONFIGS%) do (
    echo ===============================
    echo Running config: %%C
    echo ===============================

    :: Loop over each dataset
    for %%D in (%DATASETS%) do (
        echo -------------------------------
        echo Running on dataset: %%D
        echo -------------------------------
        %PYTHON_PATH% %MAIN_PATH% --config %%C --benchmark %%D
    )
)

endlocal
