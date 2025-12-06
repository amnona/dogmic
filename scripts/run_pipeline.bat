@echo off
REM --- Replace these with your actual paths ---
SET CONDA_ENV_NAME=dogmic
SET SCRIPT_PATH=C:\Users\User\git\dogmic\scripts
SET SCRIPT_FILE=pipeline_movie.py
SET VIDEO_PATH=F:\dogcam\video\dogcam1
SET CONDA_ACTIVATE_PATH=E:\programs\miniconda3\Scripts\activate.bat
SET CONDA_DEACTIVATE_PATH=E:\programs\miniconda3\Scripts\deactivate.bat

REM --- Activate the Conda environment ---
CALL "%CONDA_ACTIVATE_PATH%" %CONDA_ENV_NAME%

REM --- Run the Python script ---
IF %ERRORLEVEL% EQU 0 (
    cd "%SCRIPT_PATH%"
    python "%SCRIPT_FILE%" --dir "%VIDEO_PATH%"
) ELSE (
    ECHO Failed to activate Conda environment.
)

REM --- Deactivate the environment (optional, but good practice) ---
REM CALL "%CONDA_DEACTIVATE_PATH%"

REM --- Keep the window open if you want to see errors when testing manually ---
REM PAUSE
