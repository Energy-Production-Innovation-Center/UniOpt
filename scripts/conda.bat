@echo off

for /f %%i in ('python scripts\get_env_name.py') do set "ENV_NAME=%%i"

conda info --envs | findstr "%ENV_NAME%"

IF %ERRORLEVEL% NEQ 0 (
    echo Creating environment %ENV_NAME%
    call conda create -y -n %ENV_NAME% --no-default-packages python=3.12
    call conda run --live-stream -n %ENV_NAME% pip install --upgrade -e .[dev]
) ELSE (
    echo Environment %ENV_NAME% already exists.
)

IF "%~1"=="" (
    call conda run --live-stream -n %ENV_NAME% pip install --upgrade -e .[dev]
) ELSE (
    call conda run --live-stream -n %ENV_NAME% %*
)
