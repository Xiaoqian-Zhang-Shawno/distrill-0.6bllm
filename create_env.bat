@echo off
REM 创建 conda 环境的简化脚本
REM 如果 conda 命令不可用，请使用 Anaconda Prompt 运行

echo ========================================
echo 创建 llmNano Conda 环境
echo ========================================
echo.

REM 尝试查找 conda
set CONDA_FOUND=0

REM 方法 1: 直接使用 conda 命令
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set CONDA_FOUND=1
    goto :create_env
)

REM 方法 2: 尝试常见的 conda 路径
set CONDA_PATHS[0]=%USERPROFILE%\Anaconda3\Scripts\conda.exe
set CONDA_PATHS[1]=%USERPROFILE%\Miniconda3\Scripts\conda.exe
set CONDA_PATHS[2]=C:\ProgramData\Anaconda3\Scripts\conda.exe
set CONDA_PATHS[3]=C:\ProgramData\Miniconda3\Scripts\conda.exe

for /L %%i in (0,1,3) do (
    call set "PATH_TEST=%%CONDA_PATHS[%%i]%%"
    if exist "!PATH_TEST!" (
        set "CONDA_CMD=!PATH_TEST!"
        set CONDA_FOUND=1
        goto :create_env
    )
)

:create_env
if %CONDA_FOUND% EQU 0 (
    echo [错误] 未找到 conda 命令
    echo.
    echo 请使用以下方法之一:
    echo 1. 使用 Anaconda Prompt 运行此脚本
    echo 2. 确保 Anaconda/Miniconda 已安装并添加到 PATH
    echo 3. 手动运行以下命令:
    echo    conda create -n llmNano python=3.10 -y
    echo    conda activate llmNano
    echo    pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [信息] 找到 conda，开始创建环境...
echo.

echo [步骤 1/3] 创建 conda 环境 llmNano (Python 3.10)
if defined CONDA_CMD (
    "%CONDA_CMD%" create -n llmNano python=3.10 -y
) else (
    conda create -n llmNano python=3.10 -y
)

if %ERRORLEVEL% NEQ 0 (
    echo [错误] 创建环境失败
    pause
    exit /b 1
)

echo.
echo [步骤 2/3] 激活环境
if defined CONDA_CMD (
    call "%CONDA_CMD%" activate llmNano
) else (
    call conda activate llmNano
)

if %ERRORLEVEL% NEQ 0 (
    echo [警告] 自动激活失败，请手动运行: conda activate llmNano
    echo.
    echo [步骤 3/3] 安装依赖
    echo 请在激活环境后运行: pip install -r requirements.txt
    pause
    exit /b 0
)

echo.
echo [步骤 3/3] 安装项目依赖
python -m pip install --upgrade pip
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo [错误] 安装依赖失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo [成功] 环境设置完成！
echo ========================================
echo.
echo 使用以下命令激活环境:
echo   conda activate llmNano
echo.
echo 或运行: activate_env.bat
echo.
pause
