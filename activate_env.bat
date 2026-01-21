@echo off
REM 激活 conda 环境脚本 (Windows)
echo 激活 llmNano 环境...

REM 设置 OpenMP 环境变量以解决冲突
set KMP_DUPLICATE_LIB_OK=TRUE

REM 检查 conda 是否可用
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到 conda 命令
    echo 请使用 Anaconda Prompt 运行此脚本
    pause
    exit /b 1
)

call conda activate llmNano

if %ERRORLEVEL% NEQ 0 (
    echo 错误: 激活环境失败，请先运行 setup_env.bat 创建环境
    pause
    exit /b 1
)

echo 环境已激活！
echo OpenMP 环境变量已设置: KMP_DUPLICATE_LIB_OK=TRUE
echo.
echo 当前 Python 路径:
python --version
echo.
echo 当前工作目录:
cd
echo.
cmd /k
