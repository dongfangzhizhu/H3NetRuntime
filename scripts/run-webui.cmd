@echo off
rem Starts the general edition (通用版) of the H3 WebUI.
rem Any extra arguments are forwarded, e.g.  run-webui.cmd --port 8080
setlocal EnableDelayedExpansion

set "ROOT=%~dp0.."
pushd "%ROOT%"

if not defined H3_MODEL_DIR set "H3_MODEL_DIR=D:\AI\convert\minimax\h3-models\pdd8"

set "APP=dotnet\src\H3NetRuntime.WebUI\bin\Release\net8.0\H3NetRuntime.WebUI.dll"
if not exist "%APP%" (
    echo [!] %APP% not found. Run scripts\build.cmd first.
    popd
    exit /b 1
)

echo Starting H3 WebUI (通用版) -> http://127.0.0.1:7867/
dotnet "%APP%" --edition general --model-dir "%H3_MODEL_DIR%" %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
