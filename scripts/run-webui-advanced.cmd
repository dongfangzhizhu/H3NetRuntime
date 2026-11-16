@echo off
rem Starts the advanced edition (高级版) of the H3 WebUI: full engine knobs,
rem model pack switching, ordered Ref2VA references, phase timing breakdown.
rem Any extra arguments are forwarded, e.g.  run-webui-advanced.cmd --port 8080
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

echo Starting H3 WebUI (高级版) -> http://127.0.0.1:7867/
dotnet "%APP%" --edition advanced --model-dir "%H3_MODEL_DIR%" %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
