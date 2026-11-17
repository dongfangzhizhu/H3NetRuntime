@echo off
rem Starts the advanced edition (高级版) of the H3 WebUI: full engine knobs,
rem model pack switching, ordered Ref2VA references, phase timing breakdown.
rem
rem Model pack discovery order (see ModelLocator.cs):
rem   1. --model-dir / H3:ModelDir
rem   2. H3_MODEL_DIR environment variable
rem   3. h3.config.json  "h3_dir"
rem   4. %USERPROFILE%\h3-models\pdd8
rem   5. <app>\models\pdd8, <app>\models
rem Only an explicit override is forwarded when H3_MODEL_DIR is defined.
rem
rem Any extra arguments are forwarded, e.g.  run-webui-advanced.cmd --port 8080
setlocal EnableDelayedExpansion

set "ROOT=%~dp0.."
pushd "%ROOT%"

set "APP=dotnet\src\H3NetRuntime.WebUI\bin\Release\net8.0\H3NetRuntime.WebUI.dll"
if not exist "%APP%" (
    echo [!] %APP% not found. Run scripts\build.cmd first.
    popd
    exit /b 1
)

set "MODEL_ARGS="
if defined H3_MODEL_DIR set MODEL_ARGS=--model-dir "%H3_MODEL_DIR%"

echo Starting H3 WebUI (高级版) -> http://127.0.0.1:7867/
if defined H3_MODEL_DIR (echo   model pack: %H3_MODEL_DIR%) else (echo   model pack: ^(auto-detect^))
dotnet "%APP%" --edition advanced %MODEL_ARGS% %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
