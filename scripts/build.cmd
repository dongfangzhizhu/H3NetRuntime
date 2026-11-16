@echo off
rem Full build: native DLL + .NET solution. Usage: build.cmd [CUDA_ARCH]
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

pushd "%~dp0.."
set "ROOT=%CD%"
set "ARCH=89"
if not "%~1"=="" set "ARCH=%~1"

echo === [1/2] Native runtime ===
call "%ROOT%\scripts\build-native.cmd" "%ARCH%"
if errorlevel 1 goto :fail

echo === [2/2] .NET solution ===
set "DIST=%ROOT%\dist\win-x64"
set "H3RT_NATIVE_DIR=%DIST%"
dotnet build "%ROOT%\dotnet\H3NetRuntime.sln" -c Release --nologo
if errorlevel 1 goto :fail
copy /Y "%DIST%\h3rt.dll" "%ROOT%\dotnet\samples\H3NetRuntime.Sample\bin\Release\net8.0\" >nul 2>&1
copy /Y "%DIST%\h3rt.dll" "%ROOT%\dotnet\tests\H3NetRuntime.Smoke\bin\Release\net8.0\" >nul 2>&1
copy /Y "%DIST%\h3rt.dll" "%ROOT%\dotnet\src\H3NetRuntime.WebUI\bin\Release\net8.0\" >nul 2>&1

echo.
echo OK. Native DLL: %DIST%\h3rt.dll
echo     C smoke:      "%DIST%\h3rt_smoke.exe" [model-dir]
echo     .NET sample:  dotnet run --project "%ROOT%\dotnet\samples\H3NetRuntime.Sample" -c Release -- --help
echo     .NET smoke:   dotnet run --project "%ROOT%\dotnet\tests\H3NetRuntime.Smoke" -c Release
echo     WebUI 通用版: scripts\run-webui.cmd
echo     WebUI 高级版: scripts\run-webui-advanced.cmd
popd
exit /b 0

:fail
echo.
echo BUILD FAILED. See the output above.
popd
exit /b 1
