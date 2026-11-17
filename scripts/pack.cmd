@echo off
rem Builds the H3NetRuntime NuGet package with the native runtime inside.
rem Usage: pack.cmd [CUDA_ARCH] [--skip-native]
rem
rem The package is only useful with a native library for the target platform, so
rem this fails rather than shipping a managed-only package that throws at the
rem first P/Invoke.
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

pushd "%~dp0.."
set "ROOT=%CD%"
set "ARCH=89"
set "SKIP_NATIVE="
for %%a in (%*) do (
    if /I "%%~a"=="--skip-native" (set "SKIP_NATIVE=1") else (set "ARCH=%%~a")
)

set "WIN_DLL=%ROOT%\dist\win-x64\h3rt.dll"
set "LINUX_SO=%ROOT%\dist\linux-x64\libh3rt.so"

echo === [1/3] Native runtime ===
if defined SKIP_NATIVE (
    echo     skipped ^(--skip-native^)
) else if exist "%WIN_DLL%" (
    echo     using existing %WIN_DLL%
) else (
    call "%ROOT%\scripts\build-native.cmd" "%ARCH%"
    if errorlevel 1 goto :fail
)

if not exist "%WIN_DLL%" if not exist "%LINUX_SO%" (
    echo.
    echo ERROR: no native runtime under dist\win-x64 or dist\linux-x64.
    echo        Run scripts\build-native.cmd %ARCH% ^(Windows^) or scripts\build.sh %ARCH% ^(Linux^) first.
    goto :fail
)
if not exist "%LINUX_SO%" (
    echo     WARNING: dist\linux-x64\libh3rt.so is missing; the package will only
    echo              contain the Windows runtime. Build it on a Linux box
    echo              ^(bash scripts/build.sh %ARCH%^) and re-run pack.cmd.
)

echo === [2/3] dotnet pack ===
if exist "%ROOT%\dist\nuget" rd /s /q "%ROOT%\dist\nuget"
rem The version lives in the csproj; H3RT_PACKAGE_VERSION only overrides it.
set "VERSIONARG="
if defined H3RT_PACKAGE_VERSION set "VERSIONARG=-p:Version=%H3RT_PACKAGE_VERSION%"
dotnet pack "%ROOT%\dotnet\src\H3NetRuntime\H3NetRuntime.csproj" -c Release --nologo ^
    -o "%ROOT%\dist\nuget" %VERSIONARG%
if errorlevel 1 goto :fail

echo === [3/3] Contents ===
set "PKG="
for %%f in ("%ROOT%\dist\nuget\*.nupkg") do set "PKG=%%~ff"
if not defined PKG (
    echo ERROR: no .nupkg produced.
    goto :fail
)
echo     %PKG%
rem Zip keeps file names in the central directory as plain text, so findstr can
rem check the payload without unzip/bsdtar being installed.
call :member "runtimes/win-x64/native/h3rt.dll"
call :member "runtimes/linux-x64/native/libh3rt.so"
call :member "build/native/include/h3rt.h"
call :member "lib/net8.0/H3NetRuntime.dll"

echo.
echo OK. Publish with:
echo     dotnet nuget push "%PKG%" --source ^<your-feed^> --api-key ^<key^>
popd
exit /b 0

:member
findstr /I /M /C:"%~1" "%PKG%" >nul 2>&1
if errorlevel 1 (echo     [ ] %~1) else (echo     [x] %~1)
exit /b 0

:fail
echo.
echo PACK FAILED. See the output above.
popd
exit /b 1
