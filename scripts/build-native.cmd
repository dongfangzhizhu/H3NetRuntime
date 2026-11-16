@echo off
rem Build only the native h3rt DLL (no .NET). Usage: build-native.cmd [CUDA_ARCH]
rem
rem The MSVC environment is set up explicitly from the installed toolchain
rem directories instead of calling vcvars64.bat: vcvars queries the registry,
rem which is unavailable in locked-down/CI sandboxes. Everything it would set
rem (PATH/INCLUDE/LIB) is derived from files on disk here.
rem Note: paths containing "(x86)" must use delayed expansion inside blocks.
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

pushd "%~dp0.."
set "ROOT=%CD%"
set "ARCH=89"
if not "%~1"=="" set "ARCH=%~1"

echo === [1/3] Locating MSVC and Windows SDK ===
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Install Visual Studio with the C++ workload.
    popd & exit /b 1
)
set "VS="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS (
    echo ERROR: no Visual Studio installation with MSVC C++ tools.
    popd & exit /b 1
)
set "VCTOOLS="
for /d %%i in ("!VS!\VC\Tools\MSVC\*") do set "VCTOOLS=%%~fi"
if not defined VCTOOLS (
    echo ERROR: MSVC tools directory not found under !VS!\VC\Tools\MSVC.
    popd & exit /b 1
)
set "WINSDK=%ProgramFiles(x86)%\Windows Kits\10"
if not exist "%WINSDK%\Include" set "WINSDK=%ProgramFiles%\Windows Kits\10"
set "SDKVER="
for /d %%i in ("!WINSDK!\Include\*") do set "SDKVER=%%~nxi"
if not defined SDKVER (
    echo ERROR: Windows 10 SDK headers not found under !WINSDK!\Include.
    popd & exit /b 1
)
set "SDKBIN=!WINSDK!\bin\!SDKVER!\x64"
set "PATH=!VCTOOLS!\bin\HostX64\x64;!SDKBIN!;!PATH!"
set "INCLUDE=!VCTOOLS!\include;!WINSDK!\Include\!SDKVER!\ucrt;!WINSDK!\Include\!SDKVER!\shared;!WINSDK!\Include\!SDKVER!\um;!WINSDK!\Include\!SDKVER!\winrt;!WINSDK!\Include\!SDKVER!\cppwinrt"
set "LIB=!VCTOOLS!\lib\x64;!WINSDK!\Lib\!SDKVER!\ucrt\x64;!WINSDK!\Lib\!SDKVER!\um\x64"
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe is not on PATH.
    popd & exit /b 1
)
echo     MSVC:  !VCTOOLS!
echo     SDK:   !SDKVER!
echo     CUDA:  sm_!ARCH!

echo === [2/3] Building native h3rt ===
rem CMake cache files interpret backslashes as escapes: use forward slashes.
set "CL_FWD=!VCTOOLS!\bin\HostX64\x64\cl.exe"
set "CL_FWD=!CL_FWD:\=/!"
set "RC_FWD=!SDKBIN!\rc.exe"
set "RC_FWD=!RC_FWD:\=/!"
set "MT_FWD=!SDKBIN!\mt.exe"
set "MT_FWD=!MT_FWD:\=/!"
set "ROOT_FWD=%ROOT%"
set "ROOT_FWD=!ROOT_FWD:\=/!"
cmake -S "%ROOT%" -B "%ROOT%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  "-DH3_CUDA_ARCH=%ARCH%" "-DH3_ENGINE_DIR=!ROOT_FWD!/native/engine" ^
  "-DCMAKE_C_COMPILER=!CL_FWD!" ^
  "-DCMAKE_CXX_COMPILER=!CL_FWD!" ^
  "-DCMAKE_RC_COMPILER=!RC_FWD!" ^
  "-DCMAKE_MT=!MT_FWD!"
if errorlevel 1 goto :fail
cmake --build "%ROOT%\build" --target h3rt h3rt_smoke --parallel
if errorlevel 1 goto :fail

echo === [3/3] Staging native binaries into dist\win-x64 ===
set "DIST=%ROOT%\dist\win-x64"
if not exist "%DIST%" mkdir "%DIST%"
copy /Y "%ROOT%\build\bin\h3rt.dll" "%DIST%\" >nul
copy /Y "%ROOT%\build\bin\h3rt.lib" "%DIST%\" >nul
copy /Y "%ROOT%\build\bin\h3rt_smoke.exe" "%DIST%\" >nul
if not exist "%DIST%\h3rt.dll" goto :fail

echo OK. Native DLL: %DIST%\h3rt.dll
popd
exit /b 0

:fail
echo BUILD FAILED. See the output above.
popd
exit /b 1
