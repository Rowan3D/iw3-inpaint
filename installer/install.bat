@echo off
setlocal enabledelayedexpansion
title iw3 inpainting extras - installer
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem One click. This folder can live anywhere - Desktop, Downloads, a USB stick.
rem
rem All it does here is find a Python to run installer\install.py with. That is
rem awkward only because a nunif-windows install keeps its own Python inside
rem itself and does not put it on PATH, so:
rem
rem   1. NUNIF_DIR, if you set it
rem   2. a nunif install around this folder (if you unzipped it inside one)
rem   3. any python on PATH
rem   4. ask for the iw3 folder and use the Python next to it
rem
rem The script itself does the real work and asks everything else.
rem ---------------------------------------------------------------------------

set "PY="
set "NUNIF="

rem --- 1 / 2: a nunif install we can see from here -----------------------------
if defined NUNIF_DIR call :try_root "%NUNIF_DIR%"
if not defined PY call :try_root "%CD%"
if not defined PY call :try_root "%CD%\.."
if not defined PY call :try_root "%CD%\..\.."
if not defined PY call :try_root "%CD%\..\..\.."

rem --- 3: anything on PATH -----------------------------------------------------
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  where py >nul 2>nul && set "PY=py"
)

rem --- 4: ask, and use the Python that lives next to it ------------------------
if not defined PY (
  echo.
  echo   ------------------------------------------------------------------
  echo    Where is iw3 installed?
  echo   ------------------------------------------------------------------
  echo.
  echo    Type the folder that has  nunif\  and  iw3\  inside it, for example
  echo.
  echo       C:\nunif-windows\nunif
  echo.
  set /p NUNIF=   Path:
  if defined NUNIF call :try_root "!NUNIF!"
)

if not defined PY (
  echo.
  echo   No Python was found, so nothing has been changed.
  echo.
  echo   Two ways to fix it:
  echo     * put this folder inside your nunif install and run it again, or
  echo     * install Python from https://www.python.org/downloads/
  echo       ticking "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

if defined NUNIF (
  "%PY%" "%~dp0installer\install.py" --nunif "%NUNIF%" %*
) else (
  "%PY%" "%~dp0installer\install.py" %*
)
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo   The installer stopped with code %RC% - the message above says why.
  pause
)
exit /b %RC%

rem ---------------------------------------------------------------------------
rem :try_root <folder>   sets PY and NUNIF when <folder> looks like an install
rem Accepts either the repository root (has nunif\ and iw3\) or the install root
rem above it (has nunif\nunif\ and python\).
rem ---------------------------------------------------------------------------
:try_root
set "CAND=%~f1"
if "%CAND%"=="" goto :eof
if exist "%CAND%\nunif\" if exist "%CAND%\iw3\" (
  set "NUNIF=%CAND%"
  if exist "%CAND%\..\python\python.exe" set "PY=%CAND%\..\python\python.exe"
  if not defined PY if exist "%CAND%\..\venv\Scripts\python.exe" set "PY=%CAND%\..\venv\Scripts\python.exe"
  goto :eof
)
if exist "%CAND%\nunif\nunif\" if exist "%CAND%\nunif\iw3\" (
  set "NUNIF=%CAND%\nunif"
  if exist "%CAND%\python\python.exe" set "PY=%CAND%\python\python.exe"
  if not defined PY if exist "%CAND%\venv\Scripts\python.exe" set "PY=%CAND%\venv\Scripts\python.exe"
  goto :eof
)
goto :eof
