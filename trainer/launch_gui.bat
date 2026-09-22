@echo off
setlocal enabledelayedexpansion
title iw3 inpaint trainer - GUI
rem ---------------------------------------------------------------------------
rem Single entry point. Finds any Python 3, starts the local web GUI and opens
rem it in your browser. The GUI itself only needs the standard library; nunif's
rem own Python (the one with torch) is used for the actual work, and you point
rem the GUI at it once on the first run.
rem ---------------------------------------------------------------------------
set "GUI_DIR=%~dp0"
set "PY="

rem 1. an interpreter you chose explicitly
if defined NT_PYTHON if exist "%NT_PYTHON%" set "PY=%NT_PYTHON%"

rem 2. a nunif install sitting next to this folder (the usual layout:
rem    <install>\New_Trainer\GUI\ , with <install>\python\python.exe )
if not defined PY if exist "%GUI_DIR%..\..\python\python.exe" set "PY=%GUI_DIR%..\..\python\python.exe"
if not defined PY if exist "%GUI_DIR%..\python\python.exe"    set "PY=%GUI_DIR%..\python\python.exe"
if not defined PY if exist "%GUI_DIR%python\python.exe"       set "PY=%GUI_DIR%python\python.exe"

rem 3. anything on PATH
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  where py >nul 2>nul && set "PY=py"
)

if not defined PY (
  echo.
  echo   No Python was found.
  echo.
  echo   This GUI needs any Python 3.8 or newer to show its pages. If you have
  echo   nunif installed, the simplest fix is to copy this GUI folder into your
  echo   nunif install so it sits next to nunif's own python folder.
  echo.
  echo   Otherwise install Python from https://www.python.org/downloads/
  echo   and tick "Add python.exe to PATH" during setup.
  echo.
  pause
  exit /b 1
)

"%PY%" "%GUI_DIR%app\server.py" %*
if errorlevel 1 (
  echo.
  echo   The GUI stopped with an error - the message above says why.
  pause
)
exit /b 0
