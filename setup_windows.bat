@echo off
rem One-time setup. Run from "Miniforge Prompt" inside the cloned folder.
cd /d "%~dp0"
powershell -NoProfile -Command "if ((Get-Location).Path -match '[^\x00-\x7F]') { exit 1 }"
if errorlevel 1 (
  echo [error] The folder path contains Korean or other non-ASCII characters.
  echo         mamba cannot read files there. Move the project to e.g. C:\work\cave-drone-sim
  exit /b 1
)
rem mamba ships with Miniforge; plain "conda env create" hits a launcher bug in conda 26.x.
call mamba env create -f environment.yml -y || call mamba env update -n cave-drone -f environment.yml -y || exit /b 1
call conda activate cave-drone || exit /b 1
python -m cavesim.check
