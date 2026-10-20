@echo off
title Unity ML-Agents Training Runner
cd /d "%~dp0"

powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0train.ps1"
