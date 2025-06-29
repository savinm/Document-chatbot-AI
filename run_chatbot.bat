@echo off
SETLOCAL

:: Check if streamlit is installed
python -c "import streamlit" 2>NUL

IF %ERRORLEVEL% NEQ 0 (
    echo Streamlit not found. Installing...
    pip install streamlit
) ELSE (
    echo Streamlit is already installed. Skipping installation.
)

ENDLOCAL


start streamlit run chatbot_grog_v9.py

