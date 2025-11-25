@echo off
REM Batch file to check dependencies and run the RAG QA Chatbot
REM This script checks for required Python packages and installs them if missing

echo ========================================
echo Multi-Modal RAG QA Chatbot Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python found
python --version
echo.

REM Check if pip is available
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    pause
    exit /b 1
)

echo [2/4] Checking dependencies...
echo.

REM Check and install required packages
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing streamlit...
    python -m pip install streamlit --quiet
)

python -c "import transformers" >nul 2>&1
if errorlevel 1 (
    echo Installing transformers...
    python -m pip install transformers --quiet
)

python -c "import sentence_transformers" >nul 2>&1
if errorlevel 1 (
    echo Installing sentence-transformers...
    python -m pip install sentence-transformers --quiet
)

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Installing torch...
    python -m pip install torch --quiet
)

python -c "import faiss" >nul 2>&1
if errorlevel 1 (
    echo Installing faiss-cpu...
    python -m pip install faiss-cpu --quiet
)

python -c "import langchain" >nul 2>&1
if errorlevel 1 (
    echo Installing langchain packages...
    python -m pip install langchain langchain-community langchain-text-splitters --quiet
)

python -c "import sklearn" >nul 2>&1
if errorlevel 1 (
    echo Installing scikit-learn...
    python -m pip install scikit-learn --quiet
)

python -c "import gensim" >nul 2>&1
if errorlevel 1 (
    echo Installing gensim...
    python -m pip install gensim --quiet
)

python -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo Installing pandas...
    python -m pip install pandas --quiet
)

python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing numpy...
    python -m pip install numpy --quiet
)

python -c "import fitz" >nul 2>&1
if errorlevel 1 (
    echo Installing PyMuPDF...
    python -m pip install PyMuPDF --quiet
)

python -c "import pdfplumber" >nul 2>&1
if errorlevel 1 (
    echo Installing pdfplumber...
    python -m pip install pdfplumber --quiet
)

python -c "import pytesseract" >nul 2>&1
if errorlevel 1 (
    echo Installing pytesseract...
    python -m pip install pytesseract --quiet
)

python -c "import PIL" >nul 2>&1
if errorlevel 1 (
    echo Installing Pillow...
    python -m pip install Pillow --quiet
)

python -c "import camelot" >nul 2>&1
if errorlevel 1 (
    echo Installing camelot-py...
    python -m pip install camelot-py --quiet
)

python -c "import huggingface_hub" >nul 2>&1
if errorlevel 1 (
    echo Installing huggingface_hub...
    python -m pip install "huggingface_hub[hf_xet]>=0.23" --quiet
)

python -c "import accelerate" >nul 2>&1
if errorlevel 1 (
    echo Installing accelerate...
    python -m pip install accelerate --quiet
)

echo [3/4] Dependencies checked
echo.

REM Check if app.py exists
if not exist "app.py" (
    echo ERROR: app.py not found in current directory
    echo Please make sure you're running this from the project root directory
    pause
    exit /b 1
)

echo [4/4] Starting Streamlit chatbot...
echo.
echo The chatbot will open in your default web browser.
echo Press Ctrl+C to stop the server.
echo.

REM Run Streamlit app
streamlit run app.py

pause

