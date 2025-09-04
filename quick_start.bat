@echo off
echo 🎭 Sentiment Analysis Deep Learning Project
echo ==========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

REM Check if models exist
if exist "saved_models\imdb\best_models.json" (
    echo ✅ Models already trained!
    echo Starting web application...
    streamlit run app.py
) else (
    echo ⚠️ No trained models found
    echo Starting model training...
    echo This will take about 15-25 minutes.
    echo.

    echo Which dataset would you like to train on?
    echo 1) IMDB Movie Reviews (recommended)
    echo 2) Stanford Sentiment Treebank
    set /p choice="Enter choice (1 or 2): "

    if "%choice%"=="1" (
        set dataset=imdb
    ) else if "%choice%"=="2" (
        set dataset=stanford_sentiment
    ) else (
        echo Invalid choice. Using IMDB dataset.
        set dataset=imdb
    )

    echo Training models on %dataset% dataset...
    python train_models.py --dataset %dataset%

    if errorlevel 1 (
        echo ❌ Training failed. Check the error messages above.
        pause
        exit /b 1
    ) else (
        echo.
        echo 🎉 Training completed successfully!
        echo Starting web application...
        streamlit run app.py
    )
)

pause
