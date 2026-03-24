@echo off
echo ====================================
echo  Risk-Minimized Portfolio Optimizer
echo ====================================
echo.

pip install -r requirements.txt
echo.
echo Starting app...
streamlit run app.py --server.port 8503
pause
