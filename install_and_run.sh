#!/bin/bash
echo "===================================="
echo " 리스크 최소화 포트폴리오 최적화"
echo "===================================="
echo

pip3 install -r requirements.txt
echo
echo "앱 실행 중..."
streamlit run app.py --server.port 8503
