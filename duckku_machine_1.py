import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from pykrx import stock
from datetime import datetime, timedelta
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="👵덕구할배의 덕구머신 ver1", layout="wide")
st.subheader("👵 덕구머신은 일주일 뒤 주가가 많이 오를 주식을 찾아줍니다!")

today = datetime.today()
start = (today - timedelta(days = 365)).strftime("%Y%m%d")
acc_days = 365
st.write(today)

col1, col2 = st.columns(2)
with col1:
    
    num_thicker = st.slider("**📊  탐색 종목 범위 (Max 1500)**", min_value=100, max_value=1500, value=200, step=10)
    exp_gain = st.slider("**📊  원하는 수익률 목표(%)**", min_value=5, max_value=100, value=10, step=5) / 100    
    targ_thicker = st.slider("**📊  찾고싶은 종목 개수 (Max 50)**", min_value=10, max_value=50, value=10, step=10)
    
    today = datetime.today()
    start = (today - timedelta(days = acc_days)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    ###### Model_1 학습 ######
    model_1 = LGBMClassifier(
        n_estimators=1000,      # 트리 개수 (RandomForest의 n_estimators와 비슷)
        learning_rate=0.05,     # 학습률
        max_depth=-1,           # 트리 깊이 (-1이면 자동으로 최적 깊이 탐색)
        num_leaves=150,         # 한 트리에서 리프 노드 개수
        subsample=0.8,          # 데이터 샘플링 비율 (과적합 방지)
        colsample_bytree=0.8,   # 피처 샘플링 비율
        random_state=42
    )

    ###### Model_2 학습 ######
    model_2 = LGBMClassifier(
        n_estimators=1500,      # 트리 개수 (RandomForest의 n_estimators와 비슷)
        learning_rate=0.05,     # 학습률
        max_depth=-1,           # 트리 깊이 (-1이면 자동으로 최적 깊이 탐색)
        num_leaves=300,         # 한 트리에서 리프 노드 개수
        subsample=0.8,          # 데이터 샘플링 비율 (과적합 방지)
        colsample_bytree=0.8,   # 피처 샘플링 비율
        random_state=42
    )

    ###### Model_3 학습 ######
    model_3 = LGBMClassifier(
        n_estimators=1800,      # 트리 개수 (RandomForest의 n_estimators와 비슷)
        learning_rate=0.01,     # 학습률
        max_depth=-1,           # 트리 깊이 (-1이면 자동으로 최적 깊이 탐색)
        num_leaves=500,         # 한 트리에서 리프 노드 개수
        subsample=0.8,          # 데이터 샘플링 비율 (과적합 방지)
        colsample_bytree=0.8,   # 피처 샘플링 비율
        random_state=42
    )
    
    if st.button("할배, 알려주세요!"):
        with st.spinner("덕구머신이 종목을 추천하고 있습니다..."):
            ##### CSV File 지정 주소
            url = "https://github.com/Duckkoo-halbea/Duckkoo_machine/blob/main/market_data.csv" + "?raw=true"
            data = pd.read_csv(url)
            
            ###### 데이터 Feature 생성 (X인자) ######

            data["return_5d"] = data["종가"].pct_change(5) # 차이 100분율을 의미함
            data["return_20d"] = data["종가"].pct_change(20) # 차이 100분율을 의미함
            data["return_60d"] = data["종가"].pct_change(60) # 차이 100분율을 의미함

            data["ma5"] = data["종가"].rolling(5).mean()
            data["ma20"] = data["종가"].rolling(20).mean()
            data["ma60"] = data["종가"].rolling(60).mean()

            data["vol_ma5"] = data["거래량"].rolling(5).mean()
            data["vol_ma20"] = data["거래량"].rolling(20).mean()
            data["vol_ma60"] = data["거래량"].rolling(60).mean()

            ###### 데이터 Feature 생성 (Y인자) ######
            data["target"] = (data["return_5d"].shift(-5) > exp_gain).astype(int)
            data = data.dropna()

            ###### X인자, Y인자, Data split ######
            features = ["return_5d", "return_20d", "return_20d", "ma5", "ma20", "ma60", "vol_ma5", "vol_ma20", "vol_ma60"]

            X = data[features]
            y = data["target"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size = 0.2, shuffle = False
            )
            e_time = time.time()
            
            model_1.fit(X_train, y_train)
            model_2.fit(X_train, y_train)
            model_3.fit(X_train, y_train)

            print(f"\n#### 테스트 정확도 R_FOR_ 1: {model_1.score(X_test, y_test)*100:.4f}")
            print(f"#### 테스트 정확도 L_GBM_ 2: {model_2.score(X_test, y_test)*100:.4f}")
            print(f"#### 테스트 정확도 L_GBM_ 3: {model_3.score(X_test, y_test)*100:.4f}")

            latest_data = data.groupby("ticker").tail(1)
            X_latest = scaler.transform(latest_data[features])

            latest_data["pred"] = np.round(model_1.predict_proba(X_latest)[:, 1]*100,2)
            latest_data["pred_2"] = np.round(model_2.predict_proba(X_latest)[:, 1]*100,2)
            latest_data["pred_3"] = np.round(model_3.predict_proba(X_latest)[:, 1]*100,2)

            # 티커 → 종목명 변환
            latest_data["name"] = latest_data["ticker"].apply(stock.get_market_ticker_name)

            # 상승 확률 높은 n개 종목
            recommendations = latest_data.sort_values("pred", ascending=False).head(targ_thicker)
            recommendations_df = recommendations[["ticker", "name", "종가", "pred", "pred_2", "pred_3"]]

            recommendations_2 = latest_data.sort_values("pred_2", ascending=False).head(targ_thicker)
            recommendations_2_df = recommendations_2[["ticker", "name", "종가", "pred", "pred_2", "pred_3"]]

            recommendations_3 = latest_data.sort_values("pred_3", ascending=False).head(targ_thicker)
            recommendations_3_df = recommendations_2[["ticker", "name", "종가", "pred", "pred_2", "pred_3"]]

            recommendations_total = pd.concat([recommendations_3_df, recommendations_2_df, recommendations_df], axis = 0)
            recommendations_total["pred_aver"] = (recommendations_total["pred"] + recommendations_total["pred_2"] + recommendations_total["pred_3"]) / 3
            recommendations_total["Today"] = end
            recommendations_total_final = recommendations_total[['name', '종가', 'pred_aver']]
            recommendations_total_final = recommendations_total_final.sort_values(by='pred_aver', ascending = False)
            
            st.dataframe(recommendations_total_final)

            







