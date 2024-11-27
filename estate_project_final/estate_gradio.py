import gradio as gr
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# 모델과 라벨 인코더 로드
model = joblib.load('final_model.pkl')
encoder_dong = joblib.load('label_encoder_dong.pkl')
encoder_apt = joblib.load('label_encoder_apt.pkl')


# 예측을 위한 함수 정의
def predict_apartment(city, dong, apt, year_of_completion, transaction_year_month, floor, area):
    # dong와 apt는 라벨 인코딩 필요
    if dong not in encoder_dong.classes_:
        return f"Unknown dong: {dong}"
    if apt not in encoder_apt.classes_:
        return f"Unknown apt: {apt}"

    encoded_dong = encoder_dong.transform([dong])[0]
    encoded_apt = encoder_apt.transform([apt])[0]

    if city == '부산':
        city = 0
    if city == '서울':
        city = 1

    floor += 4


    log_area = np.log1p(area)

    # 입력 데이터 구성
    input_data = np.array([
        city,
        encoded_dong,
        encoded_apt,
        year_of_completion,
        transaction_year_month,  # 이 부분도 처리 필요
        floor,
        log_area
    ]).reshape(1, -1)

    # 예측 수행
    prediction = model.predict(input_data)

    # 로그 변환 값을 복원
    price = np.expm1(prediction[0])  # 로그 값 복원
    price_in_won = price * 1e4  # 만원 단위 → 원 단위로 변환

    # 억 단위와 나머지 계산
    billion = int(price_in_won // 1e8)  # 억 단위
    ten_thousand = int((price_in_won % 1e8) // 1e4)  # 만 단위

    # 결과 포맷팅
    if billion > 0:
        result = f"{billion}억 {ten_thousand}만 원"
    else:
        result = f"{ten_thousand}만 원"

    return f"Predicted Price: {result}"

# Gradio 인터페이스 정의
# HTML을 렌더링하는 함수
def display_html():
    with open("analysis.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content


# 탭 1: 아파트 예측 시스템
apartment_prediction_tab = gr.Interface(
    fn=predict_apartment,
    inputs=[
        gr.Dropdown(label="시", choices=["부산", "서울"]),
        gr.Textbox(label="동"),
        gr.Textbox(label="아파트"),
        gr.Slider(minimum=1961, maximum=2024, label="완공년도"),
        gr.Slider(minimum=200001, maximum=202412, step=1, label="거래년월 (예: 201703)"),
        gr.Number(label="층"),
        gr.Number(label="면적(단위: 평)")
    ],
    outputs=["text"],
    title="아파트 거래가 예측",
    description="예측하고 싶은 아파트의 정보를 입력해주세요",
)

# 탭 2: HTML 기반 아파트 입지 분석
neighborhood_analysis_tab = gr.Interface(
    fn=display_html,
    inputs=None,
    outputs="html",
    title="아파트 입지 분석",
    description="아파트 주변 시설 등을 보여드립니다",
)

# 메인 페이지: 탭으로 시스템 연결
main_page = gr.TabbedInterface(
    interface_list=[apartment_prediction_tab, neighborhood_analysis_tab],
    tab_names=["아파트 거래가 예측", "아파트 입지 분석"]
)

# 실행
if __name__ == "__main__":
    main_page.launch()
