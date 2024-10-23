import gradio as gr
import joblib
import numpy as np

# 모델과 라벨 인코더 로드
model = joblib.load('final_model.pkl')
encoder_dong = joblib.load('label_encoder_dong.pkl')
encoder_apt = joblib.load('label_encoder_apt.pkl')


# 예측을 위한 함수 정의
def predict(city, dong, apt, year_of_completion, transaction_year_month, floor, area):
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

    return f"Predicted Price: {prediction[0]}"

# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(label="City", choices=["부산", "서울"]),  # 선택 가능하도록 수정
        gr.Textbox(label="Dong"),  # 텍스트 입력
        gr.Textbox(label="Apt"),  # 텍스트 입력
        gr.Slider(minimum=1961, maximum=2024, label="Year of Completion"),  # 슬라이더로 연도 선택
        gr.Slider(minimum=200001, maximum=202412, step=1, label="Transaction Year-Month"),  # 슬라이더로 거래 연월 선택
        gr.Number(label="Floor"),  # 숫자 입력
        gr.Number(label="Area")  # 숫자 입력
    ],
    outputs="text",
    title="Apartment Price Prediction",
    description="Enter the details below to get a predicted apartment price.",
)

# Gradio 인터페이스 실행
if __name__ == "__main__":
    iface.launch()
