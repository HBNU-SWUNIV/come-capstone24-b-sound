from flask import Flask, request, jsonify
import joblib
import numpy as np

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 저장된 모델과 라벨 인코더 로드
model = joblib.load('final_model.pkl')
label_encoder_dong = joblib.load('label_encoder_dong.pkl')
label_encoder_apt = joblib.load('label_encoder_apt.pkl')


@app.route('/')
def home():
    return "머신러닝 모델 예측 API입니다."


@app.route('/predict', methods=['POST'])
def predict():
    # JSON 데이터를 받아서 배열로 변환
    data = request.get_json(force=True)

    # 'dong' 값 처리 (훈련된 인코더 사용)
    try:
        dong_encoded = label_encoder_dong.transform([data['dong']])
    except ValueError as e:
        # 라벨 인코더에 없는 동의 경우 처리
        return jsonify({"error": f"동 값 '{data['dong']}'가 학습되지 않았습니다."}), 400

    # 'apt' 값 처리 (훈련된 인코더 사용)
    try:
        apt_encoded = label_encoder_apt.transform([data['apt']])
    except ValueError as e:
        # 라벨 인코더에 없는 아파트 이름의 경우 처리
        return jsonify({"error": f"아파트 이름 '{data['apt']}'가 학습되지 않았습니다."}), 400

    # 면적 로그 변환 (exclusive_use_area에 대한 로그 변환)
    log_area = np.log1p(data['exclusive_use_area'])

    # 예측에 필요한 피처 배열 구성
    features = np.array([data['transaction_year_month'],
                         log_area,
                         data['floor'],
                         dong_encoded[0],  # 라벨 인코딩된 dong
                         apt_encoded[0],  # 라벨 인코딩된 apt
                         data['year_of_completion']]).reshape(1, -1)

    # 모델로 예측
    prediction = model.predict(features)

    # 예측 결과 반환 (로그 복원)
    prediction_actual = np.expm1(prediction[0])
    return jsonify({'prediction': float(prediction_actual)})


if __name__ == '__main__':
    app.run(debug=True)
