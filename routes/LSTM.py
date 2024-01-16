import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import multiprocessing
import random

from routes.stock import get_stock, get_stock_date, get_user_preference, get_all_stock_codes


def create_dataset(data, look_back):
    x_data, y_data = [], []
    for i in range(look_back, len(data)):
        x_data.append(data[i - look_back:i, :])  # 모든 특징을 포함하도록 수정
        y_data.append(data[i, 0])
    print(np.array(x_data).shape)  # x_data의 형태 출력
    return np.array(x_data), np.array(y_data)


def predict_stock_price(stock_code, start_date, end_date):
    # 데이터 로드
    stock_data = get_stock_date(stock_code, start_date, end_date)

    # 데이터 전처리
    price_scaler = MinMaxScaler()
    price = price_scaler.fit_transform(np.array(stock_data['price']).reshape(-1, 1))

    open_price_scaler = MinMaxScaler()
    open_price = open_price_scaler.fit_transform(np.array(stock_data['open']).reshape(-1, 1))

    high_scaler = MinMaxScaler()
    high = high_scaler.fit_transform(np.array(stock_data['high']).reshape(-1, 1))

    low_scaler = MinMaxScaler()
    low = low_scaler.fit_transform(np.array(stock_data['low']).reshape(-1, 1))

    volume_scaler = MinMaxScaler()
    volume = volume_scaler.fit_transform(np.array(stock_data['volume']).reshape(-1, 1))

    # 스케일링된 요소들을 결합
    scaled_data = np.concatenate((price, open_price, high, low, volume), axis=1)

    # 훈련 데이터와 테스트 데이터로 분리
    train_data = scaled_data[:int(len(scaled_data) * 0.8)]
    test_data = scaled_data[int(len(scaled_data) * 0.8) - 60:]

    look_back = 60
    x_train, y_train = create_dataset(train_data, look_back)
    x_test, y_test = create_dataset(test_data, look_back)

    # LSTM에 입력으로 사용할 수 있는 형태로 변환
    num_features = scaled_data.shape[1]  # 사용하는 요소의 수
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], num_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], num_features))

    # LSTM 모델 생성
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], num_features)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # 모델 컴파일 및 훈련
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # 예측
    predictions = model.predict(x_test)
    predictions = price_scaler.inverse_transform(predictions)  # 'price'에 대한 스케일러를 사용하여 역변환

    # 예측 결과와 실제 값을 그래프로 표시
    plt.figure(figsize=(8, 4))
    range_future = len(predictions)
    # plt.plot(np.arange(range_future), np.array(y_test), label='True Future')
    plt.plot(np.arange(range_future), np.array(predictions), label='Predicted Future')
    plt.legend(loc='upper left')
    plt.xlabel('Time (day)')
    plt.ylabel('Stock Price ($)')
    plt.savefig('prediction.png')

    # 예측 결과를 출력
    return predictions


def recommend_stock(user_preference, stock_data):
    # 사용자의 투자 성향과 위험 허용도에 따라서 추천할 주식 종목을 필터링
    # 이 부분은 각 사용자의 투자 성향과 위험 허용도에 따라서 다른 알고리즘을 사용하여 구현할 수 있습니다.
    # 여기서는 단순히 예시로, 사용자의 투자 성향이 '공격적'이고 위험 허용도가 '높음'인 경우에만 추천하도록 합니다.
    if user_preference['investment_style'] == '공격적' and user_preference['risk_tolerance'] == '높음':
        recommended_stocks = stock_data
    else:
        recommended_stocks = []

    # 추천할 주식 종목들의 주가를 예측
    for stock in recommended_stocks:
        start_date = "20230101"
        end_date = "20240101"
        predictions = predict_stock_price(stock['code'], start_date, end_date)

        # 예측한 주가를 기반으로 주식 종목을 추천
        # 이 부분은 각 주식 종목의 예측된 주가에 따라서 다른 알고리즘을 사용하여 구현할 수 있습니다.
        # 여기서는 단순히 예시로, 예측된 주가가 상승할 것으로 예상되는 주식 종목만 추천하도록 합니다.
        if predictions[-1] > stock['price']:
            print(f"주식 종목 {stock['code']}을(를) 추천합니다.")


def worker(stock_code):
    start_date = "20230101"
    end_date = "20240101"
    predictions = predict_stock_price(stock_code, start_date, end_date)
    print(predictions)


def get_category_stock_codes(category, max_stock_count=50):
    # 해당 카테고리의 주식 종목 코드를 가져옵니다.
    category_stock_codes = [code for code, cat in stock_category_data.items() if cat == category]

    # 주식 종목 코드의 수가 max_stock_count보다 많다면 무작위로 max_stock_count개를 선택합니다.
    if len(category_stock_codes) > max_stock_count:
        category_stock_codes = random.sample(category_stock_codes, max_stock_count)

    return category_stock_codes

if __name__ == "__main__":
    user_preference = get_user_preference()
    all_stock_codes = get_all_stock_codes()

    # 사용자의 선호 카테고리를 가져옵니다. (이 부분은 사용자의 선호 카테고리를 어떻게 가져오는지에 따라 달라질 것입니다.)
    preferred_category = get_user_preferred_category()

    # 선호 카테고리의 주식 코드를 가져옵니다.
    category_stock_codes = get_category_stock_codes(preferred_category)

    # 병렬 처리를 위한 프로세스 풀 생성
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # 각각의 주식 코드에 대해 주가 예측을 병렬로 수행
    pool.map(worker, category_stock_codes)

    recommend_stock(user_preference, all_stock_codes)

