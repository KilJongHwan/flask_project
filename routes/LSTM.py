import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from pykrx import stock
import requests
from datetime import date


def get_stock(stock_code):
    # 오늘 날짜 가져오기
    today = date.today().strftime("%Y%m%d")

    # 주식 정보 가져오기
    df = stock.get_market_ohlcv_by_ticker(today)

    # 지정한 주식 코드에 대한 정보 가져오기
    latest_stock_info = df.loc[stock_code]

    # 주식 정보를 JSON 형태로 변환
    stock_info = {
        "date": today,
        "name": stock_code,
        "price": latest_stock_info['종가'],  # 종가
        "open": latest_stock_info['시가'],  # 시가
        "high": latest_stock_info['고가'],  # 고가
        "low": latest_stock_info['저가'],  # 저가
        "volume": latest_stock_info['거래량']  # 거래량
    }

    return stock_info


def get_stock_date(stock_code, start_date, end_date):
    df = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code)

    # 주식 정보를 JSON 형태로 변환
    stock_info = {
        "date": df.index.strftime('%Y%m%d').tolist(),
        "name": stock_code,
        "price": df['종가'].values.tolist(),  # 종가
        "open": df['시가'].values.tolist(),  # 시가
        "high": df['고가'].values.tolist(),  # 고가
        "low": df['저가'].values.tolist(),  # 저가
        "volume": df['거래량'].values.tolist()  # 거래량
    }

    return stock_info


from datetime import datetime, timedelta
from multiprocessing import Pool


def calculate_volatility(stock_code, date, min_price, min_volume):
    # 최근 30일간의 데이터만 가져오기
    start_date = (datetime.strptime(date, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
    df = stock.get_market_ohlcv_by_date(start_date, date, stock_code)

    # 변동성 계산 시 최근 5일간의 데이터만 사용
    returns = df['종가'].iloc[-5:].pct_change()
    volatility = returns.std()

    # 종가와 거래량 필터
    last_close_price = df['종가'].iloc[-1]
    last_volume = df['거래량'].iloc[-1]
    if last_close_price < min_price or last_volume < min_volume:
        return None

    return (stock_code, volatility)


def get_all_stock_codes_filtered(risk_tolerance, date, min_price, min_volume):
    # 모든 주식 코드 가져오기
    all_stock_codes = stock.get_market_ticker_list(date)

    # 병렬 처리를 위한 프로세스 풀 생성
    with Pool(processes=4) as pool:  # 4는 병렬 처리를 수행할 프로세스 수
        results = pool.starmap(calculate_volatility,
                               [(stock_code, date, min_price, min_volume) for stock_code in all_stock_codes])

    # 위험 수용도와 주식의 변동성을 기준으로 필터링
    filtered_stock_codes = []
    for result in results:
        if result is None:
            continue

        stock_code, volatility = result
        if risk_tolerance == 'high' and volatility > 0.02:
            filtered_stock_codes.append(stock_code)
        elif risk_tolerance == 'low' and volatility < 0.01:
            filtered_stock_codes.append(stock_code)

    return filtered_stock_codes


def get_user_preference():
    # 사용자의 투자 성향과 위험 허용도를 묻는 질문
    investment_style = input("당신의 투자 성향을 입력해주세요 (예: 보수적, 공격적): ")
    risk_tolerance = input("당신의 위험 허용도를 입력해주세요 (예: 높음, 중간, 낮음): ")

    # 사용자의 선호도를 dictionary 형태로 변환
    user_preference = {
        "investment_style": investment_style,
        "risk_tolerance": risk_tolerance
    }

    return user_preference


def create_dataset(data, look_back):
    x_data, y_data = [], []
    for i in range(look_back, len(data)):
        x_data.append(data[i - look_back:i, :])  # 모든 특징을 포함하도록 수정
        y_data.append(data[i, 0])
    print(np.array(x_data).shape)  # x_data의 형태 출력
    return np.array(x_data), np.array(y_data),


from bs4 import BeautifulSoup


def get_news_articles(category='all'):
    # API 요청 URL 및 파라미터 설정
    url = 'https://www.bigkinds.or.kr/api/news/search.do'
    params = {
        'indexName': 'news',
        'startDate': '20230101',
        'endDate': '20240101',
        'isDuplicate': False,
        'apiKey': 'YOUR_API_KEY'  # 실제 API 키로 대체
    }

    # API 요청
    response = requests.post(url, data=json.dumps(params))

    # 응답 데이터를 JSON 형태로 변환
    data = response.json()

    # 뉴스 기사 제목 추출
    news_titles = [doc['TITLE'] for doc in data['resultList']]

    print(news_titles)


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
    # 출력 추가
    print("price shape:", price.shape)
    print("open_price shape:", open_price.shape)
    print("high shape:", high.shape)
    print("low shape:", low.shape)
    print("volume shape:", volume.shape)

    # 스케일링된 요소들을 결합
    scaled_data = np.concatenate((price, open_price, high, low, volume), axis=1)

    # 훈련 데이터와 테스트 데이터로 분리
    train_data = scaled_data[:int(len(scaled_data) * 0.8)]
    test_data = scaled_data[int(len(scaled_data) * 0.8) - 60:]

    look_back = 60
    if len(train_data) > look_back:  # train_data의 길이 검사
        x_train, y_train = create_dataset(train_data, look_back)
        x_test, y_test = create_dataset(test_data, look_back)
    else:
        print(f"train_data length: {len(train_data)}, look_back: {look_back}")
        return None  # 충분한 데이터가 없으므로 None을 반환

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
    model.fit(x_train, y_train, batch_size=32, epochs=5)

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
    plt.show()

    # 예측 결과를 출력
    return predictions


def recommend_stock(user_preference, all_stock_codes_filtered):
    max_increase_rate = -1
    recommended_stock = None
    start_date = "20230101"
    end_date = "20240101"

    for stock_code in all_stock_codes_filtered:
        # 주식 가격 예측
        predicted_price = predict_stock_price(stock_code, start_date, end_date)

        if predicted_price is None:  # predict_stock_price 함수가 None을 반환했는지 확인
            continue  # None을 반환했다면 이 주식은 건너뜀
        # 상승률 계산
        increase_rate = (predicted_price[-1] - predicted_price[0]) / predicted_price[0]

        # 상승률이 가장 높은 주식을 추천 종목으로 선정
        if increase_rate > max_increase_rate:
            max_increase_rate = increase_rate
            recommended_stock = stock_code

    recommended_stock_name = stock.get_market_ticker_name(recommended_stock)
    return recommended_stock_name, max_increase_rate * 100  # 백분율로 변환


if __name__ == "__main__":
    user_preference = get_user_preference()

    # 오늘 날짜를 가져옴
    date_today = date.today().strftime("%Y%m%d")

    # 모든 주식 코드를 가져옴
    all_stock_codes = get_all_stock_codes_filtered(user_preference['risk_tolerance'], date_today, min_price=20000,
                                                   min_volume=10000)
    all_stock_codes = all_stock_codes[:20]
    print(all_stock_codes)

    # 모든 주식 코드 중에서 가장 높은 수익률을 보여줄 것으로 예상되는 주식을 추천
    recommended_stock, max_increase_rate = recommend_stock(user_preference, all_stock_codes)

    average_increase_rate = np.mean(max_increase_rate)
    print(f"추천 주식: {recommended_stock}, 예상 수익률: {average_increase_rate:.2f}%")