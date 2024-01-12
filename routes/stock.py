from pykrx import stock
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
