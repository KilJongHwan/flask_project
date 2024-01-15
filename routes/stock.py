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

def get_all_stock_codes():
    # 모든 주식 종목 코드를 가져옵니다.
    all_stock_codes = stock.get_market_ticker_list()
    return all_stock_codes

def get_stock_codes_by_market(market):
    # 지정한 시장의 주식 종목 코드를 가져옵니다.
    stock_codes = stock.get_market_ticker_list(market=market)
    return stock_codes
def get_user_preference():
    # 사용자의 투자 성향과 위험 허용도를 묻는 질문
    investment_style = input("당신의 투자 성향을 입력해주세요 (예: 보수적, 공격적): ")
    risk_tolerance = input("당신의 위험 허용도를 입력해주세요 (예: 높음, 중간, 낮음): ")
    stock_category = input("관심있는 주식의 산업 분야를 입력해주세요: ")

    # 사용자의 선호도를 dictionary 형태로 변환
    user_preference = {
        "investment_style": investment_style,
        "risk_tolerance": risk_tolerance,
        "stock_category": stock_category
    }

    return user_preference
