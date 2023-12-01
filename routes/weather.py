import json
import requests
from bs4 import BeautifulSoup
from flask import request
import datetime


def get_weather():
    url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    output = ""

    for loc in soup.select("location"):
        output += f"<h3>{loc.select_one('city').string}<h3>"
        output += f"날씨 : {loc.select_one('wf').string}</br>"
        output += f"최저/최고 기온 : {loc.select_one('tmn').string}/{loc.select_one('tmx').string}</br>"

    return output

def get_weather2():
    print("GET 요청을 받았습니다.")
    nx_val = request.args.get('x', default=None, type=None)
    ny_val = request.args.get('y', default=None, type=None)
    print("X : ", nx_val)
    print("Y : ", ny_val)
    API_KEY = 'eYN5v7MqpYb0d+BloXXD4GVskkBqtc/uYQRUt/OZFvuagBUxKSRj3poGewmvwCRyac820tOK11wNBvE0C7DJ/Q=='
    API_KEY_decode = requests.utils.unquote(API_KEY)

    # 날짜 및 시간 설정
    now = datetime.datetime.now()

    # base_date에 날짜를 입력하기 위해 날짜를 출력 형식을 지정해 변수에 할당
    date = now.strftime('%Y%m%d')

    # base_time에 시간을 입력하기 위해 시간을 출력 형식을 지정해 변수에 할당
    time = now.strftime('%H%M')

    # 현재 분이 30분 이전이면 30분 전 시간으로 설정
    if now.minute < 30:
        now = now - datetime.timedelta(minutes=30)
        time = now.strftime('%H%M')
    else:
        time = now.strftime('%H%M')

    # 요청 주소 및 요청 변수 지정
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'

    # 발표 일자 지정
    baseDate = date
    baseTime = time

    # 예보지점 X 좌표 (서울시 강남구 역삼동)
    # nx_val = 62
    # ny_val = 126

    # 한 페이지에 포함된 결과 수
    num_of_rows = 6
    # 페이지 번호
    page_no = 1
    # 응답 데이터 형식 지정
    data_type = 'JSON'

    req_parameter = {'serviceKey': API_KEY_decode,
                     'nx': nx_val, 'ny': ny_val,
                     'base_date': baseDate, 'base_time': baseTime,
                     'pageNo': page_no, 'numOfRows': num_of_rows,
                     'dataType': data_type}

    # 요청 및 응답
    r = requests.get(url, params=req_parameter)

    # JSON 형태로 응답받은 데이터를 딕셔너리로 변환
    dict_data = r.json()

    # 출력을 이쁘게 하기 위해 json.dumps()를 사용하여 들여쓰기(indent) 옵션을 지정
    print(json.dumps(dict_data, indent=2))

    # 딕셔너리 데이터를 분석하여 원하는 데이터를 추출
    weather_items = dict_data['response']['body']['items']['item']

    print(f"[ 발표 날짜 : {weather_items[0]['baseDate']} ]")
    print(f"[ 발표 시간 : {weather_items[0]['baseTime']} ]")

    weather_data = {}

    for k in range(len(weather_items)):
        weather_item = weather_items[k]
        obsrValue = weather_item['obsrValue']
        if weather_item['category'] == 'T1H':
            weather_data['tmp'] = f"{obsrValue}℃"
        elif weather_item['category'] == 'REH':
            weather_data['hum'] = f"{obsrValue}%"
        elif weather_item['category'] == 'RN1':
            weather_data['pre'] = f"{obsrValue}mm"
    # 딕셔너리를 JSON 형태로 변환
    json_weather = json.dumps(weather_data, ensure_ascii=False, indent=4)
    return json_weather