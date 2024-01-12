import requests
import json
from flask import jsonify
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

def get_stock(stock_code):
    # 웹드라이버 설정 (Chrome, Firefox 등)
    s=Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=s)

    # 네이버 금융 접속
    driver.get(f'https://finance.naver.com/item/main.nhn?code={stock_code}')

    # 주식 정보 가져오기
    stock_info = {
        "name": driver.find_element(By.XPATH, '//*[@id="middle"]/dl/dd[2]/h2/a').text,
        "price": driver.find_element(By.XPATH, '//*[@id="chart_area"]/div[1]/div').text,
    }

    driver.quit()

    return jsonify(stock_info)
