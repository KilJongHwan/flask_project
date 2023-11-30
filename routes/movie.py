import requests
import json
from bs4 import BeautifulSoup

def get_movie():
    url = 'https://search.daum.net/search?nil_suggest=btn&w=tot&DA=SBC&q=%EC%98%81%ED%99%94%EC%88%9C%EC%9C%84'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
    }

    response = requests.get(url, headers=headers)  # headers were missing in your GET request
    soup = BeautifulSoup(response.text, 'html.parser')
    movieInfoList = soup.find('ol', attrs={'class': 'movie_list'}).find_all('li') if soup.find('ol', attrs={
        'class': 'movie_list'}) else []

    movie_data = []

    for movieInfo in movieInfoList:
        movieRank = movieInfo.find('span', attrs={'class': 'img_number'})
        movieImg = movieInfo.find('img')
        movieTitle = movieInfo.find('a', attrs={'class': 'tit_main'})
        movieScore = movieInfo.find('em', attrs={'class': 'rate'})
        movieScoreCnt = movieInfo.find('a', attrs={'class': 'link_count'})
        ticketSalesAndOpenDate = movieInfo.find_all('dd', attrs={'class': 'cont'})
        if len(ticketSalesAndOpenDate) > 1:
            movieTicketSales = ticketSalesAndOpenDate[0]
            movieOpenDate = ticketSalesAndOpenDate[1]
        else:  # If there is no open date
            movieTicketSales = ticketSalesAndOpenDate[0]
            movieOpenDate = None  # Set to None if there is no open date

        movie_data.append({
            'rank': movieRank.get_text() if movieRank else "X",
            'image': movieImg['src'] if movieImg else "X",
            'title': movieTitle.get_text().strip() if movieTitle else "X",
            'score': movieScore.get_text() if movieScore else "X",
            'eval_num': movieScoreCnt.get_text() if movieScoreCnt else "X",
            'reservation': movieTicketSales.get_text().strip() if movieTicketSales else "X",
            'open_date': movieOpenDate.get_text().strip() if movieOpenDate else "X"
        })

    # Convert the movie data to JSON
    json_data = json.dumps(movie_data, ensure_ascii=False, indent=4)
    print(json_data)
    return json_data