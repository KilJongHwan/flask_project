from flask import Flask
from flask_cors import CORS
from routes.data import get_data
from routes.weather import get_weather, get_weather2
from routes.query import get_query
from routes.item import get_path_item
from routes.movie import get_movie

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])

app.add_url_rule('/api/data', 'get_data', get_data, methods=['GET'])
app.add_url_rule('/api/weather', 'get_weather', get_weather, methods=['GET'])
app.add_url_rule('/api/weather2', 'get_weather2', get_weather2, methods=['GET'])
app.add_url_rule('/api/query', 'get_query', get_query, methods=['GET'])
app.add_url_rule('/api/item/<item_id>', 'get_path_item', get_path_item, methods=['GET'])
app.add_url_rule('/api/movie', 'get_movie', get_movie, methods=['GET'])

if __name__ == '__main__':
    app.run(debug=True)