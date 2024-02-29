from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
import logging
from recommender import Recommender


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
api = Api(app)

# Crear una instancia de la clase Recommender
my_recommender = Recommender()
my_recommender.fit()

@app.route("/recommendation/movies")
def get_movies():
    logging.info("Getting movies")
    userId = request.args.get("userId")
    logging.info(userId)
    return my_recommender.topn(int(userId), 10)

if __name__ == '__main__':
    logging.info("Starting the server")

    app.run(host='0.0.0.0', debug=True, port=8080)