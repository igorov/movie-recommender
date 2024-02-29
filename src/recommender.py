import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



class Recommender():
    def __init__(self) -> None:
        #self.data = pd.read_csv("src/data/processed_movies.csv")
        self.etiquetas = pd.read_csv("src/data/items.csv")
        self.etiquetas = self.etiquetas[['id','title','genres','original_language','release_date','spoken_languages','runtime','overview']]
        logging.info("Init")

    def ReadData(self, path):
        test_df=pd.read_csv(path, sep=",")
        test_df=test_df.rename(columns={'userId':'user_id',
                                        'itemId':'item_id'})\
        .sort_values(by=['user_id', 'item_id'], ascending=True)
        test_df.reset_index(inplace=True)
        logging.info('Cantidad registros :', test_df.shape)
        test_df=test_df[["user_id", "item_id", "rating"]]
        test_df['rating']=test_df['rating'].astype('int')

        valores_unicos = list(set(test_df.user_id.unique()))
        valores_unicos.sort()
        id_unico = list(range(1, len(valores_unicos) + 1))
        mapeo_df = pd.DataFrame({'user_id': valores_unicos, 'ID': id_unico})
        df = pd.DataFrame({'A': list(test_df.user_id.unique())})
        test_df = test_df.merge(mapeo_df, on='user_id')

        test_df=test_df[['ID',
                        'item_id',
                        'rating',
                        'user_id']]\
                        .rename(columns={'user_id':'user_idR',
                                            'ID':'user_id'})
        return test_df

    def calculate_score(self, u, i, ratings_matrix, similarity_matrix, normalized_ratings_matrix):
        # Check whether the item is in the training dataset
        if i not in ratings_matrix.columns:
            return 2

        similarity_scores = similarity_matrix[u].drop(labels=u)
        normalized_ratings = normalized_ratings_matrix[i].drop(index=u)
        # Drop users that haven't rated the item
        similarity_scores.drop(index=normalized_ratings[normalized_ratings.isnull()].index, inplace=True)
        normalized_ratings.dropna(inplace=True)

        # If none of the other users have rated items in common with the user in question return the baseline value
        if similarity_scores.isna().all():
            return 2

        total_score = 0
        total_weight = 0
        for v in normalized_ratings.index:
            # It's possible that another user rated the item but that
            # they have not rated any items in common with the user in question
            if not pd.isna(similarity_scores[v]):
                total_score += normalized_ratings[v] * similarity_scores[v]
                total_weight += abs(similarity_scores[v])

        avg_user_rating = ratings_matrix.T.mean()[u]

        return avg_user_rating + total_score / total_weight

    def fit(self):
        path="src/data/ratings.db.csv"
        train_df=self.ReadData(path)
        
        path="src/data/ratings.test.known.csv"
        test_df=self.ReadData(path)

        ratings_matrix = pd.pivot_table(train_df, values='rating', index='user_id', columns='item_id')
        normalized_ratings_matrix = ratings_matrix.subtract(ratings_matrix.mean(axis=1), axis=0)
        similarity_matrix = ratings_matrix.T.corr()

        test_ratings = np.array(test_df["rating"])
        user_item_pairs = zip(test_df["user_id"], test_df["item_id"])
        pred_ratings = np.array([self.calculate_score(user_id, item_id, ratings_matrix, similarity_matrix, normalized_ratings_matrix) for (user_id, item_id) in user_item_pairs])
        error=np.sqrt(mean_squared_error(test_ratings, pred_ratings))

        self.data = pd.DataFrame({'user_id': test_df.user_id.values,
                        'item_id': test_df.item_id.values,
                        'real': test_df.rating.values,
                        'predict': pred_ratings,
                        'user_idR': test_df.user_idR})
        
        logging.info("Recommendation model trained successfully!")
    
    def topn(self, user_idR, n):
        logging.info(f"Getting the top n movies: user_idR={user_idR}")
        aux = self.data.copy()

        top = aux.loc[aux.user_idR == user_idR, 'item_id'].head(n).values

        # Merge with etiquetas DataFrame to get the required fields
        user_recommendations = pd.merge(aux, self.etiquetas[['id', 'title', 'genres', 'original_language', 'release_date', 'spoken_languages', 'runtime', 'overview']],
                                        left_on='item_id', right_on='id', how='inner')

        logging.info(f'user_recommendations: {len(user_recommendations)}')

        # Get the unique movies of the top recommended movies
        top_movies = user_recommendations.loc[user_recommendations.item_id.isin(top)].drop_duplicates(subset=['item_id'])

        logging.info(f'Number of movies: {len(top_movies)}')
        
        # convertir el dataframe a lista
        top_moviesR = top_movies.to_dict(orient='records')
        return top_moviesR