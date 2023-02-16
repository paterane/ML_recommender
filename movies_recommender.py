import pandas as pd
import numpy as np
import tensorflow as tf
import joblib as jb

class movies_recommender:
    def __init__(self):
        self.__dict    = jb.load("movie_recommendation_data/movie_dict.pkl")
        self.__model   = tf.keras.models.load_model("movie_recommendation_data/recommender_model/")
        self.__uScaler = jb.load("movie_recommendation_data/userScaler.save")
        self.__iScaler = jb.load("movie_recommendation_data/itemScaler.save")
        self.__yScaler = jb.load("movie_recommendation_data/labelScaler.save")
        self.__item_df = pd.read_csv("movie_recommendation_data/item_vecs.csv")
        self.__item_vec   = self.__item_df.to_numpy()
        self.__item_vec_S = self.__iScaler.transform(self.__item_vec[:,1:])
        self.__user_vec_S = 0
        self.__userData = {
            'action' : 0.0,
            'adventure' : 0.0,
            'animation' : 0.0,
            'childrens' : 0.0,
            'comedy' : 0.0,
            'crime' : 0.0,
            'documentary' : 0.0,
            'drama' : 0.0,
            'fantasy' : 0.0,
            'horror' : 0.0,
            'mystery' : 0.0,
            'romance' : 0.0,
            'scifi' : 0.0,
            'thriller' : 0.0
        }
        
    def set_user_data(self, genres, rating):
        self.__userData[genres] = rating
        user_vec = np.array([[self.__userData['action'], self.__userData['adventure'],
                              self.__userData['animation'], self.__userData['childrens'],
                              self.__userData['comedy'], self.__userData['crime'], self.__userData['documentary'],
                              self.__userData['drama'], self.__userData['fantasy'], self.__userData['horror'],
                              self.__userData['mystery'], self.__userData['romance'], self.__userData['scifi'],
                              self.__userData['thriller']]])
        self.__user_vec_S = np.tile(self.__uScaler.transform(user_vec), (self.__item_vec_S.shape[0],1))
        
    def get_user_data(self):
        return self.__userData
    
    def get_recommended_movies(self, movie_count=10):
        rating_predicted = self.__yScaler.inverse_transform(self.__model.predict([self.__user_vec_S, self.__item_vec_S]))
        sorted_indices = np.argsort(-rating_predicted.reshape(-1,)).tolist()
        sorted_rating_predicted = np.around(rating_predicted[sorted_indices].astype(float), decimals=1)
        sorted_item_vec = self.__item_vec[sorted_indices]
        movie_id_list = sorted_item_vec[:movie_count,0].astype(int).tolist()
        ratingAvg_list = np.around(sorted_item_vec[:movie_count,2], decimals=1).tolist()
        title_list = []
        genres_list = []
        for idx in range(movie_count):
            title_list.append(self.__dict[movie_id_list[idx]]['title'])
            genres_list.append(self.__dict[movie_id_list[idx]]['genres'])
        movie_df = pd.DataFrame({
            "movie_id"       :movie_id_list,
            "ratingAvg"      :ratingAvg_list,
            #"ratingPredicted":sorted_rating_predicted[:movie_count].reshape(-1,).tolist(),
            "title"          :title_list,
            "genres"         :genres_list
        })
        return movie_df