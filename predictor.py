import numpy as np
import pandas as pd
from data_loader import Loader

full_matrix = np.load('final_prediction_matrix.npy')

movies_df = pd.read_csv('movies.csv')

train_set = Loader(pd.read_csv('ratings.csv'))

user_index = train_set.userid2idx[2]  # enter user id here
user_ratings = full_matrix[user_index]

top_movie_indices = user_ratings.argsort()[-3:][::-1]  # enter how mant top n movies do we need


top_movie_ids = [train_set.idx2movieid[idx] for idx in top_movie_indices]
top_movie_titles = movies_df[movies_df['movieId'].isin(top_movie_ids)][['movieId', 'title']]

print("Top 3 recommended movies for the user at index 2:")
print(top_movie_titles)
