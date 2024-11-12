import torch
import pandas as pd
from torch.utils.data import Dataset

class Loader(Dataset):
    def __init__(self, ratings_df):
        self.ratings = ratings_df.copy()
        
        users = ratings_df.userId.unique()
        movies = ratings_df.movieId.unique()
        
        self.userid2idx = {o:i for i, o in enumerate(users)}
        self.movieid2idx = {o:i for i, o in enumerate(movies)}

        self.idx2userid = {i:o for o, i in self.userid2idx.items()}
        self.idx2movieid = {i:o for o, i in self.movieid2idx.items()}
        
        self.ratings.movieId = ratings_df.movieId.apply(lambda x: self.movieid2idx[x])
        self.ratings.userId = ratings_df.userId.apply(lambda x: self.userid2idx[x])
        
        self.x = self.ratings.drop(['rating', 'timestamp'], axis=1).values
        self.y = self.ratings['rating'].values
        self.x, self.y = torch.tensor(self.x), torch.tensor(self.y) 
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.ratings)
