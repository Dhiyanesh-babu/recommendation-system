import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import Loader
from matrix_factorization import MatrixFactorization
import numpy as np

movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

n_users = len(ratings_df.userId.unique())
n_items = len(movies_df.movieId.unique())

print('unique user - ', n_users) 
print('unique movies - ', n_items) 
print('matrix size - ', n_items * n_users)
print('total ratings available - ', len(ratings_df))
print('% of filled matrix - ', len(ratings_df) / (n_users * n_items) * 100, '%')

model = MatrixFactorization(n_users, n_items, n_factors=8)

cuda = torch.cuda.is_available()
print("Is running on GPU:", cuda)
if cuda:
    model = model.cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_set = Loader(ratings_df)
train_loader = DataLoader(train_set, 4096, shuffle=True)

num_epochs = 5
for it in tqdm(range(num_epochs)):
    losses = []
    for x, y in train_loader:
        if cuda:
            x, y = x.cuda(), y.cuda()
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
    print(f'iter #{it} loss: {sum(losses)/len(losses)}')

trained_movie_embeddings = model.item_factors.weight.data.cpu().numpy()
trained_user_embeddings = model.user_factors.weight.data.cpu().numpy()

print(f"Trained user embeddings shape: {trained_user_embeddings.shape}")
print(f"Trained movie embeddings shape: {trained_movie_embeddings.shape}")

full_matrix = np.dot(trained_user_embeddings, trained_movie_embeddings.T)
print(f"Full matrix shape: {full_matrix.shape}")

np.save('final_prediction_matrix.npy', full_matrix)


