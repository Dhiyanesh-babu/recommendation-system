# Movie Recommender System using Matrix Factorization

This project implements a simple movie recommender system using matrix factorization techniques with PyTorch. The system utilizes collaborative filtering to predict user ratings for movies that they haven't rated yet.

## Files Structure

- `data_loader.py`: Contains the `RatingsDataset` class which loads and prepares the dataset for training, mapping user and movie IDs to indices.
- `matrix_factorization.py`: Contains the `MatrixFactorization` model and the `RecommenderSystem` class, which handles model initialization, training, and prediction of the full recommendation matrix.
- `main.py`: The entry point of the program that initializes the recommender system, trains it, and outputs the final recommendation matrix.
- `requirements.txt`: Contains the dependencies required to run the project.

## How the Model Works

### Matrix Factorization

Matrix factorization is a technique for collaborative filtering. It decomposes the user-item rating matrix into two lower-dimensional matrices: one representing user embeddings and the other representing item (movie) embeddings. The model then predicts missing values in the user-item matrix by computing the dot product of the corresponding user and item embeddings.

### Training

The recommender system is trained using the MatrixFactorization model, where:

- The user and movie embeddings are learned during training.
- The objective is to minimize the mean squared error (MSE) between the predicted ratings and the actual ratings in the dataset.

### Final Recommendation Matrix

After training, the full recommendation matrix is generated by multiplying the learned user and item embeddings. This matrix contains the predicted ratings for all users and movies.


## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Dhiyanesh-babu/recommendation-system.git
   cd recommendation-system
   ```
2. Install the required dependencies:
   ```bash
      pip install -r requirements.txt
      ```
3. Train the model:
   ```bash
      Run main.py file
   ```
4.How to Use the Final Matrix
   ```bash
      final_matrix = recommender.get_full_matrix() # This is in main.py file
      user_id = 2  # Example user ID, use => userid2idx[2] to get the index first
      top_n = 10  # Get the top 10 recommended movies for this user
      
      # Get predicted ratings for the user
      user_ratings = final_matrix[user_id]
      
      # Sort the movies by predicted ratings in descending order
      recommended_movie_indices = np.argsort(user_ratings)[::-1]
      
      # Exclude movies the user has already rated
      already_rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].values
      recommended_movie_indices = [movie for movie in recommended_movie_indices if movie not in already_rated_movies]
      
      # Get the top N recommended movies
      top_recommended_movies = recommended_movie_indices[:top_n]
      
      print(f"Top {top_n} recommended movies for user {user_id}: {top_recommended_movies}")
   ```


