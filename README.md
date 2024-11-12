# Movie Recommender System using Matrix Factorization

This project implements a simple movie recommender system using matrix factorization techniques with PyTorch. The system utilizes collaborative filtering to predict user ratings for movies that they haven't rated yet.

## Files Structure

- `data_loader.py`: Contains the `RatingsDataset` class which loads and prepares the dataset for training, mapping user and movie IDs to indices.
- `matrix_factorization.py`: Contains the `MatrixFactorization` model and the `RecommenderSystem` class, which handles model initialization, training, and prediction of the full recommendation matrix.
- `main.py`: The entry point of the program that initializes the recommender system, trains it, and outputs the final recommendation matrix.
- `requirements.txt`: Contains the dependencies required to run the project.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name

