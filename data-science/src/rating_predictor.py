import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
movies_path = "ml-latest-small/movies.csv"
ratings_path = "ml-latest-small/ratings.csv"
tags_path = "ml-latest-small/tags.csv"
links_path = "ml-latest-small/links.csv"

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)
tags = pd.read_csv(tags_path)
links = pd.read_csv(links_path)

# Split genres into a list and then join them back to a string
movies['genres'] = movies['genres'].str.split('|')
movies['genres'] = movies['genres'].apply(lambda x: '|'.join(x) if isinstance(x, list) else x).values

# Convert timestamp to datetime
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Merge datasets
data = ratings.merge(movies, on='movieId', how='left').merge(tags, on=['movieId', 'userId'], how='left').merge(links, on='movieId', how='left')

# Convert categorical data to numerical (fix columns to ensure consistency)
all_genres = data['genres'].str.get_dummies(sep='|').columns
data = pd.get_dummies(data, columns=['genres'], dtype=float)

# Remove rows with missing values
data = data.dropna()

# Define the target variable
y = data['rating']

# Ensure all columns in X are numeric
X = data.drop(columns=['rating', 'title', 'tag', 'timestamp_x', 'timestamp_y'])

# Save column names for future consistency
feature_columns = X.columns

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(X, y, data, test_size=0.2, random_state=5)

# Convert data to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define the model
class RatingPredictor(nn.Module):
    def __init__(self, input_dim):
        super(RatingPredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
model = RatingPredictor(input_dim)
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Select a random user and movie from the test set
random_row = data_test.sample(n=1, random_state=None)
user_id = random_row['userId'].values[0]
movie_id = random_row['movieId'].values[0]

# Prepare user and movie data for prediction
user_movie_data = data_test[(data_test['userId'] == user_id) & (data_test['movieId'] == movie_id)]

if user_movie_data.empty:
    print(f"No data found for user {user_id} and movie {movie_id}.")
else:
     # Remove unnecessary columns from DataFrame
    columns_to_drop = ['userId', 'movieId', 'rating', 'title', 'tag', 'timestamp_x', 'timestamp_y']
    user_movie_data = user_movie_data.drop(columns=[col for col in columns_to_drop if col in user_movie_data], errors='ignore')
    
    # Ensure data has the same format as the training set
    for col in feature_columns:
        if col not in user_movie_data:
            user_movie_data[col] = 0
    user_movie_data = user_movie_data[feature_columns]
    
    # Scale the data
    user_movie_data = scaler.transform(user_movie_data)
    
    # Convert to tensor
    user_movie_tensor = torch.tensor(user_movie_data, dtype=torch.float32)
    
    # Check the size of the input data
    if user_movie_tensor.shape[0] != 1:
        print(f"Error: Expected one row of input data, but got {user_movie_tensor.shape[0]} rows.")
    else:
        # Make a prediction
        with torch.no_grad():
            user_movie_prediction = model(user_movie_tensor)
            print(f'Predicted rating for user {user_id} on movie {movie_id}: {user_movie_prediction.item():.4f}')
            print(f'Actual rating: {data_test[(data_test["userId"] == user_id) & (data_test["movieId"] == movie_id)]["rating"].values[0]}')
