Movie Rating Predictor and Genre Popularity Analysis

This project aims to predict movie ratings based on user and movie data using a neural network implemented in PyTorch, and to analyze the popularity of different movie genres. The dataset used for this project is the MovieLens dataset.

Project Structure

- rating_predictor.py: Main script for loading data, preprocessing, training the neural network, and making predictions.
- genre_popularity.py: Script for analyzing the popularity of different movie genres.
- ml-latest-small/: Directory containing the MovieLens dataset files.

Dataset

The dataset used in this project is the MovieLens dataset, which contains information about movies, ratings, tags, and links. The following files are used:

- movies.csv: Contains movie information such as movie ID, title, and genres.
- ratings.csv: Contains user ratings for movies.
- tags.csv: Contains user tags for movies.
- links.csv: Contains links to additional movie information.

Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- torch
- scikit-learn

You can install the required libraries using pip:

pip install pandas numpy torch scikit-learn

Usage

1. Clone the repository:

git clone https://github.com/Lars2330/data-science-project
cd data-science-project

2. Place the MovieLens dataset files (movies.csv, ratings.csv, tags.csv, links.csv) in the ml-latest-small/ directory.

3. Run the rating_predictor.py script:

python src/rating_predictor.py

4. Run the genre_popularity.py script:

python src/genre_popularity.py

Project Overview

rating_predictor.py

The rating_predictor.py script performs the following steps:

1. Load Data: Load the MovieLens dataset files into pandas DataFrames.
2. Preprocess Data: Split genres into a list, convert timestamps to datetime, merge datasets, and convert categorical data to numerical.
3. Train-Test Split: Split the data into training and testing sets.
4. Standardize Data: Standardize the numerical features.
5. Define Neural Network: Define a neural network model using PyTorch.
6. Train Model: Train the neural network on the training data.
7. Make Predictions: Make predictions for a sample user and movie from the test set.

genre_popularity.py

The genre_popularity.py script performs the following steps:

1. Load Data: Load the MovieLens dataset files into pandas DataFrames.
2. Extract Year: Extract the release year from movie titles.
3. Split Genres: Split genres into a list.
4. Get Unique Genres: Get a list of unique genres.
5. Convert Timestamp: Convert the rating timestamps to datetime.
6. Merge Datasets: Merge the movies, ratings, tags, and links datasets.
7. Drop Unnecessary Columns: Drop columns that are not needed for the analysis.
8. Calculate Genre Popularity: Calculate the average rating and rating count for each genre.
9. Determine Genre Popularity by Year: Analyze the popularity of genres by year.

Example Prediction

The rating_predictor.py script includes an example prediction for a random user and movie from the test set. The predicted rating and the actual rating are printed to the console.

Genre Popularity Analysis

The genre_popularity.py script prints the overall genre popularity and the most popular genre by year.

Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.