import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

movies_path = "ml-latest-small/movies.csv"
ratings_path = "ml-latest-small/ratings.csv"
tags_path = "ml-latest-small/tags.csv"
links_path = "ml-latest-small/links.csv"

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

movies['genres'] = movies['genres'].str.split('|')
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

print(movies.head()['genres'])
print(ratings.head()['timestamp'])