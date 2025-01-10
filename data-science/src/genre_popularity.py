import pandas as pd
import re

# File paths for the datasets
movies_path = "ml-latest-small/movies.csv"
ratings_path = "ml-latest-small/ratings.csv"
tags_path = "ml-latest-small/tags.csv"
links_path = "ml-latest-small/links.csv"

# Load datasets
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)
tags = pd.read_csv(tags_path)
links = pd.read_csv(links_path)

# Extract year from movie titles
movies['year'] = movies['title'].apply(lambda x: re.search(r'\((\d{4})\)', x).group(1) if re.search(r'\((\d{4})\)', x) else 0)

# Split genres into a list
movies['genres'] = movies['genres'].str.split('|')

# Get unique genres
genres = movies['genres'].explode().unique().tolist()

# Convert timestamp to datetime
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Merge datasets
data = ratings.merge(movies, on='movieId', how='left').merge(tags, on=['movieId', 'userId'], how='left').merge(links, on='movieId', how='left')

# Drop unnecessary columns
data = data.drop(columns=['timestamp_x', 'timestamp_y'])

# Calculate genre popularity
genre_popularity = {}
for genre in genres:
    genre_data = data[data['genres'].apply(lambda x: genre in x)]
    avg_rating = genre_data['rating'].mean()
    rating_count = genre_data['rating'].count()
    genre_popularity[genre] = {'average_rating': avg_rating, 'rating_count': rating_count}

# Convert genre popularity dictionary to DataFrame
genre_popularity_df = pd.DataFrame.from_dict(genre_popularity, orient='index').reset_index()
genre_popularity_df.columns = ['genre', 'average_rating', 'rating_count']

# Sort by popularity (first by rating count, then by average rating)
genre_popularity_df = genre_popularity_df.sort_values(by=['rating_count', 'average_rating'], ascending=False)
genre_popularity_df.reset_index(drop=True, inplace=True)
genre_popularity_df.index += 1

# Print overall genre popularity
print("Overall Genre Popularity:")
print(genre_popularity_df)

# Determine genre popularity by year
data_exploded = data.explode('genres')
year_genre_popularity = data_exploded.groupby(['year', 'genres']).agg(
    average_rating=('rating', 'mean'),
    rating_count=('rating', 'count')
).reset_index()

# Determine the most popular genre for each year
most_popular_genre_by_year = year_genre_popularity.sort_values(['year', 'rating_count', 'average_rating'], ascending=[True, False, False]).drop_duplicates('year')
most_popular_genre_by_year.reset_index(drop=True, inplace=True)
most_popular_genre_by_year.index += 1

# Print the most popular genre by year.
print("Most Popular Genre by Year:")
print(most_popular_genre_by_year)