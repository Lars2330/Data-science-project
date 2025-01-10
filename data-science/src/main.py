import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных
movies_path = "ml-latest-small/movies.csv"
ratings_path = "ml-latest-small/ratings.csv"
tags_path = "ml-latest-small/tags.csv"
links_path = "ml-latest-small/links.csv"

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)
tags = pd.read_csv(tags_path)
links = pd.read_csv(links_path)

movies['genres'] = movies['genres'].str.split('|')
movies['genres'] = movies['genres'].apply(lambda x: '|'.join(x) if isinstance(x, list) else x).values
#print(movies['genres'].head())  
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

data = ratings.merge(movies, on='movieId', how='left').merge(tags, on=['movieId', 'userId'], how='left').merge(links, on='movieId', how='left')
# Преобразование категориальных данных в числовые
data = pd.get_dummies(data, columns=['genres'], dtype=float)
print(data.columns)

# Удаление строк с пропущенными значениями
data = data.dropna()

# Проверка наличия столбцов перед удалением
columns_to_drop = ['rating', 'timestamp', 'userId', 'movieId', 'title']
existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]

X = data.drop(columns=existing_columns_to_drop)
y = data['rating']

# Сохранение исходных столбцов
original_columns = X.columns

# Убедитесь, что все столбцы в X имеют числовой тип данных
X = X.select_dtypes(include=[np.number])

# Стандартизация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(X, y, data, test_size=0.2, random_state=42)

# Преобразование данных в тензоры
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Определение модели
class RatingPredictor(nn.Module):
    def __init__(self, input_dim):
        super(RatingPredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Инициализация модели, функции потерь и оптимизатора
input_dim = X_train.shape[1]
model = RatingPredictor(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Оценка модели
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    mse = criterion(predictions, y_test)
    print(f'Mean Squared Error on test data: {mse.item():.4f}')

# Проверка наличия столбцов в data_test
print("Columns in data_test:", data_test.columns)

# Пример предсказания для существующего пользователя
# Выбираем случайного пользователя и фильм из тестовой выборки
user_id = data_test.iloc[0]['userId']
movie_id = data_test.iloc[0]['movieId']

# Получаем данные для этого пользователя и фильма
user_movie_data = data_test[(data_test['userId'] == user_id) & (data_test['movieId'] == movie_id)]
user_movie_data = user_movie_data.drop(columns=existing_columns_to_drop)
user_movie_data = user_movie_data.select_dtypes(include=[np.number])
user_movie_data = scaler.transform(user_movie_data)
user_movie_tensor = torch.tensor(user_movie_data, dtype=torch.float32)

with torch.no_grad():
    user_movie_prediction = model(user_movie_tensor)
    print(f'Predicted rating for user {user_id} on movie {movie_id}: {user_movie_prediction.item():.4f}')