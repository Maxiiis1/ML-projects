import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class ModelQualityEvaluator:
    def __init__(self, data):
        self.data = data
        self.scaled_data = None

    def silhouette_scorer(self, model, data):
        model.fit(data)
        labels = model.predict(data)
        return silhouette_score(data, labels)

    def preprocess_data(self):
        self.data['Возраст'] = pd.to_numeric(self.data['Возраст'].str.replace('+', ''), errors='coerce')
        self.data['Возраст'] = self.data['Возраст'].fillna(self.data['Возраст'].median())

        label_encoder = LabelEncoder()
        self.data['Режиссёр'] = label_encoder.fit_transform(self.data['Режиссёр'])

        features = self.data[['Средний балл зрителей', 'Количество оценок', 'IMDb', 'Хронометраж, мин', 'Возраст', 'Год']]

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())

        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(features)

    def evaluate_model(self, model, param_grid):
        train_data, temp_data = train_test_split(self.scaled_data, test_size=0.25, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        grid_search = GridSearchCV(model(), param_grid, scoring=self.silhouette_scorer)
        grid_search.fit(train_data)

        print("Лучшие параметры:", grid_search.best_params_)
        print("Лучшее качество на тренировочной выборке:", grid_search.best_score_)

        best_model = grid_search.best_estimator_
        best_model.fit(val_data)
        val_labels = best_model.predict(val_data)
        val_silhouette = silhouette_score(val_data, val_labels)
        print("Значение Silhouette Score на валидационной выборке:", val_silhouette)

        best_model.fit(test_data)
        test_labels = best_model.predict(test_data)
        test_silhouette = silhouette_score(test_data, test_labels)
        print("Значение Silhouette Score на тестовой выборке:", test_silhouette)

        plt.figure(figsize=(10, 6))
        plt.plot(grid_search.cv_results_['mean_test_score'], label='Mean Silhouette Score')
        plt.xlabel('Гиперпараметры')
        plt.ylabel('Среднее значение Silhouette Score')
        plt.title('Качество модели на каждой эпохе обучения')
        plt.xticks(range(len(grid_search.cv_results_['mean_test_score'])), grid_search.cv_results_['params'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compare_models(self, my_kmeans, n_clusters, max_iter):

        sklearn_kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter)
        sklearn_labels = sklearn_kmeans.fit_predict(self.scaled_data)
        sklearn_silhouette = silhouette_score(self.scaled_data, sklearn_labels)
        print("Значение Silhouette Score для KMeans из sklearn на датасете:", sklearn_silhouette)

        my_model = my_kmeans(n_clusters=n_clusters, max_iter=max_iter)
        my_model.fit(self.scaled_data)
        my_labels = my_model.predict(self.scaled_data)
        my_silhouette = silhouette_score(self.scaled_data, my_labels)
        print("Значение Silhouette Score для моей модели на датасете:", my_silhouette)

        if abs(my_silhouette - sklearn_silhouette) > 0.2:
            print("Качество моделей сильно отличаются")
        else:
            print("Качество моделей похоже")
