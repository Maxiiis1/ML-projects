from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from data_converter import DataConverter
from movie import Movie


class IMDbParser:
    def __init__(self, config):
        self.config = config
        self.converter = DataConverter()
        self.driver = webdriver.Chrome()
        self.headers = {'User-Agent': self.config['user_agent']}

    def get_movies_data(self):
        movies_data = []
        url = self.config['imdb_url']
        self.driver.get(url)
        wait = WebDriverWait(self.driver, 100)

        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            wait.until(EC.presence_of_all_elements_located((By.XPATH, self.config['movie_items_xpath'])))
            try:
                wait.until(EC.element_to_be_clickable((By.XPATH, self.config['show_more_button_xpath'])))
                self.driver.find_element(By.XPATH, self.config['show_more_button_xpath']).click()
                wait.until(EC.presence_of_all_elements_located((By.XPATH, self.config['movie_items_xpath'])))
            except:
                break

        movie_items = self.driver.find_elements(By.XPATH, self.config['movie_items_xpath'])
        print(movie_items)
        for movie_item in movie_items:
            movie_data = self.get_movie_data(movie_item)
            movies_data.append(movie_data)
        return movies_data

    def get_movie_data(self, movie_item):
        movie = Movie.from_html(movie_item, self.config)

        movie_data = {
            "Средний балл зрителей": movie.rating,
            "Количество оценок": self.converter.convert_rating_count(movie.rating_count),
            "IMDb": movie.imdb_rating,
            "Режиссёр": movie.director,
            "Хронометраж, мин": self.converter.convert_to_minutes(movie.timing),
            "Возраст": self.converter.convert_age_format(movie.age),
            "Год": movie.year
        }
        print(movie_data)
        return movie_data

    def save_to_csv(self, filename="imdb_data.csv"):
        movies_data = self.get_movies_data()
        df = pd.DataFrame.from_records(movies_data)
        df.to_csv(filename, index=False)
        print(f"Данные успешно сохранены в файл {filename}")
