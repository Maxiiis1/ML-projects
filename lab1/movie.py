from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
from dataclasses import dataclass


@dataclass
class Movie:
    rating: float
    rating_count: str
    imdb_rating: float
    director: str
    timing: str
    age: str
    year: int

    @classmethod
    def from_html(cls, movie_item, config):
        rating = float(movie_item.find_element(By.CSS_SELECTOR, config['rating_css_selector']).text.strip())
        rating_count = movie_item.find_element(By.CSS_SELECTOR, config['rating_count_css_selector']).text
        try:
            imdb_rating = float(movie_item.find_element(By.CSS_SELECTOR, config['imdb_rating_css_selector']).text.strip()) / 10
        except NoSuchElementException:
            imdb_rating = 0

        director = movie_item.find_element(By.CSS_SELECTOR, config['director_css_selector']).text.strip()
        timing = movie_item.find_element(By.CSS_SELECTOR, config['hrono_css_selector']).text.strip()
        year = movie_item.find_element(By.CSS_SELECTOR, config['year_css_selector']).text.strip()

        try:
            age = movie_item.find_element(By.CSS_SELECTOR, config['age_css_selector']).text.strip()
        except NoSuchElementException:
            age = "12+"

        return cls(rating, rating_count, imdb_rating, director, timing, age, year)