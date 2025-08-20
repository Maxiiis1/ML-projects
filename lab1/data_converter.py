import re


class DataConverter:
    def convert_to_minutes(self, time):
        if "h" in time:
            if "m" in time:
                hours, minutes = map(int, re.findall(r'\d+', time))
                return hours * 60 + minutes
            else:
                hours = int(re.findall(r'\d+', time)[0])
                return hours * 60
        elif "m" in time:
            minutes = int(re.findall(r'\d+', time)[0])
            return minutes
        else:
            return 0

    def convert_rating_count(self, rating_count_string):
        try:
            rating_count = int(rating_count_string.strip().strip('()').replace(',', ''))
        except ValueError:
            if '.' in rating_count_string.strip().strip('()'):
                number = rating_count_string.strip().strip('()').replace('.', '')
                unit = number[-1]
                number = number[:-1]
                multiplier = {'M': '00000', 'K': '000', 'B': '000000000'}
                if unit in multiplier:
                    multiplier = multiplier[unit]
                    rating_count = number + multiplier
                else:
                    rating_count = None
            else:
                rating_count_text = rating_count_string.strip().strip('()').replace('M', '000000').replace('K',
                                                                                                           '000').replace(
                    'B', '000000000').replace('.', '')
                rating_count = int(rating_count_text)
        return rating_count

    def convert_age_format(self, age):
        if age == "Not Rated":
            return "0+"
        elif age == "PG":
            return "6+"
        elif age == "PG-13":
            return "12+"
        elif age == "R":
            return "16+"
        elif age == "NC-17":
            return "18+"
        else:
            return age
