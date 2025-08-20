import config
from imd_parser import IMDbParser

if __name__ == "__main__":
    parser = IMDbParser(config.data_config)
    parser.save_to_csv()