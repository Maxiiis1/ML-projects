import config
from dataset_preparation import DatasetPreparation

config = config.load_config()

data_preparation = DatasetPreparation(config)
data_preparation.download_dataset()

