import numpy as np 
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import yaml 
import logging 

# logging configure 

logger = logging.getLogger("Data ingestion") 
logger.setLevel('DEBUG') 

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG") 

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

def load_yaml(params_path: str) -> float:
    try:
        with open(params_path, 'r') as f:
            config = yaml.safe_load(f)
            return config['data_ingestion']['test_size']
    except FileNotFoundError:
        logger.error(f"Error: YAML file '{params_path}' not found.")
        raise
    except KeyError:
        logger.error(f"Error: Key 'data_ingestion -> test_size' not found in YAML file.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        df.drop(columns=['tweet_id'], inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error reading data from URL: {e}")
        raise

def update_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        return final_df
    except Exception as e:
        logger.error(f"Error updating dataframe: {e}")
        raise

def train_test_split_function(final_df: pd.DataFrame, test_size_yaml: float) -> tuple:
    try:
        return train_test_split(final_df, test_size=test_size_yaml, random_state=42)
    except Exception as e:
        logger.error(f"Error during train-test split: {e}")
        raise

def path(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        data_path = os.path.join("data", "raw")
        os.makedirs(data_path, exist_ok=True)  # Ensures directory already existing doesn't cause error
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except OSError as e:
        logger.error(f"File system error while saving files: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while writing CSV files: {e}")
        raise

def main():
    try:
        test_size_yaml = load_yaml("params.yaml")
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = update_data(df)
        train_data, test_data = train_test_split_function(final_df, test_size_yaml)
        path(train_data, test_data)
        logger.error("Data successfully split and saved.")
    except Exception as e:
        logger.error(f"Pipeline failed due to error: {e}")

if __name__ == "__main__":
    main()
