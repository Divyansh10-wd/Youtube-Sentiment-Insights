import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_evaluation_errors.log', mode='a')
    ]
)
logger = logging.getLogger('model_evaluation')

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error('Error loading model: %s', e)
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    try:
        with open(vectorizer_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error('Error loading vectorizer: %s', e)
        raise

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error('Error loading params: %s', e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return report, cm
    except Exception as e:
        logger.error('Error evaluating model: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()

def save_model_info(run_id: str, model_path: str, file_path: str):
    try:
        with open(file_path, 'w') as f:
            json.dump({'run_id': run_id, 'model_path': model_path}, f, indent=4)
    except Exception as e:
        logger.error('Error saving model info: %s', e)
        raise

def main():
    # LOCAL FILE STORAGE SETUP FOR MLFLOW
    mlflow.set_tracking_uri("file:///E:/YOUTUBE-SENTIMENT-INSIGHTS/mlruns") # Replace with local directory
    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run() as run:
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))
            for k, v in params.items():
                mlflow.log_param(k, v)

            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            mlflow.sklearn.log_model(model, "lgbm_model", signature=signature, input_example=input_example)
            artifact_uri = mlflow.get_artifact_uri()
            model_path = f"{artifact_uri}/lgbm_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            report, cm = evaluate_model(model, X_test_tfidf, y_test)
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            log_confusion_matrix(cm, "Test")

            mlflow.set_tags({
                "model_type": "LightGBM",
                "task": "Sentiment Analysis",
                "dataset": "YouTube Comments"
            })

        except Exception as e:
            logger.error("Run failed: %s", e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
