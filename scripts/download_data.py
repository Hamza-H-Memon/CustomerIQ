import shutil
from pathlib import Path
import kagglehub

path = kagglehub.dataset_download("gauravtopre/bank-customer-churn-dataset")
print("Dataset downloaded to:", path)

source_file = Path(path) / "Bank Customer Churn Prediction.csv"
destination = Path(__file__).parent.parent / "data" / "raw_churn_data.csv"

shutil.copy(source_file, destination)
print("Copied to:", destination)