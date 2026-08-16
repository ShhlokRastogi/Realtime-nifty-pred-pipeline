import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

versions = client.search_model_versions("name='crypto-model'")
print("Registered model versions:")
for v in versions:
    print(f"Version: {v.version}, Stage: {v.current_stage}, Status: {v.status}")
