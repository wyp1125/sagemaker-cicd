import sys
import json
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
            print("Configuration loaded successfully.")
            print(config)
    except FileNotFoundError:
        print(f"Error: The file {config_file} does not exist.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file {config_file} is not a valid JSON.")
        sys.exit(1)

    import mlflow
    print("MLflow tracking URI:", os.getenv('TRACKING_URI'))
    mlflow.set_tracking_uri(os.getenv('TRACKING_URI'))
    mlflow.set_experiment(config["experiment_name"])
    
    import sagemaker
    sagemaker_session = sagemaker.Session()
    print(config["role"])
    
    from sagemaker.pytorch import PyTorch
    estimator = PyTorch(
        entry_point="mlp_pytorch.py",
        role=config["role"],
        framework_version="1.4.0",
        py_version="py3",
        instance_count=1,
        instance_type="ml.c4.xlarge",
        hyperparameters={"epochs": int(config["num_epochs"]), "backend": "gloo"}
    )
    with mlflow.start_run(run_name=config["training_job_name"]) as run:
        mlflow.log_param("epochs", config["num_epochs"])
        mlflow.log_param("backend", "gloo")
        #mlflow.log_metric("run_id", 1)
        '''
        estimator.fit({"training": config["training_data_path"], 
                       "test": config["testing_data_path"]},
                        job_name=config["training_job_name"])
        '''
    print("Training job completed successfully.")

if __name__ == "__main__":
    main()