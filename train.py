import sys
import json

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
    
    import sagemaker
    sagemaker_session = sagemaker.Session()
    print(config["role"])
    
    from sagemaker.pytorch import PyTorch
    estimator = PyTorch(
        entry_point="mlp_pytorch.py",
        role=config["role"],
        framework_version="1.4.0",
        py_version="py3",
        instance_count=2,
        instance_type="ml.c4.xlarge",
        hyperparameters={"epochs": 6, "backend": "gloo"}
    )

if __name__ == "__main__":
    main()