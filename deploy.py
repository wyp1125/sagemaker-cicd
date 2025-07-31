import sys
import json
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    with open(config_file, 'r') as file:
            config = json.load(file)
    from sagemaker.estimator import Estimator
    attached_estimator = Estimator.attach(training_job_name=config["training_job_name"])
    predictor = attached_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.large'
    )

if __name__ == "__main__":
    main()