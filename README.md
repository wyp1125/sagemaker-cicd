# sagemaker-cicd
This CI/CD demo describes MLOps practices using AWS Sagemaker and Pytorch, to train a deep learning model for predicting lung cancer risks as long as any change is pushed to the main branch, and if the model performance is acceptable, deploy the model as an endpoint after a manual approval.

Key points:
1) Model training and deployment are pipelined and version controlled.
2) Configurations and parameters are separated from the programs.
3) Training parameters and metrics are tracked by a MLFlow tracking server.


