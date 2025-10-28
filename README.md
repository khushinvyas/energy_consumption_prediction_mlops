End-to-End MLOps: Energy Consumption Prediction
This repository contains the complete source code and configuration for an end-to-end MLOps project focused on predicting household energy consumption. The project demonstrates a full lifecycle from data preprocessing and experiment tracking to containerization and automated cloud deployment.
![alt text](https'//github.com/khushinvyas/energy_consumption_prediction_mlops/actions/workflows/ci-cd.yml/badge.svg')
Table of Contents
Project Overview
Tech Stack
MLOps Pipeline Architecture
Getting Started
Prerequisites
Installation & Setup
Project Structure
How to Reproduce the Pipeline
1. Data Preprocessing
2. Running Experiments
3. Training the Final Model
Containerization with Docker
CI/CD with GitHub Actions
Cloud Deployment on AWS
Project Overview
The primary goal of this project is to predict the Global_active_power from the "Individual household electric power consumption" dataset. Beyond just building a model, this project focuses on implementing a robust, automated, and reproducible MLOps workflow.
Key Features:
Data Versioning: Using DVC to track changes in data and models.
Experiment Tracking: Leveraging DVC Experiments to systematically run and compare different models (RandomForest, XGBoost) and hyperparameters.
Reproducible Pipeline: A DVC pipeline (dvc.yaml) ensures that every step from data processing to model evaluation is versioned and reproducible.
Web Application: A Flask-based web interface to serve the final model for real-time predictions.
Containerization: A Dockerfile to package the application, its dependencies, and the trained model into a portable container.
Automated CI/CD: A GitHub Actions workflow that automatically builds a new Docker image and deploys it to an AWS EC2 instance on every push to the master branch.
Large File Handling: Git LFS is configured to handle large model files that exceed GitHub's standard limits.
Tech Stack
Data & Modeling: Pandas, Scikit-learn, XGBoost
MLOps & Versioning: Git, DVC, Git LFS
Application: Flask, Gunicorn
Containerization: Docker
CI/CD: GitHub Actions
Cloud Provider: Amazon Web Services (AWS)
Hosting: EC2
Deployment: Systems Manager (SSM)
Permissions: IAM
MLOps Pipeline Architecture
The project follows a modern MLOps architecture:
Code & Data: All code is versioned with Git. DVC is used to version large data files and models, which are stored in a remote storage (e.g., S3).
Experimentation: Developers run experiments locally using dvc exp run. Results are tracked and compared with dvc exp show.
CI (Continuous Integration): A git push to the master branch triggers the GitHub Actions workflow.
Build: The workflow builds the Flask application into a Docker image.
CD (Continuous Delivery): The newly built image is pushed to a Docker Hub registry.
CD (Continuous Deployment): The workflow securely sends a command via AWS SSM to the EC2 instance.
Deployment: The EC2 instance pulls the new Docker image from the registry and restarts the container, making the update live.
<!-- You can create and host a simple diagram to make this more visual -->
Getting Started
Prerequisites
Python 3.9+
Git and Git LFS
DVC (pip install dvc)
Docker Desktop
Installation & Setup
Clone the repository:
code
Bash
git clone https://github.com/khushinvyas/energy_consumption_prediction_mlops.git
cd energy_consumption_prediction_mlops
Install Git LFS:
code
Bash
git lfs install
Create a virtual environment and install dependencies:
code
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Pull DVC-tracked data and models:
(Note: You would first need to configure a DVC remote storage)
code
Bash
dvc pull
Project Structure
code
Code
.
├── .dvc/                   # DVC internal files
├── .github/workflows/      # GitHub Actions CI/CD pipeline
├── data/
│   ├── raw/                # Raw dataset
│   └── processed/          # Processed and split data (tracked by DVC)
├── models/                 # Trained model files (tracked by DVC & Git LFS)
├── templates/              # HTML templates for Flask app
├── app.py                  # Flask application script
├── dvc.yaml                # DVC pipeline definition
├── evaluate.py             # Model evaluation script
├── Dockerfile              # Docker configuration
├── params.yaml             # Parameters for experiments
├── preprocess.py           # Data preprocessing script
├── requirements.txt        # Python dependencies
└── train.py                # Model training script
How to Reproduce the Pipeline
1. Data Preprocessing
This stage cleans the raw data and splits it into training and testing sets.
code
Bash
dvc repro preprocess
2. Running Experiments
To run all defined experiments from params.yaml:
code
Bash
dvc exp run --all
To view the results:
code
Bash
dvc exp show
3. Training the Final Model
After identifying the best experiment, apply its parameters and reproduce the full pipeline to generate the final model.
code
Bash
# Apply the parameters from the best experiment
dvc exp apply <experiment_name>

# Run the full pipeline
dvc repro
Containerization with Docker
To build and run the application as a Docker container locally:
Build the image:
code
Bash
docker build -t energy-prediction-app .
Run the container:
code
Bash
docker run -p 5000:5000 energy-prediction-app
The application will be accessible at http://localhost:5000.
CI/CD with GitHub Actions
The workflow is defined in .github/workflows/ci-cd.yml and performs the following on every push to master:
Builds the Docker image.
Pushes the image to Docker Hub.
Deploys the new image to the configured AWS EC2 instance.
Required GitHub Secrets for Deployment:
DOCKERHUB_USERNAME: Your Docker Hub username.
DOCKERHUB_TOKEN: A Docker Hub Personal Access Token.
AWS_ACCESS_KEY_ID: AWS IAM User Access Key.
AWS_SECRET_ACCESS_KEY: AWS IAM User Secret Key.
AWS_REGION: The AWS region of the EC2 instance (e.g., us-east-1).
AWS_EC2_INSTANCE_ID: The ID of the target EC2 instance.
Cloud Deployment on AWS
The application is deployed to an AWS EC2 instance. The deployment is managed by GitHub Actions using AWS Systems Manager (SSM) for a secure, keyless process.
AWS Setup Prerequisites:
An EC2 instance with Docker installed.
An IAM Role (EC2-SSM-Role) attached to the instance with the AmazonSSMManagedInstanceCore policy.
An IAM User with AmazonSSMFullAccess permissions, whose credentials are used in GitHub Secrets.
A Security Group allowing inbound traffic on port 5000 (for the app) from anywhere (0.0.0.0/0). Port 22 (SSH) is not required and can be closed for enhanced security.
