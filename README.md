# FraudDetectionModel

GCP Project Name: fraud-detection-data245

Other configs can be found in conf/ folder

# VM Setup with python and PIP

sudo apt update
sudo apt install python3 python3-dev python3-venv
curl https://bootstrap.pypa.io/get-pip.py > get-pip.py
sudo python3 get-pip.py

# Virtual Environment for python

python3 -m venv ml
source ml/bin/activate

# Upgrade pip inside python virtual environment
pip install --upgrade pip

# Install Required Packages

pip install --upgrade fsspec gcsfs pandas numpy sklearn

pip install --upgrade google-cloud-pubsub google-cloud-bigquery google-cloud-storage

