# 1. Create a virtual environment using Python 3.11
uv venv .venv --python 3.11

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install required dependencies
uv add flask scikit-learn==1.6.1 requests gunicorn

predict_v1.py -- Questions 3 & 4
predict_v2.py --- Questions 5 & 6
Step-by-Step Execution & Verification

Step 1: Verification for Q1 & Q2 (Tool Versions)

# Q1: Check uv version
uv --version

# Q2: Check scikit-learn version
python -c "import sklearn; print(sklearn.__version__)"


Step 2: Verification for Q3 & Q4 (Local Flask Endpoint)

Start the local Flask web service in Terminal 1:

python predict_v1.py


Send the customer payload in Terminal 2:

python test_client.py


Expected Output:

{"churn": true, "probability": 0.952}


Step 3: Verification for Q5 & Q6 (Docker Containerization)

Build the Docker container image:

docker build --no-cache -t zoomcamp-hw5 .


Run the container mapping port 9696:

docker run -it --rm -p 9696:9696 zoomcamp-hw5


Verify the containerized endpoint in a second terminal:

python test_client.py


Expected Output for Q6:

{"churn": true, "probability": 0.757}


File Architecture Overview

predict_v1.py: Web service serving predictions locally using pipeline_v1.bin.

predict_v2.py: Web service serving predictions inside Docker using pipeline_v2.bin.

test_client.py: Client script sending POST requests to test endpoint responses.

Dockerfile: Container specification building from agrigorev/zoomcamp-model:2025 and executing predict_v2.py via Gunicorn.

