FROM python:3.11-slim

# For a learning project, using the absolute newest Python 
# version often causes headaches like this. (compile problems)
# Python 3.11 has pre-built wheels for everything (Numpy, Pandas, MLflow). 
# pip will just download them and install instantly—no compilation needed.

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

COPY app.py .

# Hint: THIS RUN as build time, so in this way the model is saved 
# on mlruns locally in the container not on mlflow server as expected. 

#RUN python -m src.pipelines.training_pipeline --force_retrain

# When you use -m, Python adds the Current Working Directory to sys.path.
# sys.path[0]: /app (Because WORKDIR /app)
# Using -m ensures your code behaves exactly as it does when you import 
# it elsewhere. It respects your package structure.

# Use Gunicorn for production
ENV PORT=2000

# This command and the command for train the model has been moved to docker-compose.

#CMD ["gunicorn", "app:app", "-b", "0.0.0.0:2000", "--workers", "2", "--timeout", "120"]

# ------------------------------------------------------

# gunicorn: The Process Manager. Correct for Flask (WSGI).

# app:app: The first app is the filename (e.g., app.py), and the second app 
# is the variable name inside that file (app = Flask(__name__)).

# -b 0.0.0.0:2000: Binds to Port 2000 on all interfaces. Correct for Docker.

# --workers 2: Handles parallel requests. If one worker is busy predicting, the other can still say "Hello".

# --timeout 120: This is excellent for ML. Default is usually 30 seconds. 
# If your model takes 40 seconds to load into memory or 35 seconds to generate 
# a complex prediction, a standard server would kill the process. 120s gives your heavy ML model room to breathe.