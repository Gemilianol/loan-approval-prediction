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

RUN python -m src.pipelines.training_pipeline

# Here I need to train a model
#RUN python src/pipelines/training_pipeline.py --force_retrain

# Use Gunicorn for production
ENV PORT=2000
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:2000", "--workers", "2", "--timeout", "120"]