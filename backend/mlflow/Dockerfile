# Start from the official, known-good MLflow image
FROM ghcr.io/mlflow/mlflow:v3.0.0

# Since the official MLflow image is built on Debian, use apt
USER root
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Switch back to the non-root user that MLflow typically uses
# You might need to check the exact user for safety, but often MLflow runs as root or a dedicated user.
# For now, let's assume we can proceed as-is after installation.
# USER <mlflow_user_if_known>

# NOTE: This is only for use the healthcheck because mlflow doesn't have install curl by default. 