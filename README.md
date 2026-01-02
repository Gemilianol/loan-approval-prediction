project description + “How to get the data” section.



# Just for simplicity and to avoid use volumenes in this case will train then model when a new container / pod starts assuming that, the same model's setup will return the same model.













# Kubernetes execution commands ordered (Always run your builds from the project root.)
kind delete cluster --name loan-prediction
kind create cluster --name loan-prediction --config kind-config.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl wait --namespace ingress-nginx --for=condition=ready pod --selector=app.kubernetes.io/component=controller --timeout=90s
docker build -t mlflow:custom -f backend/mlflow/Dockerfile ./backend/mlflow
docker build -t backend:custom -f backend/Dockerfile ./backend
docker build -t frontend:custom -f frontend/Dockerfile ./frontend
kind load docker-image mlflow:custom --name loan-prediction
kind load docker-image backend:custom --name loan-prediction
kind load docker-image frontend:custom --name loan-prediction
# Bulk Approach (Could fail)
kubectl apply -f k8s/
# Safe Approach
# 1. Build the foundation and the "brain" (MLflow)
kubectl apply -f k8s/pvc.yaml -f k8s/mlflow-deployment.yaml -f k8s/ingress.yaml
# 2. Wait until MLflow is actually 'Running' 
kubectl wait --for=condition=ready pod -l app=mlflow --timeout=60s
# 3. Apply the rest
kubectl apply -f k8s/backend-deployment.yaml -f k8s/training-job.yaml
# 4. Wait for the Backend to be fully "Ready"
kubectl wait --for=condition=ready pod -l app=backend --timeout=120s
# 5. Apply the frontend
kubectl apply -f k8s/frontend-deployment.yaml
# 6. Finally check the pods
kubectl get pods
# If you just want to verify MLflow is working right now, use port-forward.
<!-- kubectl port-forward pod/mlflow-hash 5000:5000 -->
# Or use ingress.yaml directly
<!-- kubectl apply -f k8s/ingress.yaml -->

successfully orchestrated a stateful, dependent microservices architecture on Kubernetes.

Note here:
localhost isn't a "domain name," Ingress works best if we give it a name. 
Add this line to your computer's /etc/hosts (Mac/Linux) or
 C:\Windows\System32\drivers\etc\hosts (Windows):
127.0.0.1  mlflow.local 
and all which are necessary. 
sudo nano /etc/hosts to edit the file from terminal. Ctrl + X to exit.