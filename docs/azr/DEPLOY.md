# Deploying AZR-Planner Microservice

This guide provides instructions for building the AZR-Planner Docker image and deploying it to a local Kubernetes cluster (Kind or Minikube) using the provided Helm chart.

## Prerequisites

1.  **Docker**: Ensure Docker is installed and running. (https://www.docker.com/get-started)
2.  **Kubectl**: Install `kubectl`. (https://kubernetes.io/docs/tasks/tools/install-kubectl/)
3.  **Helm**: Install Helm v3. (https://helm.sh/docs/intro/install/)
4.  **Kind or Minikube**:
    *   **Kind**: (https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
    *   **Minikube**: (https://minikube.sigs.k8s.io/docs/start/)

## 1. Setup Local Kubernetes Cluster

### Using Kind

Create a cluster:
```bash
kind create cluster --name azr-planner-dev
```

Set kubectl context:
```bash
kubectl cluster-info --context kind-azr-planner-dev
```

### Using Minikube

Start Minikube:
```bash
minikube start --profile azr-planner-dev
```
Enable the ingress controller if you plan to use Ingress (optional for this chart's default setup):
```bash
minikube addons enable ingress --profile azr-planner-dev
```
Set kubectl context:
```bash
kubectl config use-context azr-planner-dev
```

## 2. Build and Load Docker Image

The AZR-Planner service is defined by `docker/azr-planner.Dockerfile`.

1.  **Navigate to the repository root.**
2.  **Build the Docker image:**
    ```bash
    docker build -f docker/azr-planner.Dockerfile -t azr-planner:local .
    ```
    *Note: Replace `azr-planner:local` with your desired image name and tag.*

3.  **Load the image into your local cluster:**

    ### For Kind:
    ```bash
    kind load docker-image azr-planner:local --name azr-planner-dev
    ```

    ### For Minikube:
    (Ensure your Docker daemon is the one Minikube uses)
    ```bash
    eval $(minikube -p azr-planner-dev docker-env)
    # Then build the image directly within Minikube's Docker daemon:
    docker build -f docker/azr-planner.Dockerfile -t azr-planner:local .
    # Or, if already built externally:
    # minikube image load azr-planner:local --profile azr-planner-dev
    ```
    To switch back to your host's Docker daemon after building (if you used `eval`):
    ```bash
    eval $(minikube docker-env -u -p azr-planner-dev)
    ```

## 3. Deploy with Helm

The Helm chart is located in `deploy/azr-planner-chart/`.

1.  **Navigate to the repository root.**

2.  **Install/Upgrade the Helm chart:**
    ```bash
    helm upgrade --install azr-planner-release \
      deploy/azr-planner-chart \
      --set image.repository=azr-planner \
      --set image.tag=local \
      --set image.pullPolicy=IfNotPresent \
      --namespace default # Or your preferred namespace
    ```
    *   `azr-planner-release`: This is the name for your Helm release.
    *   `deploy/azr-planner-chart`: Path to the chart.
    *   `image.repository=azr-planner` and `image.tag=local`: These tell Helm to use the image you just built and loaded.
    *   `image.pullPolicy=IfNotPresent`: Ensures K8s uses the local image if the tag matches and doesn't try to pull from a remote registry.

3.  **Check deployment status:**
    ```bash
    kubectl get deployments -n default
    kubectl get pods -n default -l app.kubernetes.io/instance=azr-planner-release
    ```
    Wait for the pod to be in the `Running` state and `1/1` in the `READY` column.

4.  **Access the service (within the cluster):**
    The service is deployed as `ClusterIP` by default. To access it, you can use port-forwarding:
    ```bash
    # Get the pod name
    POD_NAME=$(kubectl get pods -n default -l "app.kubernetes.io/instance=azr-planner-release,app.kubernetes.io/name=azr-planner" -o jsonpath="{.items[0].metadata.name}")

    # Port-forward
    echo "Access the service at http://localhost:8000"
    kubectl port-forward $POD_NAME 8000:8000 -n default
    ```
    Now you can access the application at `http://localhost:8000`. For example, try the health check:
    ```bash
    curl http://localhost:8000/health
    ```

## 4. Interacting with Prometheus (Optional)

If you have Prometheus installed in your cluster and it's configured to discover ServiceMonitors or scrape annotated services:
*   The Service created by this chart has the annotations `prometheus.io/scrape: "true"` and `prometheus.io/path: "/metrics"`.
*   Prometheus should automatically pick up the `/metrics` endpoint of the AZR-Planner service on port 8000.

## 5. Cleanup

### Using Kind
```bash
kind delete cluster --name azr-planner-dev
```

### Using Minikube
```bash
minikube stop --profile azr-planner-dev
minikube delete --profile azr-planner-dev
```

### Uninstall Helm Release
```bash
helm uninstall azr-planner-release -n default
```
