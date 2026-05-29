# 🌸 Iris MLP Classifier — MLSD Project

End-to-end MLOps pipeline using Git, DVC, Docker, GitHub Actions, Kubernetes, and Streamlit.

## 🗂️ Project Structure

```
mlsd-project/
├── train.py                    # MLP training script
├── app.py                      # Streamlit inference UI
├── Dockerfile                  # Container definition
├── requirements.txt
├── dvc.yaml                    # DVC pipeline stages
├── .dvcignore
├── .gitignore
├── data/
│   └── iris.csv                # Raw data (DVC-tracked)
├── model/
│   ├── mlp_model.pkl           # Trained model (DVC-tracked)
│   ├── scaler.pkl              # Fitted scaler (DVC-tracked)
│   └── metrics.json            # Metrics (Git-tracked)
├── k8s/
│   ├── deployment.yaml         # Kubernetes Deployment
│   └── service.yaml            # Kubernetes Service (LoadBalancer)
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions pipeline
```

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| **Git** | Source code version control |
| **DVC** | Data & model versioning, pipeline reproducibility |
| **Docker** | Containerize the Streamlit app |
| **GitHub Actions** | Auto-trigger train → build → deploy on push to main |
| **Kubernetes** | Deploy and scale the container |
| **Streamlit** | Interactive UI for predictions |

---

## 🚀 Local Setup

### 1. Clone & install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/mlsd-project.git
cd mlsd-project
pip install -r requirements.txt
pip install dvc
```

### 2. Train the model (via DVC)
```bash
dvc repro
```
This runs `train.py`, saves artifacts to `data/` and `model/`, and logs metrics to `model/metrics.json`.

Check metrics:
```bash
dvc metrics show
```

### 3. Run the Streamlit app locally
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 🐳 Docker

### Build the image
```bash
# First train the model so model/ artifacts exist
dvc repro

# Then build
docker build -t iris-mlp-app .
```

### Run the container
```bash
docker run -p 8501:8501 iris-mlp-app
```
Open http://localhost:8501.

---

## ☸️ Kubernetes (local with Minikube)

### Start Minikube
```bash
minikube start
```

### Update the image name
In `k8s/deployment.yaml`, replace `YOUR_GITHUB_USERNAME` with your actual GitHub username, and `IMAGE_TAG` with `latest` for local testing.

### Apply manifests
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Access the app
```bash
minikube service iris-mlp-service
```

---

## 🔁 GitHub Actions CI/CD

On every push to `main`, the pipeline automatically:
1. **Trains** the model via `dvc repro`
2. **Prints** metrics from `model/metrics.json`
3. **Builds** and **pushes** a Docker image to GitHub Container Registry (ghcr.io)
4. **Deploys** to Kubernetes using your kubeconfig

### Required GitHub Secrets
Go to your repo → Settings → Secrets → Actions and add:

| Secret | Value |
|---|---|
| `KUBECONFIG` | Contents of your `~/.kube/config` file (base64 or raw) |

The `GITHUB_TOKEN` secret is auto-provided by GitHub Actions.

---

## 📊 Model Details

- **Algorithm**: MLPClassifier (Multi-Layer Perceptron)
- **Architecture**: `64 → 32` hidden layers, ReLU activation, Adam optimizer
- **Dataset**: Iris (150 samples, 4 features, 3 classes)
- **Accuracy**: ~97% on test set

---

## 🔄 DVC Pipeline

The `dvc.yaml` defines one stage:

```
train.py  ──►  data/iris.csv
               model/mlp_model.pkl
               model/scaler.pkl
               model/metrics.json  (Git-tracked)
```

Re-run only if `train.py` changes:
```bash
dvc repro
```

Compare metrics across Git commits:
```bash
dvc metrics diff HEAD~1
```
