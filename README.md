# ML Pipeline Project - Kubeflow Deployment Example

This repository contains a **simple Machine Learning project** designed to demonstrate **end-to-end ML pipeline deployment using Kubeflow**.  
The ML task itself is kept simple (predicting house prices) so that the main focus is on **creating and deploying pipelines** in a production-ready setup.

---

## **Project Overview**

This project demonstrates how to structure an ML workflow for **production pipelines**, including:

1. **Data Preprocessing** – Load, clean, and scale data.  
2. **Model Training** – Train a simple Linear Regression model.  
3. **Model Evaluation** – Evaluate model performance (MSE).  
4. **Pipeline Modularity** – Each step can be converted into a **Kubeflow pipeline component**.  
5. **Deployment-Ready** – Designed for Docker containerization and Kubernetes/Kubeflow integration.

> This project is intended as a **starter template** for learning and practicing **Kubeflow ML pipeline deployment**.

---


## **Getting Started**

### **1. Clone the repository**
```bash
git clone https://github.com/your-username/ml_project.git
cd ml_project
```


### **2. Install dependencies**
```
pip install -r requirements.txt
```


### **3. Prepare the dataset**
```
Place your dataset in data/raw/boston.csv.
The expected format includes feature columns and a target column MEDV (Median value of owner-occupied homes).
```

### **4. Run locally**
```
python main.py
```


### **This runs the full ML workflow locally and saves a trained model in models.**

### **Project Components**

```
1. src/data_preprocessing.py – Load and preprocess data.
2. src/train_model.py – Train and save ML model.
3. src/evaluate_model.py – Evaluate model performance.
4. main.py – Run the entire ML workflow locally.
5. pipeline.py – Placeholder for Kubeflow pipeline definition.
```

### **Each script is intended to become a pipeline step in Kubeflow.**

Kubeflow Pipeline Deployment
The main goal of this project is to teach and demonstrate:
Converting ML scripts into Kubeflow pipeline components
Creating end-to-end pipelines for training, evaluation, and deployment
Containerizing ML workloads with Docker
Deploying pipelines on Kubernetes clusters with Kubeflow
Integrating model artifacts and datasets in a production-ready workflow

### **Next Steps**
```
1. Define Kubeflow pipeline components using kfp SDK.
2. Create a Dockerfile for the ML project.
3. Build and push Docker image to a container registry.
4. Deploy the pipeline to a Kubeflow instance on Kubernetes.
5. Monitor and manage ML workflows using Kubeflow dashboard.
```

### **License**
```
MIT License – free to use and modify.
```