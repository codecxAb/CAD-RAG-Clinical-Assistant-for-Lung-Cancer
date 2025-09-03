CAD-RAG: An Explainable, Evidence-Based Clinical Decision Support System
An advanced, multi-component system designed to assist clinicians in the diagnosis of lung cancer from CT scans. This project integrates a high-accuracy Convolutional Neural Network (CNN) with state-of-the-art Explainable AI (XAI) and a Retrieval-Augmented Generation (RAG) engine to provide not just a prediction, but also visual evidence and contextual clinical information.

📜 Project Overview
Lung cancer is a leading cause of cancer-related mortality, where early and accurate diagnosis is critical for improving patient survival rates. While deep learning models have shown remarkable accuracy in classifying medical images, their "black-box" nature is a significant barrier to clinical adoption.   

This project, CAD-RAG, addresses this challenge by creating a holistic decision support tool. Our system:

Classifies lung CT scans into categories such as Normal, Benign, and Malignant subtypes (e.g., Adenocarcinoma, Squamous Cell Carcinoma) using a powerful CNN.   

Explains its predictions visually using Grad-CAM heatmaps, highlighting the exact regions in the CT scan that influenced the model's decision.   

Informs the clinician by using a RAG engine to query a curated medical knowledge base, providing a summary of relevant clinical guidelines, treatment options, and recent research based on the classification result.

✨ Key Features
High-Accuracy Classification: Utilizes transfer learning with proven CNN architectures like VGG16, ResNet50, or Xception for robust classification of lung cancer subtypes.

Visual Explainability (XAI): Implements Grad-CAM to generate intuitive heatmaps, making the AI's decision process transparent and trustworthy for clinicians.   

Evidence-Based Context (RAG): Integrates a RAG pipeline with LangChain and a vector database to provide clinicians with up-to-date, relevant information from medical literature, reducing the cognitive load and supporting treatment planning.

Interactive Web Interface: A user-friendly application built with Streamlit for easy demonstration, allowing users to upload a CT scan and receive a comprehensive diagnostic brief in real-time.   

Scalable Monorepo Architecture: All components (ML model, backend API, frontend) are organized in a single repository for streamlined development, versioning, and future deployment.

🏗️ System Architecture
This project is structured as a monorepo to ensure seamless integration and management across its different components.

/ml/: Contains Jupyter notebooks for experimentation, Python scripts for data preprocessing, and the model training pipeline. The final trained model artifacts are stored here.

/backend/: A production-ready FastAPI application that serves the trained model via a REST API. It includes the core logic for the RAG pipeline and XAI heatmap generation.

/streamlit_demo/: A self-contained Streamlit application for rapid prototyping and demonstration of the project's core features.

/frontend/: Placeholder for the future React-based production frontend.

/docs/: Project documentation, including the presentation plan and architecture diagrams.

🛠️ Tech Stack
Component	Technologies & Libraries
Machine Learning	Python, TensorFlow, Keras/PyTorch, Scikit-learn, OpenCV
Backend API	FastAPI, Uvicorn, LangChain, ChromaDB (Vector Store)
Demo App	
Streamlit    

Deployment	
Docker, Heroku/Render (for backend), Vercel/Netlify (for frontend)    

💾 Dataset
This model is trained on publicly available, annotated medical imaging datasets to ensure reproducibility and robust performance. The primary dataset used is the IQ-OTH/NCCD Lung Cancer Dataset, which contains CT scan images classified into Normal, Benign, and Malignant cases. For subtype classification, the LC25000 dataset may also be used.   

🚀 Getting Started
Prerequisites
Python 3.10+

Conda (or another virtual environment manager)

Git

Installation & Setup
Clone the repository:

Bash

git clone https://github.com/codecxAb/CAD-RAG-Clinical-Assistant-for-Lung-Cancer.git
cd CAD-RAG-Clinical-Assistant-for-Lung-Cancer
Set up the Machine Learning Environment:
(This environment is for training the model)

Bash

cd ml
conda create --name cad-rag-ml python=3.10
conda activate cad-rag-ml
pip install -r requirements.txt
Set up the Backend Environment:
(This environment is for running the API and demo)

Bash

cd../backend
conda create --name cad-rag-backend python=3.10
conda activate cad-rag-backend
pip install -r requirements.txt
How to Run
Train the Model (Optional, if using a pre-trained model):

Navigate to the /ml/scripts/ directory.

Run the training script:

Bash

# Make sure you are in the 'cad-rag-ml' conda environment
python train.py
Ensure the final model artifact (e.g., lung_classifier_v1.h5) is saved in the /ml/models/ directory.

Run the Backend API Server:

Navigate to the /backend/ directory.

Start the FastAPI server:

Bash

# Make sure you are in the 'cad-rag-backend' conda environment
uvicorn app.main:app --reload
The API will be available at http://127.0.0.1:8000.

Run the Streamlit Demo Application:

Navigate to the /streamlit_demo/ directory.

Launch the Streamlit app:

Bash

# Make sure you are in the 'cad-rag-backend' conda environment
streamlit run app.py
The application will open in your web browser.

