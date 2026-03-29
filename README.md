# Support Ticket Classification & Prioritization System

## 📌 Project Overview
In modern business environments, managing customer support effectively is critical. Support teams process thousands of incoming tickets daily, often leading to slow response times, poorly categorized issues, and delayed resolution for high-priority requests.

This project implements a **Machine Learning System** that reads raw text from support tickets, automatically categorizes them into appropriate departments (e.g., Hardware, Access, HR), and predicts their priority level (High, Medium, Low). 

This is a **decision-support system** that helps SaaS companies and IT teams:
- **Respond Faster:** Tickets are instantly routed to the correct desk.
- **Reduce Backlog:** Automated sorting saves thousands of human hours per month.
- **Improve Customer Satisfaction:** Urgent problems are treated as emergencies, not buried in queues.

## 🛠 Features
- **Text Cleaning Pipeline:** Lowercasing, punctuation stripping, and stopword removal using NLTK.
- **Feature Extraction:** Conversion of cleaned text into dense numerical vectors using `TfidfVectorizer` (Term Frequency-Inverse Document Frequency).
- **Multi-Class Classification (Category):** Uses a robust Machine Learning algorithm (Random Forest / Logistic Regression) to map text features to specific topic groups.
- **Priority Prediction (Priority):** A specialized classifier assigns priority labels based on urgency-signaling language within tickets.
- **Performance Evaluation:** Generates a detailed breakdown of Precision, Recall, Accuracy, and a Confusion Matrix for both models.

## 📂 Project Structure
```text
ticket-classification-system/
│
├── train_model.py          # Complete ML pipeline (Cleaning, Training, Evaluation)
├── requirements.txt        # Required python packages
└── README.md               # Project documentation
```

## 📊 Dataset Requirements
This code is written to utilize the **[IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/itservice-ticket-classification-dataset)** from Kaggle.
Columns required:
- `Document`: The actual ticket text.
- `Topic_group`: The category assigned to the ticket.

*Note: Since the base Kaggle dataset lacks a `Priority` column, our pipeline includes a simulated labeling process built on keyword logic to allow for priority model training. In production, this would be replaced with actual user-defined priority labels.*

## 🚀 How to Run the Pipeline

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data
The script will attempt to dynamically download necessary `nltk` data.

### 3. Run the ML Training Script
```bash
python train_model.py
```
*(If no `tickets.csv` is found locally, the script will automatically generate a highly realistic synthetic dataset of 1,500 rows to demonstrate the complete end-to-end functionality!)*

## 📈 Evaluation Results & Insights
When running this system, the model generates metrics for both **Category** and **Priority**.
- **Categorization:** TF-IDF effectively captures domain-specific vocabulary (e.g., "password" -> Access, "mouse" -> Hardware). The categorization model ensures that specialized technicians receive only relevant tickets to their skills.
- **Prioritization:** By analyzing urgency markers (e.g., "crash", "immediately", "critical"), the priority model effectively clusters tickets requiring immediate human intervention, minimizing system downtime.

**Metrics used for evaluation:**
- **Accuracy:** The percentage of completely correct predictions.
- **Precision:** Of the tickets predicted as 'High' priority, how many were actually 'High'?
- **Recall:** Of all actual 'High' priority tickets, how many did the model identify? (Crucial for ensuring no urgent tickets fall through the cracks).
- **Confusion Matrix:** Allows us to identify cases where the model mixes up structurally similar categories (e.g., Internal Project vs Administration).

---
*Built as a professional real-world implementation for SaaS operations optimization.*
