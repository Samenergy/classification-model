# Article Relevance Classifier API

A FastAPI-based machine learning service that automatically classifies news articles based on company objectives using transformer-based models and weak supervision techniques.

## Features

- **Custom Company Objectives**: Input your own company objectives for personalized classification
- **Predefined Use Cases**: Choose from mobile wallet, healthcare tech, or edtech scenarios
- **CSV File Upload**: Upload CSV files containing articles for batch classification
- **JSON API**: Send article data directly via JSON for real-time classification
- **Weak Supervision**: Uses semantic similarity to company objectives for training
- **High Performance**: Achieves 79.9% F1-score with transformer-based embeddings

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Ensure model files exist**:
The API expects the trained model files in `models/best_model/`:
- `best_classifier.pkl` - Trained classifier model
- `scaler.pkl` - Feature scaler
- `model_config.json` - Model configuration
- `sentence_model_info.json` - Sentence transformer info

## Quick Start

1. **Start the API server**:
```bash
python main.py
```

2. **The API will be available at**: `http://localhost:8000`

3. **View API documentation**: `http://localhost:8000/docs`

## API Endpoints

### 1. Health Check
```bash
GET /health
```
Check if the API and model are ready.

### 2. Upload CSV with Custom Objective
```bash
POST /classify-upload
```
Upload a CSV file with your company objective.

**Parameters**:
- `file`: CSV file (must contain 'title' and 'content' columns)
- `company_objective`: Your company's objective text

### 3. Classify JSON Data
```bash
POST /classify-data
```
Send article data as JSON with your company objective.

**Request Body**:
```json
{
  "company_objective": "Your company objective here",
  "articles": [
    {
      "title": "Article title",
      "content": "Article content"
    }
  ]
}
```

### 4. Use Predefined Objectives
```bash
POST /classify-predefined
```
Use predefined company objectives for common use cases.

**Parameters**:
- `file`: CSV file
- `use_case`: One of `mobile_wallet`, `healthcare_tech`, or `edtech`

## CSV File Format

Your CSV file must contain at least these columns:
- `title`: Article title
- `content`: Article content/text

Optional columns:
- `id`, `url`, `source`, `published_date`, `created_at`

**Example CSV**:
```csv
title,content,source
"Mobile payments in Africa","Mobile money services are expanding...","Tech News"
"Education technology trends","Digital learning platforms are...","EdTech Weekly"
```

## Classification Categories

Articles are classified into three categories:

1. **Not Relevant** (0): Articles with no meaningful connection to your objectives
2. **Indirectly Useful** (1): Articles related to your broader ecosystem
3. **Directly Relevant** (2): Articles that directly support your company objectives

## Response Format

```json
{
  "results": [
    {
      "title": "Article title",
      "content": "Article content...",
      "prediction": 2,
      "prediction_label": "Directly Relevant",
      "confidence_score": 0.95,
      "probabilities": {
        "not_relevant": 0.01,
        "indirectly_useful": 0.04,
        "directly_relevant": 0.95
      },
      "weak_similarity_score": 0.67,
      "weak_label": 2,
      "weak_label_name": "Directly Relevant"
    }
  ],
  "summary": {
    "total_articles": 100,
    "prediction_distribution": {
      "Not Relevant": 45,
      "Indirectly Useful": 35,
      "Directly Relevant": 20
    },
    "average_confidence": 0.82,
    "high_confidence_predictions": 78,
    "company_objective": "Your company objective",
    "model_weak_supervision_agreement": 0.85
  },
  "model_info": {
    "model_type": "Logistic Regression",
    "performance_metrics": {
      "accuracy": 0.79,
      "precision": 0.81,
      "recall": 0.79,
      "f1_score": 0.799
    }
  }
}
```

## Usage Examples

### Using cURL

**Upload CSV with custom objective**:
```bash
curl -X POST "http://localhost:8000/classify-upload" \
  -F "file=@articles.csv" \
  -F "company_objective=We provide fintech solutions for small businesses in Africa"
```

**Send JSON data**:
```bash
curl -X POST "http://localhost:8000/classify-data" \
  -H "Content-Type: application/json" \
  -d '{
    "company_objective": "We provide fintech solutions for small businesses in Africa",
    "articles": [
      {
        "title": "Mobile payments expanding in Kenya",
        "content": "M-Pesa and other mobile money services are growing rapidly..."
      }
    ]
  }'
```

**Use predefined objective**:
```bash
curl -X POST "http://localhost:8000/classify-predefined" \
  -F "file=@articles.csv" \
  -F "use_case=mobile_wallet"
```

### Using Python

```python
import requests
import pandas as pd

# Upload CSV file
url = "http://localhost:8000/classify-upload"
files = {"file": open("articles.csv", "rb")}
data = {"company_objective": "We provide fintech solutions for small businesses in Africa"}

response = requests.post(url, files=files, data=data)
results = response.json()

# Process results
for article in results["results"]:
    print(f"Title: {article['title']}")
    print(f"Prediction: {article['prediction_label']}")
    print(f"Confidence: {article['confidence_score']:.3f}")
    print("---")
```

## Model Architecture

- **Sentence Transformer**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Classification Model**: Logistic Regression with L2 regularization
- **Feature Scaling**: StandardScaler for normalization
- **Weak Supervision**: Semantic similarity + keyword boosting

## Performance

- **Accuracy**: 79.0%
- **F1-Score**: 79.9%
- **Precision**: 81.2%
- **Recall**: 79.0%

## Troubleshooting

### Common Issues

1. **Model not loaded**: Ensure all model files exist in `models/best_model/`
2. **CSV format errors**: Check that your CSV has 'title' and 'content' columns
3. **Memory issues**: For large CSV files, process in smaller batches
4. **Encoding errors**: The API tries multiple encodings (UTF-8, Latin-1, CP1252)

### Error Messages

- `"Missing required columns"`: Add 'title' and 'content' columns to your CSV
- `"Model is not loaded"`: Check that model files exist and are readable
- `"Error processing CSV"`: Verify CSV format and encoding

## Development

To run in development mode:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## License

This project is for educational and research purposes.
