# Customer Churn Prediction API

## Overview

This project uses Machine Learning to predict whether a customer is likely to leave a service.

The model is built using Python and Scikit-learn and is exposed through a REST API developed with FastAPI. Docker is used to package and run the application in a container.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Docker
- Git and GitHub

## Project Features

- Predict customer churn.
- Estimate the probability of churn.
- Classify customers into High or Low churn risk.
- Validate incoming customer data.
- Expose predictions through a REST API.
- Run the application inside a Docker container.

## Run the Project with Docker

### 1. Build the Docker Image

```bash
docker build -t churn-prediction-api .
```

### 2. Start the Container

```bash
docker run -d -p 8000:8000 --name churn-api churn-prediction-api
```

### 3. Open the API Documentation

Open the following URL in your browser:

http://localhost:8000/docs

Use the interactive Swagger UI to test the API endpoints.

### 4. Stop the Container

```bash
docker stop churn-api
```

### 5. Start the Container Again

```bash
docker start churn-api
```

## API Usage

### Endpoint

`POST /predict`

### Input Parameters

| Parameter | Type | Description |
|---|---|---|
| Age | Integer | Customer age |
| Monthly_Charges | Float | Customer's monthly service charges |
| Tenure_Months | Integer | Number of months the customer has used the service |
| Support_Calls | Integer | Number of customer support calls |

### Example Request

```json
{
  "Age": 25,
  "Monthly_Charges": 40,
  "Tenure_Months": 4,
  "Support_Calls": 6
}
```

### Example Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.99999,
  "risk_level": "High"
}
```

**Note:** The response values above are illustrative. Actual predictions depend on the trained model and input data.