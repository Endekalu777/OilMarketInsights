# Brent Oil Prices Analysis and Prediction System

This project aims to develop a comprehensive analysis and prediction system for Brent oil prices. It includes tasks such as defining data workflows, model building for time series analysis, integrating economic and geopolitical factors, and deploying an interactive dashboard to visualize the results. Below is a breakdown of each task and its deliverables.

## Overview
This project provides an in-depth analysis and predictive modeling of Brent oil prices using:

- Data analysis and preprocessing
- Time series and econometric models
- Machine learning for pattern recognition
- Interactive dashboard visualization

## Folder Strucutre

```
├── notebooks/                     # Jupyter notebooks for analysis and prototyping
│   ├── EDA_event_analysis.ipynb   # Exploratory Data Analysis for events
│   ├── Oil_price_analysis.ipynb   # Analysis of historical oil price data
│   ├── pre_processor.ipynb        # Data preprocessing notebook
│   ├── price_forecast.ipynb       # Notebook for forecasting prices
│
├── scripts/                       # Python scripts for backend processing and analysis
│   ├── EDA_event_analysis.py      # EDA implementation script
│   ├── Oil_price_analysis.py      # Script for oil price analysis
│   ├── fetch_external_factors.py  # Script to retrieve external factors (e.g., economic indicators)
│   ├── pre_processor.py           # Preprocessing logic for raw data
│   ├── price_forecasts.py         # Forecasting models and predictions
│   └── __init__.py  
├──oil_price_dashboard/
│   │
│   ├── backend/                # Backend APIs built with Flask
│   │   ├── app.py              # Main Flask app
│   │   ├── models/             # Scripts for model training and inference
│   │   ├── data/               # Datasets used for analysis
│   │   ├── requirements.txt    # Python dependencies for the backend
│   │   └── tests/              # Unit tests for the backend
│   │
│   ├── frontend/               # Frontend built with React
│   │   ├── public/             # Static files for the frontend
│   │   ├── src/                # Source code for the React app
│   │   │   ├── components/     # React components (e.g., charts, filters)
│   │   │   └── App.js          # Main React entry point
│   │   └── package.json        # Node.js dependencies for the frontend
│
├── Dockerfile              # Docker configuration for containerization
├── docker-compose.yml      # Docker Compose setup for the full stack
├── README.md               # Documentation for the project
└──requirements.txt
```
## Key Features
1. **Modeling Brent Oil Prices (Task1 branch)**
   - Time series models: ARIMA, GARCH
   - Machine learning models: LSTM for complex patterns
   - Factor analysis: Economic, technological, and geopolitical influences


3. **Interactive Dashboard (Task3 branch)**
   - Backend (Flask) and React frontend
   - Visualize historical trends, forecasts, and key indicators
   - User-friendly interface with interactive data exploration

## Technical Implementation
### Task 1: Data Analysis Workflow
- Defining Data Workflow: Detailed steps for analyzing Brent oil prices, understanding model inputs, and determining assumptions.
- Understanding Models and Data: Familiarize with time series models like ARIMA and GARCH, exploring their application to Brent oil data. Identify factors influencing price fluctuations.

### Task 2: Modeling & Analysis
- Time Series and Econometric Modeling: Implementation of ARIMA, GARCH, and VAR for price prediction and multivariate analysis.
- Additional Factors: Economic (GDP, inflation), technological (renewable energy growth), and geopolitical (trade policies) influences on oil prices.
- Machine Learning Models: LSTM networks to capture dependencies and non-linear relationships in oil prices.

### Task 3: Interactive Dashboard
- Backend (Flask): API development to serve data and model outputs to the React frontend.
- Frontend (React): Interactive visualizations, with filters and event highlights, to explore the impact of specific events on oil prices.
- Dashboard Features: Responsive design for multiple devices, with options to view volatility, historical trends, forecasts, and specific event influences.

## Setup Instructions
1. **Clone the Repository**
    ```bash
    git clone https://github.com/YourUsername/Oil_Market_Insights.git
    ```

2. **Virtual Environment Setup**
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Unix
    venv\Scripts\activate     # For Windows
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
- **API**: Access the API endpoints to retrieve model predictions and analysis data.
- **Dashboard**: Run the dashboard to interactively explore Brent oil price trends and event impacts.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or more information, please contact Endekalu.simon.haile@example.com.