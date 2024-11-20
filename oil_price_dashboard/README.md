# Oil Price Dashboard (task3 branch)

This branch contains the implementation of the Oil Price Dashboard as a full-stack interactive application for analyzing Brent oil prices and forecasting future trends. The dashboard combines a Flask backend and a React frontend to deliver a seamless and user-friendly experience.

## Branch Overview

This is the third branch of the project, which introduces significant features and improvements:

- **Backend APIs:**
  - `/api/historical-prices`: Serves historical oil price data.
  - `/api/forecast`: Provides LSTM-based future price forecasts.
  - `/api/indicators`: Shares economic data like GDP, inflation, and unemployment.
  - `/api/predict`: Accepts user-provided data for custom predictions.
- **Frontend Integration:**
  - Implements dynamic visualizations and interactive components for data exploration.
- **New Functionalities:**
  - Enhancements in filtering, event highlighting, and responsive design.
- For details on earlier implementations, please refer to the main branch and other branches in the repository.

## Features

### Backend (Flask)

- **API Endpoints:**
  - `/api/historical-prices`: Serves historical oil price data.
  - `/api/forecast`: Provides LSTM-based future price forecasts.
  - `/api/indicators`: Shares economic data like GDP, inflation, and unemployment.
  - `/api/predict`: Accepts user-provided data for custom predictions.
- **LSTM Model:**
  - Predicts oil prices based on historical trends and indicators.

### Frontend (React)

- **Interactive Charts:**
  - Historical trends, forecasts, and correlations with economic events.
- **Event Highlighting:**
  - Visual representation of key events affecting oil prices.
- **Filter Options:**
  - Select specific date ranges for targeted analysis.

## User Experience

- Fully responsive design optimized for desktop, tablet, and mobile devices.
- Intuitive layout for exploring data and insights interactively.

## Installation

### Backend Setup

1. Set up a virtual environment and activate it:

```
    python -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate
```

2. Clone the task3 branch:
```
   git clone -b task3 https://github.com/Endekalu777/Oil_Market_Insight.git
   cd oil_price_dashboard/backend
```

3. Install required requirements
```
    pip install -r requirements.txt
```

4. Start the Flask server:
```
python app.py
```

### Frontend Setup
1. Navigate to the frontend directory:
```
cd oil_price_dashboard/frontend
```

2. Install Node.js dependencies:
```
npm install
```

3. Start the development server:
```
npm start
```
### Usage

1. **Ensure both the Flask backend and React frontend servers are running.**
   - Make sure the Flask server (backend) and the React development server (frontend) are both up and operational.

2. **Access the application at http://localhost:3000.**
   - Open your web browser and navigate to the specified URL to interact with the Oil Price Dashboard.

3. **Explore the following features:**
   - **Historical Oil Price Trends and Predictive Insights:**
     - View historical trends in Brent oil prices and gain insights into past fluctuations.
   - **Interactive Charts with Customizable Date Ranges:**
     - Use interactive charts to analyze data within specific timeframes.
   - **Event Highlights for Geopolitical and Economic Influences:**
     - Identify key events that have impacted oil prices, such as geopolitical tensions or economic shifts.
     
### File Structure
```
oil_price_dashboard/
├── backend/
│   ├── app.py               # Backend logic
│   ├── models/              # Pre-trained LSTM model
│   ├── data/                # Historical datasets
│   ├── requirements.txt     # Dependencies for Flask
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React application logic
│   │   ├── components/      # React components
│   │   │   ├── Chart.js     # Chart rendering
│   │   │   ├── Dashboard.js # Dashboard layout
│   │   │   ├── Filters.js   # Filter functionality
│   ├── package.json         # Frontend dependencies
├── README.md                # Documentation for the third branch

```

Notes
This branch is intended to showcase the latest development stage of the project.
Features and functionality may vary from the main branch and earlier implementations.