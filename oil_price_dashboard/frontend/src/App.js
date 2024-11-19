import React, { useEffect, useState } from "react";
import axios from "axios";
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from "recharts";

function App() {
  const [historicalData, setHistoricalData] = useState([]);
  const [forecastData, setForecastData] = useState([]);

  // Fetch data from Flask API
  useEffect(() => {
    axios.get("http://127.0.0.1:5000/api/historical-prices")
      .then((response) => setHistoricalData(response.data))
      .catch((error) => console.error("Error fetching historical prices:", error));

    axios.get("http://127.0.0.1:5000/api/forecast")
      .then((response) => setForecastData(response.data))
      .catch((error) => console.error("Error fetching forecast data:", error));
  }, []);

  return (
    <div>
      <h1>Brent Oil Prices Dashboard</h1>

      {/* Historical Prices Chart */}
      <h2>Historical Prices</h2>
      <LineChart width={600} height={300} data={historicalData}>
        <Line type="monotone" dataKey="price" stroke="#8884d8" />
        <CartesianGrid stroke="#ccc" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
      </LineChart>

      {/* Forecast Chart */}
      <h2>Forecasted Prices</h2>
      <LineChart width={600} height={300} data={forecastData}>
        <Line type="monotone" dataKey="price" stroke="#82ca9d" />
        <CartesianGrid stroke="#ccc" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
      </LineChart>
    </div>
  );
}

export default App;
