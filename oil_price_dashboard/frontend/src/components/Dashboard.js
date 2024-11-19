import React, { useState, useEffect } from 'react';
import Chart from 'Chart';
import Filters from './Filters';

const Dashboard = () => {
  const [forecast, setForecast] = useState([]);
  const [prices, setPrices] = useState([]);
  const [indicators, setIndicators] = useState([]);

  useEffect(() => {
    fetch('/api/forecast')
      .then((res) => res.json())
      .then(setForecast);

    fetch('/api/prices')
      .then((res) => res.json())
      .then(setPrices);

    fetch('/api/indicators')
      .then((res) => res.json())
      .then(setIndicators);
  }, []);

  return (
    <div>
      <h1>Brent Oil Price Dashboard</h1>
      <Filters />
      <Chart data={prices} title="Brent Oil Prices" />
      <Chart data={forecast} title="LSTM Forecasted Prices" />
      <Chart data={indicators} title="Oil Indicators" />
    </div>
  );
};

export default Dashboard;
