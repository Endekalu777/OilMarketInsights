import React, { useState, useEffect } from 'react';
import PriceChart from './PriceChart';
import EventTimeline from './EventTimeline';
import AnalyticsPanel from './AnalyticsPanel';
import ModelResults from './ModelResults';
import { fetchDashboardData, fetchModelAnalysis } from '../services/api';
import { Box, Grid, Container, CircularProgress } from '@mui/material';

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [modelResults, setModelResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [dateRange, setDateRange] = useState({
    start: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end: new Date().toISOString().split('T')[0]
  });

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const [dashboardResponse, modelResponse] = await Promise.all([
          fetchDashboardData(dateRange.start, dateRange.end),
          fetchModelAnalysis()
        ]);

        setDashboardData(dashboardResponse.data);
        setModelResults(modelResponse.data);
      } catch (error) {
        console.error('Error loading dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [dateRange]);

  if (loading || !dashboardData) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="xl">
      <Box sx={{ flexGrow: 1, mt: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <PriceChart 
              priceData={dashboardData.price_data}
              forecast={dashboardData.forecast}
              events={dashboardData.events}
              dateRange={dateRange}
              onDateRangeChange={setDateRange}
            />
          </Grid>
          <Grid item xs={12} md={8}>
            <EventTimeline events={dashboardData.events} />
          </Grid>
          <Grid item xs={12} md={4}>
            <AnalyticsPanel metrics={dashboardData.metrics} />
          </Grid>
          <Grid item xs={12}>
            <ModelResults results={modelResults} />
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default Dashboard;