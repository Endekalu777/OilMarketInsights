import React from 'react';
import { Paper, Box, Typography, Grid } from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

const ModelResults = ({ results }) => {
  if (!results) return null;

  return (
    <Paper elevation={3}>
      <Box p={3}>
        <Typography variant="h6" gutterBottom>
          Model Analysis Results
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Typography variant="subtitle1">LSTM Model Performance</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={results.lstm_results}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" />
                <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted" />
              </LineChart>
            </ResponsiveContainer>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1">VAR Model Results</Typography>
            <pre>{JSON.stringify(results.var_results, null, 2)}</pre>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1">Markov Switching Model Results</Typography>
            <pre>{JSON.stringify(results.markov_results, null, 2)}</pre>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
};

export default ModelResults;