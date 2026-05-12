import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend
} from 'recharts';
import { Paper, Typography, useTheme } from '@mui/material';
import type { WeatherData } from '../api/weatherApi';

interface WeatherChartProps {
  recent: WeatherData[];
  forecast: WeatherData[];
}

const WeatherChart: React.FC<WeatherChartProps> = ({ recent, forecast }) => {
  const theme = useTheme();
  
  // Combine data for the chart
  const data = [
    ...recent.map(d => ({ ...d, type: 'Historical' })),
    ...forecast.map(d => ({ ...d, type: 'Forecast' }))
  ].map(d => ({
    ...d,
    formattedTime: new Date(d.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  }));

  const nowIndex = recent.length;

  return (
    <Paper sx={{ p: 3, borderRadius: 3, boxShadow: 2, height: 400 }}>
      <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
        Temperature Trend (24h History & 24h Forecast)
      </Typography>
      <ResponsiveContainer width="100%" height="85%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={theme.palette.primary.main} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={theme.palette.primary.main} stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={theme.palette.secondary.main} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={theme.palette.secondary.main} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis 
            dataKey="formattedTime" 
            interval={3} 
            tick={{ fontSize: 12 }} 
            axisLine={false}
            tickLine={false}
          />
          <YAxis 
            unit="°C" 
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12 }}
          />
          <Tooltip 
            contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
          />
          <Legend verticalAlign="top" height={36}/>
          
          <ReferenceLine x={data[nowIndex]?.formattedTime} stroke="red" label={{ position: 'top', value: 'Now', fill: 'red', fontSize: 12 }} />
          
          <Area
            name="Historical"
            type="monotone"
            dataKey="temp_c"
            data={data.slice(0, nowIndex + 1)}
            stroke={theme.palette.primary.main}
            strokeWidth={3}
            fillOpacity={1}
            fill="url(#colorTemp)"
            connectNulls
          />
          <Area
            name="Forecast"
            type="monotone"
            dataKey="temp_c"
            data={data.slice(nowIndex)}
            stroke={theme.palette.secondary.main}
            strokeWidth={3}
            strokeDasharray="5 5"
            fillOpacity={1}
            fill="url(#colorForecast)"
            connectNulls
          />
        </AreaChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export default WeatherChart;
