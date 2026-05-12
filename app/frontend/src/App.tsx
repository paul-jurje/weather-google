import { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  CircularProgress,
  Alert,
  AppBar,
  Toolbar,
  CssBaseline,
  ThemeProvider,
  createTheme,
  Button,
  Stack,
  LinearProgress
} from '@mui/material';
import {
  Thermometer,
  Droplets,
  Wind,
  CloudRain,
  Cloud,
  Sun,
  RefreshCw,
  Zap,
  ArrowUp
} from 'lucide-react';
import { fetchForecast, refreshForecastData, startFineTuning, getFineTuneStatus } from './api/weatherApi';
import type { ForecastResponse, FineTuneStatus } from './api/weatherApi';
import MetricCard from './components/MetricCard';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1a73e8', // Google Blue
    },
    secondary: {
      main: '#34a853', // Google Green
    },
    background: {
      default: '#f8f9fa',
    },
  },
  typography: {
    fontFamily: '"Google Sans", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

function App() {
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [fineTuningStatus, setFineTuningStatus] = useState<FineTuneStatus | null>(null);
  const [selectedHour, setSelectedHour] = useState<number | null>(null);

  const getData = async () => {
    try {
      const result = await fetchForecast();
      setData(result);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch weather data:', err);
      setError('Could not load weather data. Please ensure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    getData();
    const interval = setInterval(getData, 1000 * 60 * 15); // Refresh every 15 mins
    return () => clearInterval(interval);
  }, []);

  // Poll fine-tune status if running
  useEffect(() => {
    let pollInterval: number | undefined;

    if (fineTuningStatus?.status === 'running') {
      pollInterval = setInterval(async () => {
        try {
          const status = await getFineTuneStatus();
          setFineTuningStatus(status);
          if (status.status !== 'running') {
            clearInterval(pollInterval);
            getData(); // Refresh data after completion
          }
        } catch (err) {
          console.error('Failed to poll fine-tune status:', err);
        }
      }, 2000);
    }

    return () => clearInterval(pollInterval);
  }, [fineTuningStatus?.status]);

  const handleRefresh = async () => {
    try {
      setRefreshing(true);
      await refreshForecastData();
      await getData();
    } catch (err) {
      console.error('Refresh failed:', err);
    } finally {
      setRefreshing(false);
    }
  };

  const handleFineTune = async () => {
    try {
      await startFineTuning();
      const status = await getFineTuneStatus();
      setFineTuningStatus(status);
    } catch (err) {
      console.error('Fine-tune start failed:', err);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error || !data) {
    return (
      <Container sx={{ mt: 4 }}>
        <Alert severity="error">{error || 'Unknown error occurred'}</Alert>
      </Container>
    );
  }

  const current = data.recent[data.recent.length - 1];
  const nextForecast = data.forecast[0];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static" color="inherit" elevation={1}>
        <Toolbar>
          <Sun size={24} color="#fbbc04" style={{ marginRight: '12px' }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
            Weather AI Dashboard
          </Typography>
          <Stack direction="row" spacing={2}>
            <Button
              variant="outlined"
              startIcon={<RefreshCw size={18} className={refreshing ? 'animate-spin' : ''} />}
              onClick={handleRefresh}
              disabled={refreshing || fineTuningStatus?.status === 'running'}
            >
              Refresh Forecast
            </Button>
            <Button
              variant="contained"
              startIcon={<Zap size={18} />}
              onClick={handleFineTune}
              disabled={fineTuningStatus?.status === 'running'}
              color="secondary"
            >
              Fine-Tune Model
            </Button>
          </Stack>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        {fineTuningStatus && fineTuningStatus.status !== 'idle' && (
          <Box sx={{ mb: 4 }}>
            <Alert
              severity={fineTuningStatus.status === 'error' ? 'error' : fineTuningStatus.status === 'success' ? 'success' : 'info'}
              action={
                fineTuningStatus.status !== 'running' && (
                  <Button color="inherit" size="small" onClick={() => setFineTuningStatus(null)}>
                    Dismiss
                  </Button>
                )
              }
            >
              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                {fineTuningStatus.message}
              </Typography>
              {fineTuningStatus.status === 'running' && (
                <Box sx={{ width: '100%', mt: 1 }}>
                  <LinearProgress />
                  <Typography variant="caption">
                    Model is training in the background...
                  </Typography>
                </Box>
              )}
            </Alert>
          </Box>
        )}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold' }}>
            Current Conditions
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Last updated: {new Date(current.time).toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })}
          </Typography>
        </Box>

        <Grid container spacing={3}>
          {/* Metric Cards */}
          <Grid size={{ xs: 12, sm: 6, md: 4, lg: 2 }}>
            <MetricCard
              title="Temperature"
              value={current.temp_c}
              unit="°C"
              icon={<Thermometer size={20} />}
              color="#ea4335"
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 6, md: 4, lg: 2 }}>
            <MetricCard
              title="Humidity"
              value={current.humidity_pct}
              unit="%"
              icon={<Droplets size={20} />}
              color="#1a73e8"
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 6, md: 4, lg: 2 }}>
            <MetricCard
              title="Wind Speed"
              value={current.wind_speed}
              unit="km/h"
              icon={<Wind size={20} />}
              color="#5f6368"
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 6, md: 4, lg: 2 }}>
            <MetricCard
              title="Precipitation"
              value={current.rain_mm}
              unit="mm"
              icon={<CloudRain size={20} />}
              color="#4285f4"
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 6, md: 4, lg: 2 }}>
            <MetricCard
              title="Cloud Cover"
              value={current.cloud_pct}
              unit="%"
              icon={<Cloud size={20} />}
              color="#9aa0a6"
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 6, md: 4, lg: 2 }}>
            <MetricCard
              title="Solar Radiation"
              value={current.solar_wm2}
              unit="W/m²"
              icon={<Sun size={20} />}
              color="#fbbc04"
            />
          </Grid>
        </Grid>

        <Box sx={{ mt: 6, mb: 2 }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold' }}>
            24-Hour Forecast
          </Typography>
          <Stack
            direction="row"
            spacing={2}
            sx={{
              overflowX: 'auto',
              pb: 2,
              '&::-webkit-scrollbar': { height: 8 },
              '&::-webkit-scrollbar-thumb': { backgroundColor: '#cbd5e1', borderRadius: 4 }
            }}
          >
            {data.forecast.map((hourData, index) => (
              <Box
                key={index}
                onClick={() => setSelectedHour(selectedHour === index ? null : index)}
                sx={{
                  minWidth: 100,
                  p: 2,
                  bgcolor: 'background.paper',
                  borderRadius: 2,
                  boxShadow: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 1.5,
                  cursor: 'pointer',
                  border: selectedHour === index ? '2px solid #1a73e8' : '2px solid transparent',
                  transition: 'all 0.2s ease',
                  '&:hover': { transform: 'translateY(-2px)', boxShadow: 2 }
                }}
              >
                <Typography variant="subtitle2" color="text.secondary">
                  {new Date(hourData.time).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}
                </Typography>
                <Typography variant="h6" sx={{ fontWeight: 'bold', mb: 1 }}>
                  {Math.round(hourData.temp_c)}°C
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: '#4285f4' }} title="Precipitation">
                  <CloudRain size={16} />
                  <Typography variant="body2">{hourData.rain_mm}mm</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: '#5f6368' }} title="Wind Speed">
                  <Wind size={16} />
                  <Typography variant="body2">{Math.round(hourData.wind_speed)}km/h</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: '#9aa0a6' }} title="Cloud Cover">
                  <Cloud size={16} />
                  <Typography variant="body2">{Math.round(hourData.cloud_pct)}%</Typography>
                </Box>
              </Box>
            ))}
          </Stack>

          {/* Expanded Forecast Details */}
          {selectedHour !== null && (
            <Box sx={{ mt: 3, p: 3, bgcolor: 'background.paper', borderRadius: 2, boxShadow: 1 }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                Detailed Forecast for {new Date(data.forecast[selectedHour].time).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}
              </Typography>
              <Grid container spacing={2} sx={{ mt: 1, textAlign: 'center' }}>
                <Grid size={{ xs: 6, sm: 4 }}>
                  <Typography variant="body2" color="text.secondary">Temperature</Typography>
                  <Typography variant="body1" sx={{ fontWeight: 'bold' }}>{data.forecast[selectedHour].temp_c.toFixed(1)} °C</Typography>
                </Grid>
                <Grid size={{ xs: 6, sm: 4 }}>
                  <Typography variant="body2" color="text.secondary">Precipitation</Typography>
                  <Typography variant="body1" sx={{ fontWeight: 'bold' }}>{data.forecast[selectedHour].rain_mm.toFixed(1)} mm</Typography>
                </Grid>
                <Grid size={{ xs: 6, sm: 4 }}>
                  <Typography variant="body2" color="text.secondary">Wind</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5 }}>
                    <ArrowUp size={16} style={{ transform: `rotate(${data.forecast[selectedHour].wind_deg + 180}deg)` }} />
                    <Typography variant="body1" sx={{ fontWeight: 'bold' }}>{data.forecast[selectedHour].wind_speed.toFixed(1)} km/h</Typography>
                  </Box>
                </Grid>
                <Grid size={{ xs: 6, sm: 4 }}>
                  <Typography variant="body2" color="text.secondary">Humidity</Typography>
                  <Typography variant="body1" sx={{ fontWeight: 'bold' }}>{Math.round(data.forecast[selectedHour].humidity_pct)}%</Typography>
                </Grid>
                <Grid size={{ xs: 6, sm: 4 }}>
                  <Typography variant="body2" color="text.secondary">Cloud Cover</Typography>
                  <Typography variant="body1" sx={{ fontWeight: 'bold' }}>{Math.round(data.forecast[selectedHour].cloud_pct)}%</Typography>
                </Grid>
                <Grid size={{ xs: 6, sm: 4 }}>
                  <Typography variant="body2" color="text.secondary">Solar Radiation</Typography>
                  <Typography variant="body1" sx={{ fontWeight: 'bold' }}>{data.forecast[selectedHour].solar_wm2.toFixed(1)} W/m²</Typography>
                </Grid>
              </Grid>
            </Box>
          )}
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
