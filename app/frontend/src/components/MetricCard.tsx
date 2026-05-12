import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit: string;
  icon: React.ReactNode;
  color?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, unit, icon, color = 'primary.main' }) => {
  return (
    <Card sx={{ height: '100%', borderRadius: 3, boxShadow: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box sx={{ color, mr: 1, display: 'flex' }}>
            {icon}
          </Box>
          <Typography variant="subtitle2" color="text.secondary" sx={{ fontWeight: 'bold' }}>
            {title}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'baseline' }}>
          <Typography variant="h4" component="span" sx={{ fontWeight: 'medium' }}>
            {value}
          </Typography>
          <Typography variant="body1" component="span" color="text.secondary" sx={{ ml: 0.5 }}>
            {unit}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default MetricCard;
