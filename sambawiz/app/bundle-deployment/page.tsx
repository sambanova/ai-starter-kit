import { Typography, Box } from '@mui/material';
import AppLayout from '../components/AppLayout';
import BundleDeploymentManager from '../components/BundleDeploymentManager';

export default function BundleDeploymentPage() {
  return (
    <AppLayout>
      <Box>
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600, mb: 1 }}>
          Bundle Deployment
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
          Manage and monitor your bundle deployments
        </Typography>

        <BundleDeploymentManager />
      </Box>
    </AppLayout>
  );
}
