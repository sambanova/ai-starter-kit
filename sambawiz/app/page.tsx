import { Typography, Box } from '@mui/material';
import AppLayout from './components/AppLayout';
import BundleForm from './components/BundleForm';

export default function Home() {
  return (
    <AppLayout>
      <Box>
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600, mb: 1 }}>
          SambaWiz
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
          Create and configure model bundles with PEF configurations
        </Typography>

        <BundleForm />
      </Box>
    </AppLayout>
  );
}
