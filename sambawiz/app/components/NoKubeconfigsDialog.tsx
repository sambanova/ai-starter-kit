'use client';

import { useRouter } from 'next/navigation';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Alert,
  List,
  ListItem,
  ListItemText,
  Box,
} from '@mui/material';

interface NoKubeconfigsDialogProps {
  open: boolean;
  onClose: () => void;
}

export default function NoKubeconfigsDialog({ open, onClose }: NoKubeconfigsDialogProps) {
  const router = useRouter();

  const handleAddEnvironment = () => {
    router.push('/add-environment');
  };

  const handleRefresh = () => {
    window.location.reload();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>No kubeconfigs found!</DialogTitle>
      <DialogContent>
        <Alert severity="warning" sx={{ mb: 3 }}>
          No kubeconfig files were found in the kubeconfigs directory.
        </Alert>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          To use SambaWiz, you need to provide a valid kubeconfig file. You have two options:
        </Typography>
        <List sx={{ pl: 2 }}>
          <ListItem sx={{ display: 'list-item', listStyleType: 'decimal', pl: 0 }}>
            <ListItemText
              primary="Manually copy a kubeconfig file"
              secondary="Copy your kubeconfig YAML file into the kubeconfigs directory and refresh this page."
              primaryTypographyProps={{ fontWeight: 600 }}
            />
          </ListItem>
          <ListItem sx={{ display: 'list-item', listStyleType: 'decimal', pl: 0 }}>
            <ListItemText
              primary="Add an environment using an encoded config"
              secondary="Use the 'Add Environment' page to paste an encoded kubeconfig (e.g., from 1Password)."
              primaryTypographyProps={{ fontWeight: 600 }}
            />
          </ListItem>
        </List>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 3, pt: 1 }}>
        <Button onClick={handleRefresh} variant="outlined">
          Refresh Page
        </Button>
        <Button
          onClick={handleAddEnvironment}
          variant="contained"
          sx={{
            background: 'linear-gradient(135deg, #FF6B35 0%, #FF8E53 100%)',
            '&:hover': {
              background: 'linear-gradient(135deg, #FF5722 0%, #FF7043 100%)',
            },
          }}
        >
          Add Environment
        </Button>
      </DialogActions>
    </Dialog>
  );
}
