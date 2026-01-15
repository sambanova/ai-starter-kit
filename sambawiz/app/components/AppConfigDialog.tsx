'use client';

import { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
} from '@mui/material';

interface AppConfigDialogProps {
  open: boolean;
  onClose: () => void;
  onConfigCreated: () => void;
}

export default function AppConfigDialog({ open, onClose, onConfigCreated }: AppConfigDialogProps) {
  const [checkpointsDir, setCheckpointsDir] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCreate = async () => {
    if (!checkpointsDir || checkpointsDir.trim() === '') {
      setError('Checkpoints Directory is required');
      return;
    }

    setCreating(true);
    setError(null);

    try {
      const response = await fetch('/api/check-app-config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          checkpointsDir: checkpointsDir.trim(),
        }),
      });

      const data = await response.json();

      if (data.success) {
        onConfigCreated();
        onClose();
      } else {
        setError(data.error || 'Failed to create app-config.json');
      }
    } catch (err) {
      console.error('Error creating app-config.json:', err);
      setError('Failed to create app-config.json');
    } finally {
      setCreating(false);
    }
  };

  const handleClose = () => {
    if (!creating) {
      onClose();
    }
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>App Configuration Required</DialogTitle>
      <DialogContent>
        <Alert severity="warning" sx={{ mb: 3 }}>
          The app-config.json file does not exist or the checkpointsDir field is not populated.
          Please create the configuration file to continue.
        </Alert>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          The app-config.json file must exist in the SambaWiz root directory and contain a valid checkpoints directory path.
        </Typography>
        <TextField
          fullWidth
          label="Checkpoints Dir"
          placeholder="gs://your-bucket-name/path/to/checkpoints/"
          value={checkpointsDir}
          onChange={(e) => setCheckpointsDir(e.target.value)}
          variant="outlined"
          helperText="Enter the GCS checkpoint directory path (must end with /)"
          sx={{ mb: 2 }}
        />
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} disabled={creating}>
          Cancel
        </Button>
        <Button
          onClick={handleCreate}
          variant="contained"
          disabled={creating || !checkpointsDir.trim()}
          sx={{
            background: 'linear-gradient(135deg, #FF6B35 0%, #FF8E53 100%)',
            '&:hover': {
              background: 'linear-gradient(135deg, #FF5722 0%, #FF7043 100%)',
            },
          }}
        >
          {creating ? (
            <>
              <CircularProgress size={20} sx={{ mr: 1, color: 'white' }} />
              Creating...
            </>
          ) : (
            'Create App Config'
          )}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
