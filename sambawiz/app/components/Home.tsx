'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Box,
  Typography,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  SelectChangeEvent,
  Button,
  Alert,
  CircularProgress,
  InputAdornment,
  IconButton,
} from '@mui/material';
import { Visibility, VisibilityOff } from '@mui/icons-material';

export default function Home() {
  const router = useRouter();
  const [selectedEnvironment, setSelectedEnvironment] = useState<string>('');
  const [namespace, setNamespace] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [showApiKey, setShowApiKey] = useState<boolean>(false);
  const [environments, setEnvironments] = useState<string[]>([]);
  const [saving, setSaving] = useState<boolean>(false);
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Fetch available kubeconfig files on component mount
  useEffect(() => {
    const fetchEnvironments = async () => {
      try {
        const response = await fetch('/api/environments');
        const data = await response.json();

        if (data.success) {
          setEnvironments(data.environments);
          // Set default environment from app-config.json
          if (data.defaultEnvironment) {
            setSelectedEnvironment(data.defaultEnvironment);
          }
          // Set default namespace from app-config.json
          if (data.defaultNamespace) {
            setNamespace(data.defaultNamespace);
          }
          // Set default API key from app-config.json
          if (data.defaultApiKey) {
            setApiKey(data.defaultApiKey);
          }
        }
      } catch (error) {
        console.error('Failed to fetch environments:', error);
      }
    };

    fetchEnvironments();
  }, []);

  const handleEnvironmentChange = (event: SelectChangeEvent<string>) => {
    setSelectedEnvironment(event.target.value);
    setSaveSuccess(false);
    setSaveError(null);
  };

  const handleNamespaceChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setNamespace(event.target.value);
    setSaveSuccess(false);
    setSaveError(null);
  };

  const handleApiKeyChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setApiKey(event.target.value);
    setSaveSuccess(false);
    setSaveError(null);
  };

  const handleToggleShowApiKey = () => {
    setShowApiKey(!showApiKey);
  };

  const handleAddEnvironment = () => {
    router.push('/add-environment');
  };

  const handleApply = async () => {
    if (!selectedEnvironment || !namespace) {
      setSaveError('Please select an environment and enter a namespace');
      return;
    }

    setSaving(true);
    setSaveSuccess(false);
    setSaveError(null);

    try {
      const response = await fetch('/api/update-config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          environment: selectedEnvironment,
          namespace: namespace,
          apiKey: apiKey,
        }),
      });

      const data = await response.json();

      if (data.success) {
        setSaveSuccess(true);
        // Clear success message after 3 seconds
        setTimeout(() => setSaveSuccess(false), 3000);
      } else {
        setSaveError(data.error || 'Failed to save configuration');
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
      setSaveError('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: 'calc(100vh - 100px)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        py: 4,
      }}
    >
      {/* Hero Section */}
      <Box
        sx={{
          textAlign: 'center',
          mb: 8,
          maxWidth: '800px',
        }}
      >
        <Typography
          variant="h2"
          component="h1"
          sx={{
            fontWeight: 700,
            mb: 2,
            fontSize: { xs: '2.5rem', md: '3.5rem' },
            background: 'linear-gradient(135deg, #FF6B35 0%, #FF8E53 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          SambaWiz
        </Typography>
        <Typography
          variant="h5"
          sx={{
            color: 'text.secondary',
            mb: 3,
            fontWeight: 400,
            fontSize: { xs: '1.1rem', md: '1.5rem' },
          }}
        >
          Your AI-Powered Bundle Configuration Wizard
        </Typography>
        <Typography
          variant="body1"
          sx={{
            color: 'text.secondary',
            maxWidth: '600px',
            mx: 'auto',
            lineHeight: 1.7,
          }}
        >
          Create, configure, and deploy model bundles with ease.
        </Typography>
      </Box>

      {/* Environment Selection Section */}
      <Paper
        elevation={3}
        sx={{
          p: 4,
          maxWidth: '600px',
          width: '100%',
          borderRadius: 3,
          background: 'linear-gradient(135deg, rgba(255, 107, 53, 0.05) 0%, rgba(255, 142, 83, 0.05) 100%)',
          border: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 3,
          }}
        >
          <Typography
            variant="h6"
            sx={{
              fontWeight: 600,
              color: 'primary.main',
            }}
          >
            Select your SambaStack environment
          </Typography>
          <Button
            variant="outlined"
            size="small"
            onClick={handleAddEnvironment}
            sx={{
              textTransform: 'none',
              fontWeight: 600,
              borderColor: 'primary.main',
              color: 'primary.main',
              '&:hover': {
                borderColor: 'primary.dark',
                backgroundColor: 'rgba(255, 107, 53, 0.05)',
              },
            }}
          >
            Add an Environment
          </Button>
        </Box>

        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel id="environment-select-label">Environment</InputLabel>
          <Select
            labelId="environment-select-label"
            id="environment-select"
            value={selectedEnvironment}
            label="Environment"
            onChange={handleEnvironmentChange}
            sx={{
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: 'divider',
              },
              '&:hover .MuiOutlinedInput-notchedOutline': {
                borderColor: 'primary.main',
              },
            }}
          >
            {environments.length === 0 ? (
              <MenuItem disabled value="">
                No environments available
              </MenuItem>
            ) : (
              environments.map((env) => (
                <MenuItem key={env} value={env}>
                  {env}
                </MenuItem>
              ))
            )}
          </Select>
        </FormControl>

        <TextField
          fullWidth
          label="Namespace"
          value={namespace}
          onChange={handleNamespaceChange}
          variant="outlined"
          sx={{
            mb: 3,
            '& .MuiOutlinedInput-root': {
              '& fieldset': {
                borderColor: 'divider',
              },
              '&:hover fieldset': {
                borderColor: 'primary.main',
              },
            },
          }}
        />

        <TextField
          fullWidth
          label="API Key"
          type={showApiKey ? 'text' : 'password'}
          value={apiKey}
          onChange={handleApiKeyChange}
          variant="outlined"
          slotProps={{
            input: {
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    aria-label="toggle api key visibility"
                    onClick={handleToggleShowApiKey}
                    edge="end"
                  >
                    {showApiKey ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            },
            inputLabel: {
              shrink: true,
            },
          }}
          sx={{
            mb: 3,
            '& .MuiOutlinedInput-root': {
              '& fieldset': {
                borderColor: 'divider',
              },
              '&:hover fieldset': {
                borderColor: 'primary.main',
              },
            },
          }}
        />

        {/* Success/Error Messages */}
        {saveSuccess && (
          <Alert severity="success" sx={{ mb: 3 }}>
            Configuration saved successfully!
          </Alert>
        )}
        {saveError && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {saveError}
          </Alert>
        )}

        {/* Apply Button */}
        <Button
          fullWidth
          variant="contained"
          size="large"
          onClick={handleApply}
          disabled={saving || !selectedEnvironment || !namespace}
          sx={{
            py: 1.5,
            fontWeight: 600,
            fontSize: '1rem',
            textTransform: 'none',
            background: 'linear-gradient(135deg, #FF6B35 0%, #FF8E53 100%)',
            '&:hover': {
              background: 'linear-gradient(135deg, #FF5722 0%, #FF7043 100%)',
            },
            '&:disabled': {
              background: '#ccc',
              color: '#666',
            },
          }}
        >
          {saving ? (
            <>
              <CircularProgress size={20} sx={{ mr: 1, color: 'white' }} />
              Applying...
            </>
          ) : (
            'Apply Configuration'
          )}
        </Button>
      </Paper>

      {/* Footer Info */}
      <Box
        sx={{
          mt: 6,
          textAlign: 'center',
          color: 'text.secondary',
        }}
      >
        <Typography variant="body2" sx={{ fontSize: '0.875rem' }}>
          Ready to build? Use the navigation menu to access the bundle builder and deployment tools.
        </Typography>
      </Box>
    </Box>
  );
}
