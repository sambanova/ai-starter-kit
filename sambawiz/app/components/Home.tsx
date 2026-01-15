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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from '@mui/material';
import { Visibility, VisibilityOff, ContentCopy } from '@mui/icons-material';
import AppConfigDialog from './AppConfigDialog';
import NoKubeconfigsDialog from './NoKubeconfigsDialog';
import DocumentationPanel from './DocumentationPanel';

interface KubeconfigEntry {
  file: string;
  namespace: string;
  uiDomain?: string;
  apiDomain?: string;
  apiKey?: string;
}

export default function Home() {
  const router = useRouter();
  const [selectedEnvironment, setSelectedEnvironment] = useState<string>('');
  const [namespace, setNamespace] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [apiDomain, setApiDomain] = useState<string>('');
  const [uiDomain, setUiDomain] = useState<string>('');
  const [showApiKey, setShowApiKey] = useState<boolean>(false);
  const [environments, setEnvironments] = useState<string[]>([]);
  const [kubeconfigs, setKubeconfigs] = useState<Record<string, KubeconfigEntry>>({});
  const [saving, setSaving] = useState<boolean>(false);
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [prerequisiteWarning, setPrerequisiteWarning] = useState<string | null>(null);
  const [showPrerequisiteDialog, setShowPrerequisiteDialog] = useState<boolean>(false);
  const [showAppConfigDialog, setShowAppConfigDialog] = useState<boolean>(false);
  const [showNoKubeconfigsDialog, setShowNoKubeconfigsDialog] = useState<boolean>(false);
  const [showApiKeyInstructionsDialog, setShowApiKeyInstructionsDialog] = useState<boolean>(false);
  const [keycloakUsername, setKeycloakUsername] = useState<string>('');
  const [keycloakPassword, setKeycloakPassword] = useState<string>('');
  const [showPassword, setShowPassword] = useState<boolean>(false);
  const [loadingCredentials, setLoadingCredentials] = useState<boolean>(false);
  const [credentialsError, setCredentialsError] = useState<string | null>(null);

  // Check prerequisites on component mount
  useEffect(() => {
    const checkPrerequisites = async () => {
      try {
        // Check kubectl and helm
        const prereqResponse = await fetch('/api/check-prerequisites');
        const prereqData = await prereqResponse.json();

        if (prereqData.success) {
          const missing = [];
          if (!prereqData.prerequisites.kubectl) {
            missing.push('kubectl');
          }
          if (!prereqData.prerequisites.helm) {
            missing.push('helm');
          }

          if (missing.length > 0) {
            setPrerequisiteWarning(
              `The following required tools are not installed: ${missing.join(', ')}. Please install them on your system.`
            );
            setShowPrerequisiteDialog(true);
            return;
          }
        }

        // Check helm version after successful prerequisite check
        const helmVersionResponse = await fetch('/api/kubeconfig-validate');
        const helmVersionData = await helmVersionResponse.json();

        if (!helmVersionData.success && helmVersionData.helmVersionError) {
          setPrerequisiteWarning(helmVersionData.errorDetails || 'Helm version check failed');
          setShowPrerequisiteDialog(true);
          return;
        }

        // Check app-config.json
        const configResponse = await fetch('/api/check-app-config');
        const configData = await configResponse.json();

        if (configData.success && (!configData.exists || !configData.valid)) {
          setShowAppConfigDialog(true);
          return;
        }

        // If config exists and is valid, check if we need to auto-populate kubeconfigs
        if (configData.success && configData.exists && configData.valid) {
          const config = configData.config;
          const kubeconfigsEmpty = Object.keys(config.kubeconfigs || {}).length === 0;
          const currentKubeconfigEmpty = !config.currentKubeconfig || config.currentKubeconfig.trim() === '';

          if (kubeconfigsEmpty && currentKubeconfigEmpty) {
            // Check if there are any kubeconfig files in the kubeconfigs directory
            const kubeconfigFilesResponse = await fetch('/api/check-kubeconfig-files');
            const kubeconfigFilesData = await kubeconfigFilesResponse.json();

            if (kubeconfigFilesData.success && !kubeconfigFilesData.hasFiles) {
              // No kubeconfig files found
              setShowNoKubeconfigsDialog(true);
              return;
            }

            // Try to auto-populate if files exist
            if (kubeconfigFilesData.success && kubeconfigFilesData.hasFiles) {
              try {
                const autoPopResponse = await fetch('/api/auto-populate-kubeconfigs', {
                  method: 'POST',
                });
                const autoPopData = await autoPopResponse.json();

                if (autoPopData.success) {
                  // Refresh the page to load the new config
                  window.location.reload();
                }
              } catch (error) {
                console.error('Failed to auto-populate kubeconfigs:', error);
              }
            }
          }
        }
      } catch (error) {
        console.error('Failed to check prerequisites:', error);
      }
    };

    checkPrerequisites();
  }, []);

  // Fetch available kubeconfig files on component mount
  useEffect(() => {
    const fetchEnvironments = async () => {
      try {
        const response = await fetch('/api/environments');
        const data = await response.json();

        if (data.success) {
          setEnvironments(data.environments);
          setKubeconfigs(data.kubeconfigs || {});
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
          // Set default API domain from app-config.json
          if (data.defaultApiDomain) {
            setApiDomain(data.defaultApiDomain);
          }
          // Set default UI domain from app-config.json
          if (data.defaultUiDomain) {
            setUiDomain(data.defaultUiDomain);
          }
        }
      } catch (error) {
        console.error('Failed to fetch environments:', error);
      }
    };

    fetchEnvironments();
  }, []);

  const handleEnvironmentChange = (event: SelectChangeEvent<string>) => {
    const envName = event.target.value;
    setSelectedEnvironment(envName);
    setSaveSuccess(false);
    setSaveError(null);

    // Auto-populate namespace, API key, and domains from kubeconfigs
    if (envName && kubeconfigs[envName]) {
      setNamespace(kubeconfigs[envName].namespace || 'default');
      setApiKey(kubeconfigs[envName].apiKey || '');
      setApiDomain(kubeconfigs[envName].apiDomain || '');
      setUiDomain(kubeconfigs[envName].uiDomain || '');
    } else {
      setNamespace('default');
      setApiKey('');
      setApiDomain('');
      setUiDomain('');
    }
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

  const handleApiDomainChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setApiDomain(event.target.value);
    setSaveSuccess(false);
    setSaveError(null);
  };

  const handleUiDomainChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setUiDomain(event.target.value);
    setSaveSuccess(false);
    setSaveError(null);
  };

  const handleToggleShowApiKey = () => {
    setShowApiKey(!showApiKey);
  };

  const handleAddEnvironment = () => {
    router.push('/add-environment');
  };

  const handleGetApiKey = async () => {
    setShowApiKeyInstructionsDialog(true);
    setLoadingCredentials(true);
    setCredentialsError(null);
    setKeycloakUsername('');
    setKeycloakPassword('');
    setShowPassword(false);

    if (!selectedEnvironment) {
      setCredentialsError('Please select an environment first');
      setLoadingCredentials(false);
      return;
    }

    try {
      const response = await fetch('/api/get-keycloak-credentials', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          environment: selectedEnvironment,
        }),
      });

      const data = await response.json();

      if (data.success) {
        setKeycloakUsername(data.username);
        setKeycloakPassword(data.password);
      } else {
        setCredentialsError(data.error || 'Failed to retrieve credentials');
      }
    } catch (error) {
      console.error('Error fetching credentials:', error);
      setCredentialsError('Failed to retrieve credentials');
    } finally {
      setLoadingCredentials(false);
    }
  };

  const handleCopyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
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
          apiDomain: apiDomain,
          uiDomain: uiDomain,
        }),
      });

      const data = await response.json();

      if (data.success) {
        setSaveSuccess(true);
        // Reload the page after a short delay to refresh the navbar and all app state
        setTimeout(() => {
          window.location.reload();
        }, 1000);
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

  const handleAppConfigCreated = async () => {
    // Refresh the page to load the new config
    window.location.reload();
  };

  return (
    <>
      {/* Documentation Panel */}
      <DocumentationPanel docFile="home.md" />

      {/* Prerequisite Warning Dialog */}
      <Dialog
        open={showPrerequisiteDialog}
        onClose={() => setShowPrerequisiteDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Prerequisites Missing</DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            {prerequisiteWarning}
          </Alert>
          <DialogContentText>
            Please install the required tools and restart the application.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowPrerequisiteDialog(false)} autoFocus>
            OK
          </Button>
        </DialogActions>
      </Dialog>

      {/* App Config Dialog */}
      <AppConfigDialog
        open={showAppConfigDialog}
        onClose={() => setShowAppConfigDialog(false)}
        onConfigCreated={handleAppConfigCreated}
      />

      {/* No Kubeconfigs Dialog */}
      <NoKubeconfigsDialog
        open={showNoKubeconfigsDialog}
        onClose={() => setShowNoKubeconfigsDialog(false)}
      />

      {/* API Key Instructions Dialog */}
      <Dialog
        open={showApiKeyInstructionsDialog}
        onClose={() => setShowApiKeyInstructionsDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>API Key Instructions</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 3 }}>
            Login to the following UI domain using the following credentials to create your API key
          </DialogContentText>

          {/* Loading State */}
          {loadingCredentials && (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
              <CircularProgress size={40} />
            </Box>
          )}

          {/* Error State */}
          {credentialsError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {credentialsError}
            </Alert>
          )}

          {/* UI Domain */}
          {!loadingCredentials && uiDomain && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                UI Domain:
              </Typography>
              <Typography
                component="a"
                href={uiDomain}
                target="_blank"
                rel="noopener noreferrer"
                sx={{
                  color: 'primary.main',
                  textDecoration: 'underline',
                  wordBreak: 'break-all',
                  '&:hover': {
                    color: 'primary.dark',
                  },
                }}
              >
                {uiDomain}
              </Typography>
            </Box>
          )}

          {/* Credentials */}
          {!loadingCredentials && keycloakUsername && keycloakPassword && (
            <Box>
              {/* Username */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                  Username:
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TextField
                    fullWidth
                    value={keycloakUsername}
                    variant="outlined"
                    size="small"
                    slotProps={{
                      input: {
                        readOnly: true,
                      },
                    }}
                  />
                  <IconButton
                    onClick={() => handleCopyToClipboard(keycloakUsername)}
                    size="small"
                    sx={{ color: 'primary.main' }}
                  >
                    <ContentCopy fontSize="small" />
                  </IconButton>
                </Box>
              </Box>

              {/* Password */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                  Password:
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TextField
                    fullWidth
                    type={showPassword ? 'text' : 'password'}
                    value={keycloakPassword}
                    variant="outlined"
                    size="small"
                    slotProps={{
                      input: {
                        readOnly: true,
                        endAdornment: (
                          <InputAdornment position="end">
                            <IconButton
                              onClick={() => setShowPassword(!showPassword)}
                              edge="end"
                              size="small"
                            >
                              {showPassword ? <VisibilityOff fontSize="small" /> : <Visibility fontSize="small" />}
                            </IconButton>
                          </InputAdornment>
                        ),
                      },
                    }}
                  />
                  <IconButton
                    onClick={() => handleCopyToClipboard(keycloakPassword)}
                    size="small"
                    sx={{ color: 'primary.main' }}
                  >
                    <ContentCopy fontSize="small" />
                  </IconButton>
                </Box>
              </Box>
            </Box>
          )}

          {!loadingCredentials && !uiDomain && (
            <Alert severity="warning">
              Please select an environment with a UI domain configured.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowApiKeyInstructionsDialog(false)} autoFocus>
            Close
          </Button>
        </DialogActions>
      </Dialog>

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
            background: 'linear-gradient(to right, #A2297D, #4E226B)',
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
          Your SambaStack Bundle Configuration Wizard
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
          label="API Domain"
          value={apiDomain}
          onChange={handleApiDomainChange}
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
          label="UI Domain"
          value={uiDomain}
          onChange={handleUiDomainChange}
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

        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 0.5 }}>
            <Typography
              component="a"
              onClick={handleGetApiKey}
              sx={{
                fontSize: '0.75rem',
                color: 'primary.main',
                cursor: 'pointer',
                textDecoration: 'underline',
                '&:hover': {
                  color: 'primary.dark',
                },
              }}
            >
              Get API Key
            </Typography>
          </Box>
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
        </Box>

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
            background: '#A2297D',
            '&:hover': {
              background: '#8B2268',
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
    </>
  );
}
