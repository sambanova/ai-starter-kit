'use client';

import { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  SelectChangeEvent,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import SaveIcon from '@mui/icons-material/Save';

interface BundleDeployment {
  name: string;
  namespace: string;
  bundle: string;
  creationTimestamp: string;
  status?: {
    conditions?: Array<{
      type: string;
      status: string;
      reason: string;
      message: string;
    }>;
  };
}

interface Bundle {
  name: string;
  namespace: string;
  template: string;
  creationTimestamp: string;
  isValid: boolean;
  validationReason: string;
  validationMessage: string;
  models: { [key: string]: any };
}

export default function BundleDeploymentManager() {
  const [bundleDeployments, setBundleDeployments] = useState<BundleDeployment[]>([]);
  const [deploymentToDelete, setDeploymentToDelete] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState<boolean>(false);
  const [deleting, setDeleting] = useState<boolean>(false);

  // Section 2: Deploy a Bundle
  const [bundles, setBundles] = useState<Bundle[]>([]);
  const [validBundles, setValidBundles] = useState<Bundle[]>([]);
  const [selectedBundle, setSelectedBundle] = useState<string>('');
  const [deploymentName, setDeploymentName] = useState<string>('');
  const [loadingBundles, setLoadingBundles] = useState<boolean>(false);
  const [deploymentYaml, setDeploymentYaml] = useState<string>('');
  const [copiedYaml, setCopiedYaml] = useState<boolean>(false);
  const [deploying, setDeploying] = useState<boolean>(false);
  const [deploymentResult, setDeploymentResult] = useState<{
    success: boolean;
    message: string;
    output?: string;
  } | null>(null);

  // Section 3: Check Deployment Status
  const [podLogs, setPodLogs] = useState<string>('');
  const [podLogsError, setPodLogsError] = useState<string | null>(null);
  const [monitoredDeployment, setMonitoredDeployment] = useState<string>('');

  // Save functionality
  const [saveDialogOpen, setSaveDialogOpen] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveResult, setSaveResult] = useState<{ success: boolean; message: string } | null>(null);

  // Fetch bundle deployments
  const fetchBundleDeployments = async () => {
    setLoading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const response = await fetch('/api/bundle-deployment');
      const data = await response.json();

      if (data.success) {
        setBundleDeployments(data.bundleDeployments);
      } else {
        setError(data.error || 'Failed to fetch bundle deployments');
      }
    } catch (err: any) {
      setError('Failed to connect to the server');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Load bundle deployments on mount
  useEffect(() => {
    fetchBundleDeployments();
    fetchBundles();
  }, []);

  // Auto-refresh pod logs every 3 seconds
  useEffect(() => {
    if (!monitoredDeployment) {
      setPodLogs('');
      setPodLogsError(null);
      return;
    }

    const fetchPodLogs = async () => {
      const podName = `inf-${monitoredDeployment}-cache-0`;

      try {
        const response = await fetch(`/api/pod-logs?podName=${podName}&lines=5`);
        const data = await response.json();

        if (data.success) {
          setPodLogs(data.logs);
          setPodLogsError(null);
        } else {
          setPodLogsError(data.message || 'Failed to fetch logs');
        }
      } catch (err: any) {
        setPodLogsError('Failed to connect to the server');
      }
    };

    // Fetch immediately
    fetchPodLogs();

    // Set up interval to fetch every 3 seconds
    const intervalId = setInterval(fetchPodLogs, 3000);

    // Cleanup interval on unmount or when monitoredDeployment changes
    return () => clearInterval(intervalId);
  }, [monitoredDeployment]);

  // Fetch bundles
  const fetchBundles = async () => {
    setLoadingBundles(true);

    try {
      const response = await fetch('/api/bundles');
      const data = await response.json();

      if (data.success) {
        setBundles(data.bundles);
        // Filter to only valid bundles
        const valid = data.bundles.filter((b: Bundle) => b.isValid);
        setValidBundles(valid);
      } else {
        console.error('Failed to fetch bundles:', data.error);
      }
    } catch (err: any) {
      console.error('Failed to connect to the server', err);
    } finally {
      setLoadingBundles(false);
    }
  };

  // Generate BundleDeployment YAML
  const generateDeploymentYaml = (bundleName: string, deploymentName: string): string => {
    return `apiVersion: sambanova.ai/v1alpha1
kind: BundleDeployment
metadata:
  name: ${deploymentName}
spec:
  bundle: ${bundleName}
  groups:
  - minReplicas: 1
    name: default
    qosList:
    - free
  owner: no-reply@sambanova.ai
  secretNames:
  - sambanova-artifact-reader`;
  };

  // Handle bundle selection
  const handleBundleChange = (event: SelectChangeEvent<string>) => {
    const bundleName = event.target.value;
    setSelectedBundle(bundleName);

    // Auto-suggest deployment name
    let suggestedName = '';
    if (bundleName) {
      if (bundleName.startsWith('b-')) {
        suggestedName = bundleName.replace('b-', 'bd-');
      } else {
        suggestedName = `bd-${bundleName}`;
      }
      setDeploymentName(suggestedName);

      // Generate YAML
      const yaml = generateDeploymentYaml(bundleName, suggestedName);
      setDeploymentYaml(yaml);
    } else {
      setDeploymentName('');
      setDeploymentYaml('');
    }
  };

  // Handle deployment name change
  const handleDeploymentNameChange = (newName: string) => {
    setDeploymentName(newName);

    // Regenerate YAML with new deployment name
    if (selectedBundle && newName) {
      const yaml = generateDeploymentYaml(selectedBundle, newName);
      setDeploymentYaml(yaml);
    }
  };

  // Handle copy YAML to clipboard
  const handleCopyYaml = async () => {
    try {
      await navigator.clipboard.writeText(deploymentYaml);
      setCopiedYaml(true);
      setTimeout(() => setCopiedYaml(false), 2000);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  // Handle deploy
  const handleDeploy = async () => {
    if (!deploymentYaml) return;

    setDeploying(true);
    setDeploymentResult(null);

    try {
      const response = await fetch('/api/deploy-bundle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yaml: deploymentYaml }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setDeploymentResult({
          success: true,
          message: 'Bundle deployment applied successfully!',
          output: data.output,
        });
        // Set the monitored deployment to start log monitoring
        setMonitoredDeployment(deploymentName);
        // Refresh the bundle deployments list
        await fetchBundleDeployments();
      } else {
        setDeploymentResult({
          success: false,
          message: data.error || 'Deployment failed',
          output: data.stderr || data.stdout || data.message || '',
        });
      }
    } catch (err: any) {
      setDeploymentResult({
        success: false,
        message: 'Failed to connect to deployment service',
        output: err.message,
      });
    } finally {
      setDeploying(false);
    }
  };

  // Open delete confirmation dialog for a specific deployment
  const handleDeleteClick = (name: string) => {
    setDeploymentToDelete(name);
    setDeleteDialogOpen(true);
  };

  // Close delete confirmation dialog
  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setDeploymentToDelete(null);
  };

  // Confirm deletion
  const handleDeleteConfirm = async () => {
    if (!deploymentToDelete) return;

    setDeleteDialogOpen(false);
    setDeleting(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const response = await fetch('/api/bundle-deployment', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: deploymentToDelete }),
      });

      const data = await response.json();

      if (data.success) {
        setSuccessMessage(`Successfully deleted ${deploymentToDelete}`);
        // If the deleted deployment is being monitored, clear it
        if (monitoredDeployment === deploymentToDelete) {
          setMonitoredDeployment('');
        }
      } else {
        setError(`Failed to delete ${deploymentToDelete}: ${data.error}`);
      }

      // Refresh the list
      await fetchBundleDeployments();
    } catch (err: any) {
      setError('Failed to delete bundle deployment');
      console.error(err);
    } finally {
      setDeleting(false);
      setDeploymentToDelete(null);
    }
  };

  // Handle status button click
  const handleStatusClick = (name: string) => {
    setMonitoredDeployment(name);
  };

  // Get validation status display
  const getStatusDisplay = (deployment: BundleDeployment) => {
    if (!deployment.status?.conditions || deployment.status.conditions.length === 0) {
      return { text: 'Unknown', color: 'text.secondary' };
    }

    const condition = deployment.status.conditions[0];
    if (condition.reason === 'ValidationSucceeded' || condition.status === 'True') {
      return { text: 'Valid', color: 'success.main' };
    } else {
      return { text: 'Invalid', color: 'error.main' };
    }
  };

  // Handle save button click
  const handleSaveClick = () => {
    setSaveResult(null);
    setSaveDialogOpen(false);
    handleSaveFile(false);
  };

  // Handle save file
  const handleSaveFile = async (overwrite: boolean) => {
    if (!deploymentYaml || !deploymentName) return;

    setIsSaving(true);
    setSaveResult(null);

    const fileName = `${deploymentName}.yaml`;

    try {
      const endpoint = '/api/save-artifact';
      const method = overwrite ? 'PUT' : 'POST';

      const response = await fetch(endpoint, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fileName, content: deploymentYaml }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setSaveResult({
          success: true,
          message: `Bundle deployment saved successfully to saved_artifacts/${fileName}`,
        });
      } else if (response.status === 409 && data.fileExists) {
        // File exists, show overwrite dialog
        setSaveDialogOpen(true);
      } else {
        setSaveResult({
          success: false,
          message: data.error || 'Failed to save bundle deployment',
        });
      }
    } catch (error: any) {
      setSaveResult({
        success: false,
        message: 'Failed to connect to save service',
      });
    } finally {
      setIsSaving(false);
    }
  };

  // Handle overwrite confirmation
  const handleOverwrite = () => {
    setSaveDialogOpen(false);
    handleSaveFile(true);
  };

  // Handle cancel save
  const handleCancelSave = () => {
    setSaveDialogOpen(false);
    setSaveResult(null);
  };

  return (
    <Box>
      {/* Section 1: Check for existing Bundle Deployments */}
      <Paper elevation={0} sx={{ p: 3, mb: 3, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            1. Check for existing Bundle Deployments
          </Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchBundleDeployments}
            disabled={loading || deleting}
          >
            Refresh
          </Button>
        </Box>

        {/* Success/Error Messages */}
        {successMessage && (
          <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccessMessage(null)}>
            {successMessage}
          </Alert>
        )}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Loading State */}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {/* Empty State */}
        {!loading && bundleDeployments.length === 0 && (
          <Alert severity="info">
            No bundle deployments found in the namespace
          </Alert>
        )}

        {/* Bundle Deployments Table */}
        {!loading && bundleDeployments.length > 0 && (
          <TableContainer>
            <Table size="small" sx={{ border: '1px solid', borderColor: 'divider' }}>
              <TableHead>
                <TableRow sx={{ bgcolor: 'grey.50' }}>
                  <TableCell sx={{ fontWeight: 600 }}>Name</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Bundle</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Status</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Created</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {bundleDeployments.map((deployment) => {
                  const status = getStatusDisplay(deployment);
                  return (
                    <TableRow key={deployment.name} hover>
                      <TableCell>{deployment.name}</TableCell>
                      <TableCell>{deployment.bundle}</TableCell>
                      <TableCell>
                        <Typography sx={{ color: status.color, fontWeight: 500 }}>
                          {status.text}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {new Date(deployment.creationTimestamp).toLocaleString()}
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Button
                            variant="outlined"
                            size="small"
                            onClick={() => handleStatusClick(deployment.name)}
                          >
                            Status
                          </Button>
                          <Button
                            variant="outlined"
                            color="error"
                            size="small"
                            onClick={() => handleDeleteClick(deployment.name)}
                            disabled={deleting}
                          >
                            Delete
                          </Button>
                        </Box>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Paper>

      {/* Section 2: Deploy a Bundle */}
      <Paper elevation={0} sx={{ p: 3, mb: 3, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
          2. Deploy a Bundle
        </Typography>

        {/* Loading State */}
        {loadingBundles && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            <CircularProgress />
          </Box>
        )}

        {/* Empty State */}
        {!loadingBundles && validBundles.length === 0 && (
          <Alert severity="info">
            No valid bundles found. Please create and validate a bundle first.
          </Alert>
        )}

        {/* Bundle Selection Form */}
        {!loadingBundles && validBundles.length > 0 && (
          <Box>
            <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
              Select a valid bundle to deploy
            </Typography>

            {/* Bundle Dropdown */}
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel id="bundle-select-label">Bundle</InputLabel>
              <Select
                labelId="bundle-select-label"
                id="bundle-select"
                value={selectedBundle}
                onChange={handleBundleChange}
                label="Bundle"
              >
                {validBundles.map((bundle) => (
                  <MenuItem key={bundle.name} value={bundle.name}>
                    {bundle.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Deployment Name */}
            {selectedBundle && (
              <Box>
                <TextField
                  fullWidth
                  label="Deployment Name"
                  value={deploymentName}
                  onChange={(e) => handleDeploymentNameChange(e.target.value)}
                  helperText="Enter the name for this bundle deployment (e.g., bd-your-bundle-name)"
                  variant="outlined"
                  sx={{ mb: 3 }}
                />

                {/* Generated YAML */}
                <Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      Generated YAML
                    </Typography>
                    <Button
                      startIcon={<ContentCopyIcon />}
                      onClick={handleCopyYaml}
                      size="small"
                      disabled={!deploymentYaml}
                      sx={{
                        color: copiedYaml ? 'success.main' : 'primary.main',
                      }}
                    >
                      {copiedYaml ? 'Copied!' : 'Copy'}
                    </Button>
                  </Box>
                  <TextField
                    fullWidth
                    multiline
                    rows={15}
                    value={deploymentYaml}
                    onChange={(e) => setDeploymentYaml(e.target.value)}
                    variant="outlined"
                    sx={{
                      '& .MuiInputBase-root': {
                        fontFamily: 'monospace',
                        fontSize: '0.875rem',
                      },
                    }}
                  />
                </Box>

                {/* Deployment Result */}
                {deploymentResult && (
                  <Box sx={{ mt: 2 }}>
                    <Alert
                      severity={deploymentResult.success ? 'success' : 'error'}
                      onClose={() => setDeploymentResult(null)}
                    >
                      <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: deploymentResult.output ? 1 : 0 }}>
                        {deploymentResult.message}
                      </Typography>
                      {deploymentResult.output && (
                        <Box
                          component="pre"
                          sx={{
                            mt: 1,
                            p: 1.5,
                            bgcolor: 'rgba(0, 0, 0, 0.05)',
                            borderRadius: 1,
                            fontSize: '0.75rem',
                            overflow: 'auto',
                            maxHeight: '150px',
                          }}
                        >
                          {deploymentResult.output}
                        </Box>
                      )}
                    </Alert>
                  </Box>
                )}

                {/* Save Result */}
                {saveResult && (
                  <Box sx={{ mt: 2 }}>
                    <Alert
                      severity={saveResult.success ? 'success' : 'error'}
                      onClose={() => setSaveResult(null)}
                    >
                      {saveResult.message}
                    </Alert>
                  </Box>
                )}

                {/* Save and Deploy Buttons */}
                <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                  <Button
                    variant="outlined"
                    color="primary"
                    size="large"
                    onClick={handleSaveClick}
                    disabled={isSaving || !deploymentYaml || !deploymentName}
                    startIcon={isSaving ? <CircularProgress size={20} /> : <SaveIcon />}
                  >
                    {isSaving ? 'Saving...' : 'Save'}
                  </Button>
                  <Button
                    variant="contained"
                    color="primary"
                    size="large"
                    startIcon={deploying ? <CircularProgress size={20} color="inherit" /> : <RocketLaunchIcon />}
                    onClick={handleDeploy}
                    disabled={deploying || !deploymentYaml}
                  >
                    {deploying ? 'Deploying...' : 'Deploy'}
                  </Button>
                </Box>
              </Box>
            )}
          </Box>
        )}
      </Paper>

      {/* Section 3: Check Deployment Status */}
      {monitoredDeployment && (
        <Paper elevation={0} sx={{ p: 3, mb: 3, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
            3. Check Deployment Status
          </Typography>

          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Last 5 lines of Cache Pod
          </Typography>

          <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary', fontFamily: 'monospace', fontSize: '0.875rem' }}>
            Monitoring: inf-{monitoredDeployment}-cache-0
          </Typography>

          {podLogsError ? (
            <Alert severity="error" sx={{ mb: 2 }}>
              {podLogsError}
            </Alert>
          ) : (
            <Box
              component="pre"
              sx={{
                p: 2,
                bgcolor: 'black',
                color: 'white',
                borderRadius: 1,
                fontSize: '0.875rem',
                fontFamily: 'monospace',
                overflow: 'auto',
                minHeight: '120px',
                whiteSpace: 'pre-wrap',
                wordWrap: 'break-word',
              }}
            >
              {podLogs || 'Waiting for logs...'}
            </Box>
          )}

          <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
            Auto-refreshing every 3 seconds
          </Typography>
        </Paper>
      )}

      {/* Save Overwrite Confirmation Dialog */}
      <Dialog open={saveDialogOpen} onClose={handleCancelSave}>
        <DialogTitle>File Already Exists</DialogTitle>
        <DialogContent>
          <DialogContentText>
            A file named <strong>{deploymentName}.yaml</strong> already exists in saved_artifacts.
            Do you want to overwrite it?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelSave} color="primary">
            Cancel
          </Button>
          <Button onClick={handleOverwrite} color="primary" variant="contained">
            Overwrite
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={handleDeleteCancel}
      >
        <DialogTitle>Confirm Deletion</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the bundle deployment: <strong>{deploymentToDelete}</strong>?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel} color="primary">
            Cancel
          </Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
