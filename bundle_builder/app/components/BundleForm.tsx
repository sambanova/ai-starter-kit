'use client';

import { useState, useMemo, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  ListItemText,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Divider,
  SelectChangeEvent,
  TextField,
  Button,
  Alert,
  CircularProgress,
} from '@mui/material';
import type { PefConfigs, PefMapping, CheckpointMapping, ConfigSelection } from '../types/bundle';
import { generateBundleYaml, generateBundleName } from '../lib/bundle-yaml-generator';

// Import the JSON data
import pefConfigsData from '../data/pef_configs.json';
import pefMappingData from '../data/pef_mapping.json';
import checkpointMappingData from '../data/checkpoint_mapping.json';

const pefConfigs: PefConfigs = pefConfigsData;
const pefMapping: PefMapping = pefMappingData;
const checkpointMapping: CheckpointMapping = checkpointMappingData;

// Get checkpoints directory from environment
const CHECKPOINTS_DIR = process.env.NEXT_PUBLIC_CHECKPOINTS_DIR || process.env.CHECKPOINTS_DIR || '';

interface ModelConfig {
  ss: string;
  bs: string;
}

export default function BundleForm() {
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedConfigs, setSelectedConfigs] = useState<ConfigSelection[]>([]);
  const [bundleName, setBundleName] = useState<string>('');
  const [generatedYaml, setGeneratedYaml] = useState<string>('');
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<{
    success: boolean;
    message: string;
    applyOutput?: string;
    statusConditions?: string;
    bundleName?: string;
  } | null>(null);

  // Get available models (intersection of checkpoint and pef mapping keys with non-empty values)
  const availableModels = useMemo(() => {
    const checkpointKeys = Object.keys(checkpointMapping).filter(
      (key) => checkpointMapping[key] !== ''
    );
    const pefMappingKeys = Object.keys(pefMapping).filter(
      (key) => pefMapping[key].length > 0
    );
    return checkpointKeys.filter((key) => pefMappingKeys.includes(key)).sort();
  }, []);

  // Get available configurations for selected models
  const modelConfigurations = useMemo(() => {
    const configs: { [modelName: string]: ModelConfig[] } = {};

    selectedModels.forEach((modelName) => {
      const pefs = pefMapping[modelName] || [];
      const configSet = new Set<string>();

      pefs.forEach((pefName) => {
        const config = pefConfigs[pefName];
        if (config) {
          configSet.add(`${config.ss}|${config.bs}`);
        }
      });

      configs[modelName] = Array.from(configSet)
        .map((key) => {
          const [ss, bs] = key.split('|');
          return { ss, bs };
        })
        .sort((a, b) => {
          // Sort by ss first, then by bs
          const ssA = parseInt(a.ss);
          const ssB = parseInt(b.ss);
          if (ssA !== ssB) return ssA - ssB;
          return parseInt(a.bs) - parseInt(b.bs);
        });
    });

    return configs;
  }, [selectedModels]);

  const handleModelChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    const models = typeof value === 'string' ? value.split(',') : value;
    setSelectedModels(models);

    // Remove configs for deselected models
    setSelectedConfigs((prev) =>
      prev.filter((config) => models.includes(config.modelName))
    );
  };

  const handleConfigToggle = (modelName: string, ss: string, bs: string) => {
    // Find the PEF name that matches this model and config
    const pefs = pefMapping[modelName] || [];
    const matchingPef = pefs.find((pefName) => {
      const config = pefConfigs[pefName];
      return config && config.ss === ss && config.bs === bs;
    });

    if (!matchingPef) return;

    const existingIndex = selectedConfigs.findIndex(
      (config) => config.modelName === modelName && config.ss === ss && config.bs === bs
    );

    if (existingIndex >= 0) {
      // Remove the config
      setSelectedConfigs((prev) => prev.filter((_, i) => i !== existingIndex));
    } else {
      // Add the config
      setSelectedConfigs((prev) => [
        ...prev,
        { modelName, ss, bs, pefName: matchingPef },
      ]);
    }
  };

  const isConfigSelected = (modelName: string, ss: string, bs: string): boolean => {
    return selectedConfigs.some(
      (config) => config.modelName === modelName && config.ss === ss && config.bs === bs
    );
  };

  const handleSelectAllConfigs = (modelName: string) => {
    const configs = modelConfigurations[modelName] || [];
    const allSelected = configs.every((config) =>
      isConfigSelected(modelName, config.ss, config.bs)
    );

    if (allSelected) {
      // Deselect all configs for this model
      setSelectedConfigs((prev) =>
        prev.filter((config) => config.modelName !== modelName)
      );
    } else {
      // Select all configs for this model
      const newConfigs = configs
        .filter((config) => !isConfigSelected(modelName, config.ss, config.bs))
        .map((config) => {
          const pefs = pefMapping[modelName] || [];
          const matchingPef = pefs.find((pefName) => {
            const pefConfig = pefConfigs[pefName];
            return pefConfig && pefConfig.ss === config.ss && pefConfig.bs === config.bs;
          });
          return matchingPef ? { modelName, ss: config.ss, bs: config.bs, pefName: matchingPef } : null;
        })
        .filter((config): config is ConfigSelection => config !== null);

      setSelectedConfigs((prev) => [...prev, ...newConfigs]);
    }
  };

  const areAllConfigsSelected = (modelName: string): boolean => {
    const configs = modelConfigurations[modelName] || [];
    if (configs.length === 0) return false;
    return configs.every((config) => isConfigSelected(modelName, config.ss, config.bs));
  };

  // Auto-generate bundle name when models change
  useMemo(() => {
    if (selectedModels.length > 0 && bundleName === '') {
      const generatedName = generateBundleName(selectedModels);
      setBundleName(generatedName);
    }
  }, [selectedModels, bundleName]);

  // Get selected PEFs grouped by model
  const selectedPefsByModel = useMemo(() => {
    const grouped: { [modelName: string]: string[] } = {};
    selectedConfigs.forEach((config) => {
      if (!grouped[config.modelName]) {
        grouped[config.modelName] = [];
      }
      grouped[config.modelName].push(config.pefName);
    });
    return grouped;
  }, [selectedConfigs]);

  // Generate YAML automatically when configs or bundle name change
  useEffect(() => {
    if (selectedConfigs.length === 0 || !bundleName) {
      setGeneratedYaml('');
      return;
    }
    const yaml = generateBundleYaml(selectedConfigs, checkpointMapping, pefConfigs, bundleName, CHECKPOINTS_DIR);
    setGeneratedYaml(yaml);
  }, [selectedConfigs, bundleName]);

  // Handle validation
  const handleValidate = async () => {
    if (!generatedYaml) return;

    setIsValidating(true);
    setValidationResult(null);

    try {
      const response = await fetch('/api/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ yaml: generatedYaml }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setValidationResult({
          success: true,
          message: 'Bundle validated and applied successfully!',
          applyOutput: data.applyOutput,
          statusConditions: data.statusConditions,
          bundleName: data.bundleName,
        });
      } else {
        setValidationResult({
          success: false,
          message: data.error || 'Validation failed',
          applyOutput: data.applyOutput || data.stderr || data.stdout || data.message,
        });
      }
    } catch (error: any) {
      setValidationResult({
        success: false,
        message: 'Failed to connect to validation service',
        applyOutput: error.message,
      });
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <Box>
      {/* Model Selection */}
      <Paper elevation={0} sx={{ p: 3, mb: 3, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
          1. Select Models
        </Typography>
        <FormControl fullWidth>
          <InputLabel id="model-select-label">Models</InputLabel>
          <Select
            labelId="model-select-label"
            id="model-select"
            multiple
            value={selectedModels}
            onChange={handleModelChange}
            label="Models"
            renderValue={(selected) => (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {selected.map((value) => (
                  <Chip key={value} label={value} size="small" />
                ))}
              </Box>
            )}
          >
            {availableModels.map((model) => (
              <MenuItem key={model} value={model}>
                <Checkbox checked={selectedModels.indexOf(model) > -1} />
                <ListItemText primary={model} />
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Paper>

      {/* Configuration Selection Tables */}
      {selectedModels.length > 0 && (
        <Paper elevation={0} sx={{ p: 3, mb: 3, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
            2. Select Configurations
          </Typography>
          {selectedModels.map((modelName, idx) => (
            <Box key={modelName} sx={{ mb: idx < selectedModels.length - 1 ? 3 : 0 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1.5 }}>
                {modelName}
              </Typography>
              <TableContainer>
                <Table size="small" sx={{ border: '1px solid', borderColor: 'divider' }}>
                  <TableHead>
                    <TableRow sx={{ bgcolor: 'grey.50' }}>
                      <TableCell sx={{ fontWeight: 600 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Checkbox
                            checked={areAllConfigsSelected(modelName)}
                            indeterminate={
                              !areAllConfigsSelected(modelName) &&
                              selectedConfigs.some((config) => config.modelName === modelName)
                            }
                            onChange={() => handleSelectAllConfigs(modelName)}
                          />
                          Select All
                        </Box>
                      </TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Sequence Length (SS)</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Batch Size (BS)</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {modelConfigurations[modelName]?.map((config) => (
                      <TableRow
                        key={`${config.ss}-${config.bs}`}
                        hover
                        sx={{
                          cursor: 'pointer',
                          '&:hover': { bgcolor: 'action.hover' },
                        }}
                        onClick={() => handleConfigToggle(modelName, config.ss, config.bs)}
                      >
                        <TableCell onClick={(e) => e.stopPropagation()}>
                          <Checkbox
                            checked={isConfigSelected(modelName, config.ss, config.bs)}
                            onChange={() => handleConfigToggle(modelName, config.ss, config.bs)}
                          />
                        </TableCell>
                        <TableCell>{config.ss}</TableCell>
                        <TableCell>{config.bs}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              {idx < selectedModels.length - 1 && <Divider sx={{ mt: 3 }} />}
            </Box>
          ))}
        </Paper>
      )}

      {/* Selected PEFs Display */}
      {selectedConfigs.length > 0 && (
        <Paper elevation={0} sx={{ p: 3, mb: 3, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
            3. Selected PEFs
          </Typography>
          {Object.entries(selectedPefsByModel).map(([modelName, pefs]) => (
            <Box key={modelName} sx={{ mb: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1, color: 'text.secondary' }}>
                {modelName}
              </Typography>
              <Box component="ul" sx={{ pl: 3, mt: 0, mb: 1 }}>
                {pefs.map((pef) => {
                  const version = pefConfigs[pef]?.latestVersion || '1';
                  return (
                    <Typography component="li" key={pef} variant="body2" sx={{ mb: 0.5 }}>
                      {pef}:{version}
                    </Typography>
                  );
                })}
              </Box>
            </Box>
          ))}
        </Paper>
      )}

      {/* Checkpoints Display */}
      {selectedModels.length > 0 && (
        <Paper elevation={0} sx={{ p: 3, mb: 3, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
            4. Model Checkpoints
          </Typography>
          {selectedModels.map((modelName) => (
            <Box key={modelName} sx={{ mb: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                {modelName}
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  color: 'text.secondary',
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                  wordBreak: 'break-all',
                }}
              >
                {checkpointMapping[modelName] ? `${CHECKPOINTS_DIR}${checkpointMapping[modelName]}` : 'No checkpoint available'}
              </Typography>
            </Box>
          ))}
        </Paper>
      )}

      {/* Bundle YAML Generation */}
      {selectedConfigs.length > 0 && (
        <Paper elevation={0} sx={{ p: 3, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
            5. Bundle YAML
          </Typography>

          {/* Bundle Name Input */}
          <Box sx={{ mb: 2 }}>
            <TextField
              fullWidth
              label="Bundle Name"
              value={bundleName}
              onChange={(e) => setBundleName(e.target.value)}
              helperText="Edit the bundle name (used for bt-* and b-* resources)"
              variant="outlined"
              size="small"
            />
          </Box>

          {/* Generated YAML */}
          <Box>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
              Generated YAML
            </Typography>
            <TextField
              fullWidth
              multiline
              rows={25}
              value={generatedYaml}
              onChange={(e) => setGeneratedYaml(e.target.value)}
              variant="outlined"
              sx={{
                '& .MuiInputBase-root': {
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                },
              }}
            />
          </Box>

          {/* Validation Result */}
          {validationResult && (
            <Box sx={{ mt: 2 }}>
              <Alert severity={validationResult.success ? 'success' : 'error'}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                  {validationResult.message}
                </Typography>

                {/* Apply Output */}
                {validationResult.applyOutput && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
                      kubectl apply output:
                    </Typography>
                    <Box
                      component="pre"
                      sx={{
                        p: 1,
                        bgcolor: 'rgba(0, 0, 0, 0.05)',
                        borderRadius: 1,
                        fontSize: '0.75rem',
                        overflow: 'auto',
                        maxHeight: '150px',
                      }}
                    >
                      {validationResult.applyOutput}
                    </Box>
                  </Box>
                )}

                {/* Status Conditions */}
                {validationResult.statusConditions && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
                      Bundle Status (kubectl describe bundle {validationResult.bundleName}):
                    </Typography>
                    <Box
                      component="pre"
                      sx={{
                        p: 1,
                        bgcolor: 'rgba(0, 0, 0, 0.05)',
                        borderRadius: 1,
                        fontSize: '0.75rem',
                        overflow: 'auto',
                        maxHeight: '300px',
                        whiteSpace: 'pre-wrap',
                        wordWrap: 'break-word',
                      }}
                    >
                      {validationResult.statusConditions}
                    </Box>
                  </Box>
                )}
              </Alert>
            </Box>
          )}

          {/* Validate Button */}
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="contained"
              color="primary"
              size="large"
              onClick={handleValidate}
              disabled={isValidating || !generatedYaml}
              startIcon={isValidating ? <CircularProgress size={20} /> : null}
            >
              {isValidating ? 'Validating...' : 'Validate'}
            </Button>
          </Box>
        </Paper>
      )}
    </Box>
  );
}
