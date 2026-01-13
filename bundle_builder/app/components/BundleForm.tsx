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
  Tooltip,
  IconButton,
} from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DeleteIcon from '@mui/icons-material/Delete';
import HandymanIcon from '@mui/icons-material/Handyman';
import type { PefConfigs, PefMapping, CheckpointMapping, ConfigSelection } from '../types/bundle';
import { generateBundleYaml } from '../lib/bundle-yaml-generator';

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
  const [bundleName, setBundleName] = useState<string>('bundle1');
  const [generatedYaml, setGeneratedYaml] = useState<string>('');
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [draftModels, setDraftModels] = useState<{ [modelName: string]: string }>({});
  const [copiedToClipboard, setCopiedToClipboard] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<{
    success: boolean;
    message: string;
    applyOutput?: string;
    validationStatus?: {
      reason: string;
      message: string;
      isValid: boolean;
    };
    bundleName?: string;
  } | null>(null);

  // Get available models (intersection of checkpoint and pef mapping keys with non-empty values)
  const availableModels = useMemo(() => {
    const checkpointKeys = Object.keys(checkpointMapping).filter(
      (key) => checkpointMapping[key]?.path !== ''
    );
    const pefMappingKeys = Object.keys(pefMapping).filter(
      (key) => pefMapping[key].length > 0
    );
    return checkpointKeys.filter((key) => pefMappingKeys.includes(key)).sort();
  }, []);

  // Check if a model supports speculative decoding
  const modelSupportsSpeculativeDecoding = useMemo(() => {
    const sdSupport: { [modelName: string]: boolean } = {};

    selectedModels.forEach((modelName) => {
      const pefs = pefMapping[modelName] || [];
      // A model supports SD only if ALL its PEF configs contain "-sd" followed by a number
      const allConfigsSupportSD = pefs.length > 0 && pefs.every((pefName) => {
        // Check if the config name contains "-sd" followed by digits in a hyphenated section
        const parts = pefName.split('-');
        return parts.some((part) => /^sd\d+$/.test(part));
      });
      sdSupport[modelName] = allConfigsSupportSD;
    });

    return sdSupport;
  }, [selectedModels]);

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

    // Remove draft model selections for deselected models
    setDraftModels((prev) => {
      const updated = { ...prev };
      Object.keys(updated).forEach((modelName) => {
        if (!models.includes(modelName)) {
          delete updated[modelName];
        }
      });
      return updated;
    });
  };

  const handleDraftModelChange = (targetModel: string, draftModel: string) => {
    setDraftModels((prev) => ({
      ...prev,
      [targetModel]: draftModel,
    }));

    // If a draft model is selected (not "Skip"), add it to selectedModels if not already present
    if (draftModel !== 'skip' && !selectedModels.includes(draftModel)) {
      setSelectedModels((prev) => [...prev, draftModel]);
    }

    // If a draft model is selected (not "Skip") and the target model has configs selected,
    // automatically select matching configs for the draft model
    if (draftModel !== 'skip') {
      const targetConfigs = selectedConfigs.filter((config) => config.modelName === targetModel);

      if (targetConfigs.length > 0) {
        const draftPefs = pefMapping[draftModel] || [];
        const newDraftConfigs: ConfigSelection[] = [];

        targetConfigs.forEach((targetConfig) => {
          // Check if draft model already has this config selected
          const draftConfigExists = selectedConfigs.some(
            (config) => config.modelName === draftModel &&
                       config.ss === targetConfig.ss &&
                       config.bs === targetConfig.bs
          );

          if (!draftConfigExists) {
            // Find matching draft model PEF for this config
            const matchingDraftPef = draftPefs.find((pefName) => {
              const config = pefConfigs[pefName];
              return config && config.ss === targetConfig.ss && config.bs === targetConfig.bs;
            });

            if (matchingDraftPef) {
              newDraftConfigs.push({
                modelName: draftModel,
                ss: targetConfig.ss,
                bs: targetConfig.bs,
                pefName: matchingDraftPef,
              });
            }
          }
        });

        if (newDraftConfigs.length > 0) {
          setSelectedConfigs((prev) => [...prev, ...newDraftConfigs]);
        }
      }
    }
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
      const newConfigs: ConfigSelection[] = [
        { modelName, ss, bs, pefName: matchingPef },
      ];

      // If this model has a draft model selected, try to auto-select the same config for the draft model
      const draftModel = draftModels[modelName];
      if (draftModel && draftModel !== 'skip') {
        const draftPefs = pefMapping[draftModel] || [];
        const matchingDraftPef = draftPefs.find((pefName) => {
          const config = pefConfigs[pefName];
          return config && config.ss === ss && config.bs === bs;
        });

        // If the draft model has this config and it's not already selected, add it
        if (matchingDraftPef) {
          const draftConfigExists = selectedConfigs.some(
            (config) => config.modelName === draftModel && config.ss === ss && config.bs === bs
          );
          if (!draftConfigExists) {
            newConfigs.push({ modelName: draftModel, ss, bs, pefName: matchingDraftPef });
          }
        }
      }

      // Check if this model is used as a draft model by any other model
      // If so, also select the same config for that target model if it exists
      Object.entries(draftModels).forEach(([targetModel, targetDraftModel]) => {
        if (targetDraftModel === modelName) {
          // This model is used as a draft model for targetModel
          const targetPefs = pefMapping[targetModel] || [];
          const matchingTargetPef = targetPefs.find((pefName) => {
            const config = pefConfigs[pefName];
            return config && config.ss === ss && config.bs === bs;
          });

          if (matchingTargetPef) {
            const targetConfigExists = selectedConfigs.some(
              (config) => config.modelName === targetModel && config.ss === ss && config.bs === bs
            );
            // Also check if we're about to add it in this operation
            const aboutToAddTargetConfig = newConfigs.some(
              (config) => config.modelName === targetModel && config.ss === ss && config.bs === bs
            );
            if (!targetConfigExists && !aboutToAddTargetConfig) {
              newConfigs.push({ modelName: targetModel, ss, bs, pefName: matchingTargetPef });
            }
          }
        }
      });

      setSelectedConfigs((prev) => [...prev, ...newConfigs]);
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

  // Bundle name is manually set to 'bundle1' by default
  // Auto-generation disabled per user request

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

  // Check if a target model PEF has a corresponding draft model PEF selected
  const pefHasDraftModelConfig = (modelName: string, pefName: string): boolean => {
    const draftModel = draftModels[modelName];
    if (!draftModel || draftModel === 'skip') return true; // No draft model required

    const targetConfig = pefConfigs[pefName];
    if (!targetConfig) return true;

    // Check if a draft model config with matching SS/BS is currently SELECTED
    const hasDraftConfigSelected = selectedConfigs.some((config) => {
      return config.modelName === draftModel &&
             config.ss === targetConfig.ss &&
             config.bs === targetConfig.bs;
    });

    return hasDraftConfigSelected;
  };

  // Generate YAML automatically when configs or bundle name change
  useEffect(() => {
    if (selectedConfigs.length === 0 || !bundleName) {
      setGeneratedYaml('');
      return;
    }
    const yaml = generateBundleYaml(selectedConfigs, checkpointMapping, pefConfigs, bundleName, CHECKPOINTS_DIR, draftModels);
    setGeneratedYaml(yaml);
  }, [selectedConfigs, bundleName, draftModels]);

  // Handle copy to clipboard
  const handleCopyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(generatedYaml);
      setCopiedToClipboard(true);
      setTimeout(() => setCopiedToClipboard(false), 2000); // Reset after 2 seconds
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  // Handle removing a PEF configuration
  const handleRemovePefConfig = (modelName: string, pefName: string) => {
    const pefConfig = pefConfigs[pefName];
    if (!pefConfig) return;

    setSelectedConfigs((prev) =>
      prev.filter(
        (config) =>
          !(config.modelName === modelName && config.ss === pefConfig.ss && config.bs === pefConfig.bs)
      )
    );
  };

  // Handle adding missing draft model configuration
  const handleAddMissingDraftConfig = (modelName: string, pefName: string) => {
    const draftModel = draftModels[modelName];
    if (!draftModel || draftModel === 'skip') return;

    const targetConfig = pefConfigs[pefName];
    if (!targetConfig) return;

    const draftPefs = pefMapping[draftModel] || [];
    const matchingDraftPef = draftPefs.find((draftPefName) => {
      const config = pefConfigs[draftPefName];
      return config && config.ss === targetConfig.ss && config.bs === targetConfig.bs;
    });

    if (matchingDraftPef) {
      const draftConfigExists = selectedConfigs.some(
        (config) =>
          config.modelName === draftModel &&
          config.ss === targetConfig.ss &&
          config.bs === targetConfig.bs
      );

      if (!draftConfigExists) {
        setSelectedConfigs((prev) => [
          ...prev,
          {
            modelName: draftModel,
            ss: targetConfig.ss,
            bs: targetConfig.bs,
            pefName: matchingDraftPef,
          },
        ]);
      }
    }
  };

  // Check if a draft model config exists but is not selected
  const draftConfigExistsButNotSelected = (modelName: string, pefName: string): boolean => {
    const draftModel = draftModels[modelName];
    if (!draftModel || draftModel === 'skip') return false;

    const targetConfig = pefConfigs[pefName];
    if (!targetConfig) return false;

    const draftPefs = pefMapping[draftModel] || [];
    const matchingDraftPef = draftPefs.find((draftPefName) => {
      const config = pefConfigs[draftPefName];
      return config && config.ss === targetConfig.ss && config.bs === targetConfig.bs;
    });

    if (!matchingDraftPef) return false;

    // Check if the draft config is NOT selected
    const draftConfigSelected = selectedConfigs.some(
      (config) =>
        config.modelName === draftModel &&
        config.ss === targetConfig.ss &&
        config.bs === targetConfig.bs
    );

    return !draftConfigSelected;
  };

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
          validationStatus: data.validationStatus,
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
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 0.5 }}>
                {modelName}
              </Typography>
              {modelSupportsSpeculativeDecoding[modelName] && (
                <Box sx={{ mb: 1.5 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      This model supports speculative decoding. Enable it by choosing a draft model:
                    </Typography>
                    <FormControl size="small" sx={{ minWidth: 200 }}>
                      <Select
                        value={draftModels[modelName] || 'skip'}
                        onChange={(e) => handleDraftModelChange(modelName, e.target.value)}
                        displayEmpty
                      >
                        <MenuItem value="skip">skip</MenuItem>
                        {availableModels
                          .filter((model) => model !== modelName)
                          .map((model) => (
                            <MenuItem key={model} value={model}>
                              {model}
                            </MenuItem>
                          ))}
                      </Select>
                    </FormControl>
                  </Box>
                </Box>
              )}
              {!modelSupportsSpeculativeDecoding[modelName] && (
                <Box sx={{ mb: 1.5 }} />
              )}
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
                  const hasDraftConfig = pefHasDraftModelConfig(modelName, pef);
                  const canFixDraftConfig = draftConfigExistsButNotSelected(modelName, pef);
                  return (
                    <Box component="li" key={pef} sx={{ mb: 0.5, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <Typography variant="body2">
                        {pef}:{version}
                      </Typography>
                      {!hasDraftConfig && (
                        <>
                          <Tooltip title="No draft model assigned to this PEF for speculative decoding">
                            <WarningAmberIcon sx={{ fontSize: 16, color: 'warning.main' }} />
                          </Tooltip>
                          <Tooltip title="Click here to remove this config">
                            <IconButton
                              size="small"
                              onClick={() => handleRemovePefConfig(modelName, pef)}
                              sx={{ p: 0.25 }}
                            >
                              <DeleteIcon sx={{ fontSize: 16, color: 'error.main' }} />
                            </IconButton>
                          </Tooltip>
                          {canFixDraftConfig && (
                            <Tooltip title="Click here to add the missing draft model configuration">
                              <IconButton
                                size="small"
                                onClick={() => handleAddMissingDraftConfig(modelName, pef)}
                                sx={{ p: 0.25 }}
                              >
                                <HandymanIcon sx={{ fontSize: 16, color: 'primary.main' }} />
                              </IconButton>
                            </Tooltip>
                          )}
                        </>
                      )}
                    </Box>
                  );
                })}
              </Box>
            </Box>
          ))}
        </Paper>
      )}

      {/* Bundle YAML Generation */}
      {selectedConfigs.length > 0 && (
        <Paper elevation={0} sx={{ p: 3, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
            4. Bundle YAML
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
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                Generated YAML
              </Typography>
              <Tooltip title={copiedToClipboard ? "Copied!" : "Copy to clipboard"}>
                <IconButton
                  onClick={handleCopyToClipboard}
                  size="small"
                  disabled={!generatedYaml}
                  sx={{
                    color: copiedToClipboard ? 'success.main' : 'primary.main',
                  }}
                >
                  <ContentCopyIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
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
              {/* Apply Output */}
              {validationResult.applyOutput && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
                    kubectl apply output:
                  </Typography>
                  <Box
                    component="pre"
                    sx={{
                      p: 1.5,
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

              {/* Validation Status */}
              {validationResult.validationStatus && (
                <Box
                  sx={{
                    p: 2,
                    bgcolor: validationResult.validationStatus.isValid
                      ? 'success.light'
                      : 'error.dark',
                    color: validationResult.validationStatus.isValid
                      ? 'success.contrastText'
                      : 'white',
                    borderRadius: 1,
                  }}
                >
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                    {validationResult.validationStatus.isValid
                      ? 'Validation succeeded!'
                      : 'Validation failed with the following errors:'}
                  </Typography>
                  {!validationResult.validationStatus.isValid && (
                    <Box
                      component="pre"
                      sx={{
                        mt: 1,
                        p: 1.5,
                        bgcolor: 'black',
                        color: 'white',
                        borderRadius: 1,
                        fontSize: '0.75rem',
                        overflow: 'auto',
                        maxHeight: '300px',
                        whiteSpace: 'pre-wrap',
                        wordWrap: 'break-word',
                      }}
                    >
                      {validationResult.validationStatus.message}
                    </Box>
                  )}
                </Box>
              )}

              {/* Fallback for errors without validation status */}
              {!validationResult.validationStatus && !validationResult.success && (
                <Alert severity="error">
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                    {validationResult.message}
                  </Typography>
                </Alert>
              )}
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
