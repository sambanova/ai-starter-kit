'use client';

import { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  CircularProgress,
  Alert,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import CodeIcon from '@mui/icons-material/Code';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import CleaningServicesIcon from '@mui/icons-material/CleaningServices';
import { getBundleDeploymentStatus } from './BundleDeploymentManager';
import ViewCodeDialog from './ViewCodeDialog';
import DocumentationPanel from './DocumentationPanel';

interface BundleDeployment {
  name: string;
  namespace: string;
  bundle: string;
  creationTimestamp: string;
}

interface PodStatusInfo {
  ready: number;
  total: number;
  status: string;
}

interface Metrics {
  tokensPerSecond: number | null;
  totalLatency: number | null;
  timeToFirstToken: number | null;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metrics?: Metrics;
}

export default function Playground() {
  const [bundleDeployments, setBundleDeployments] = useState<BundleDeployment[]>([]);
  const [selectedDeployment, setSelectedDeployment] = useState<string>('');
  const [deploymentStatuses, setDeploymentStatuses] = useState<Record<string, {
    cachePod: PodStatusInfo | null;
    defaultPod: PodStatusInfo | null;
  }>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Model selection state
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loadingModels, setLoadingModels] = useState<boolean>(false);
  const [modelsError, setModelsError] = useState<string | null>(null);

  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState<string>('');
  const [isSending, setIsSending] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // View Code dialog state
  const [viewCodeDialogOpen, setViewCodeDialogOpen] = useState<boolean>(false);
  const [apiKey, setApiKey] = useState<string>('');
  const [apiDomain, setApiDomain] = useState<string>('');
  const [currentEnvironment, setCurrentEnvironment] = useState<string>('');

  // Fetch bundle deployments and their statuses
  const fetchBundleDeployments = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/bundle-deployment');
      const data = await response.json();

      if (data.success) {
        setBundleDeployments(data.bundleDeployments);

        // Fetch pod statuses for all deployments
        const statuses: Record<string, {
          cachePod: PodStatusInfo | null;
          defaultPod: PodStatusInfo | null;
        }> = {};

        await Promise.all(
          data.bundleDeployments.map(async (deployment: BundleDeployment) => {
            try {
              const statusResponse = await fetch(`/api/pod-status?deploymentName=${deployment.name}`);
              const statusData = await statusResponse.json();

              if (statusData.success) {
                statuses[deployment.name] = statusData.podStatus;
              } else {
                statuses[deployment.name] = { cachePod: null, defaultPod: null };
              }
            } catch (err) {
              statuses[deployment.name] = { cachePod: null, defaultPod: null };
            }
          })
        );

        setDeploymentStatuses(statuses);
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

  useEffect(() => {
    fetchBundleDeployments();
    fetchEnvironmentConfig();
  }, []);

  // Fetch environment configuration
  const fetchEnvironmentConfig = async () => {
    try {
      const response = await fetch('/api/environments');
      const data = await response.json();

      if (data.success) {
        setCurrentEnvironment(data.defaultEnvironment || '');
        setApiKey(data.defaultApiKey || '');
        setApiDomain(data.defaultApiDomain || '');
      }
    } catch (err) {
      console.error('Error fetching environment config:', err);
    }
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Fetch models for a deployment
  const fetchModelsForDeployment = async (deploymentName: string) => {
    setLoadingModels(true);
    setModelsError(null);
    setAvailableModels([]);
    setSelectedModel('');

    try {
      const response = await fetch(`/api/deployment-models?deploymentName=${deploymentName}`);
      const data = await response.json();

      if (data.success && data.models) {
        setAvailableModels(data.models);
        // Auto-select first model if available
        if (data.models.length > 0) {
          setSelectedModel(data.models[0]);
        }
      } else {
        setModelsError(data.error || 'Failed to fetch models');
      }
    } catch (err: any) {
      console.error('Error fetching models:', err);
      setModelsError('Failed to connect to the server');
    } finally {
      setLoadingModels(false);
    }
  };

  // Handle deployment selection
  const handleDeploymentChange = (event: SelectChangeEvent<string>) => {
    const newDeployment = event.target.value;
    setSelectedDeployment(newDeployment);
    // Clear chat history when switching deployments
    setMessages([]);
    // Clear previous models
    setAvailableModels([]);
    setSelectedModel('');
    setModelsError(null);

    // Fetch models for the new deployment ONLY if a deployment is selected
    if (newDeployment) {
      // Small delay to ensure the dropdown has rendered
      setTimeout(() => {
        fetchModelsForDeployment(newDeployment);
      }, 100);
    }
  };

  // Handle model selection
  const handleModelChange = (event: SelectChangeEvent<string>) => {
    setSelectedModel(event.target.value);
    // Optionally clear chat history when switching models
    setMessages([]);
  };

  // Handle clear chat
  const handleClearChat = () => {
    setMessages([]);
  };

  // Get only deployed bundles
  const deployedBundles = bundleDeployments.filter((deployment) => {
    const podStatusInfo = deploymentStatuses[deployment.name];
    if (!podStatusInfo) return false;
    const status = getBundleDeploymentStatus(
      podStatusInfo.cachePod,
      podStatusInfo.defaultPod
    );
    return status === 'Deployed';
  });

  // Handle send message
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !selectedDeployment || !selectedModel) {
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage,
      timestamp: new Date(),
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInputMessage('');
    setIsSending(true);

    try {
      // Build conversation history for API
      const conversationHistory = [
        {
          role: 'system',
          content: 'You are a helpful assistant',
        },
        ...updatedMessages.map((msg) => ({
          role: msg.role,
          content: msg.content,
        })),
      ];

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: conversationHistory,
          model: selectedModel,
        }),
      });

      const data = await response.json();

      if (data.success) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.content,
          timestamp: new Date(),
          metrics: data.metrics || undefined,
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        // Show error message as an assistant response
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `Error: ${data.error}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      }
    } catch (err: any) {
      console.error('Error sending message:', err);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: Failed to send message - ${err.message}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsSending(false);
    }
  };

  // Handle Enter key press
  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Box>
      {/* Documentation Panel */}
      <DocumentationPanel docFile="playground.md" />

      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600, mb: 1 }}>
        Playground
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Chat with your deployed models
      </Typography>

      {/* Main Playground Container */}
      <Paper
        elevation={0}
        sx={{
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 2,
          overflow: 'hidden',
          height: 'calc(100vh - 250px)',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Header with Bundle and Model Selectors */}
        <Box
          sx={{
            p: 2,
            borderBottom: '1px solid',
            borderColor: 'divider',
            backgroundColor: 'grey.50',
            display: 'flex',
            alignItems: 'center',
            gap: 2,
            flexWrap: 'wrap',
          }}
        >
          <FormControl sx={{ minWidth: 300 }} size="small">
            <InputLabel id="deployment-select-label">Select Deployed Bundle</InputLabel>
            <Select
              labelId="deployment-select-label"
              id="deployment-select"
              value={selectedDeployment}
              onChange={handleDeploymentChange}
              label="Select Deployed Bundle"
              disabled={loading || deployedBundles.length === 0}
              sx={{ backgroundColor: 'white' }}
            >
              {deployedBundles.map((deployment) => (
                <MenuItem key={deployment.name} value={deployment.name}>
                  {deployment.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {selectedDeployment && (
            <>
              <FormControl sx={{ minWidth: 250 }} size="small">
                <InputLabel id="model-select-label">Select Model</InputLabel>
                <Select
                  labelId="model-select-label"
                  id="model-select"
                  value={selectedModel}
                  onChange={handleModelChange}
                  label="Select Model"
                  disabled={loadingModels || availableModels.length === 0}
                  sx={{ backgroundColor: 'white' }}
                >
                  {availableModels.map((model) => (
                    <MenuItem key={model} value={model}>
                      {model}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {selectedModel && (
                <>
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<CodeIcon />}
                    onClick={() => setViewCodeDialogOpen(true)}
                    sx={{
                      backgroundColor: 'white',
                      textTransform: 'none',
                    }}
                  >
                    View Code
                  </Button>
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<CleaningServicesIcon />}
                    onClick={handleClearChat}
                    disabled={messages.length === 0}
                    sx={{
                      backgroundColor: 'white',
                      textTransform: 'none',
                    }}
                  >
                    Clear Chat
                  </Button>
                </>
              )}
            </>
          )}

          {loading && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={20} />
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                Loading deployments...
              </Typography>
            </Box>
          )}

          {loadingModels && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={20} />
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                Loading models...
              </Typography>
            </Box>
          )}
        </Box>

        {/* Error State */}
        {error && (
          <Box sx={{ p: 2 }}>
            <Alert severity="error">{error}</Alert>
          </Box>
        )}

        {/* Models Error State */}
        {modelsError && selectedDeployment && (
          <Box sx={{ p: 2 }}>
            <Alert severity="warning">
              Failed to load models: {modelsError}
            </Alert>
          </Box>
        )}

        {/* No Deployed Bundles State */}
        {!loading && deployedBundles.length === 0 && !error && (
          <Box sx={{ p: 3 }}>
            <Alert severity="info">
              No deployed bundles found. Please deploy a bundle first to use the playground.
            </Alert>
          </Box>
        )}

        {/* Chat Interface - Only show when deployment and model are selected */}
        {selectedDeployment && selectedModel && (
          <>
            {/* Messages Container */}
            <Box
              sx={{
                flex: 1,
                overflowY: 'auto',
                p: 3,
                display: 'flex',
                flexDirection: 'column',
                gap: 2,
                backgroundColor: '#fafafa',
              }}
            >
              {messages.length === 0 ? (
                <Box
                  sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '100%',
                    color: 'text.secondary',
                  }}
                >
                  <SmartToyIcon sx={{ fontSize: 60, mb: 2, opacity: 0.3 }} />
                  <Typography variant="h6" sx={{ mb: 1 }}>
                    Start a conversation
                  </Typography>
                  <Typography variant="body2">
                    Chatting with <strong>{selectedModel}</strong> in {selectedDeployment}
                  </Typography>
                </Box>
              ) : (
                <>
                  {messages.map((message) => (
                    <Box
                      key={message.id}
                      sx={{
                        display: 'flex',
                        gap: 2,
                        alignItems: 'flex-start',
                        flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                      }}
                    >
                      {/* Avatar */}
                      <Box
                        sx={{
                          width: 36,
                          height: 36,
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                          backgroundColor: message.role === 'user' ? 'primary.main' : '#e0e0e0',
                          color: message.role === 'user' ? 'white' : 'text.primary',
                        }}
                      >
                        {message.role === 'user' ? (
                          <PersonIcon sx={{ fontSize: 20 }} />
                        ) : (
                          <SmartToyIcon sx={{ fontSize: 20 }} />
                        )}
                      </Box>

                      {/* Message Content */}
                      <Box sx={{ maxWidth: '70%' }}>
                        <Box
                          sx={{
                            p: 2,
                            borderRadius: 2,
                            backgroundColor: message.role === 'user' ? 'primary.main' : 'white',
                            color: message.role === 'user' ? 'white' : 'text.primary',
                            boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
                          }}
                        >
                          <Typography
                            variant="body1"
                            sx={{
                              whiteSpace: 'pre-wrap',
                              wordBreak: 'break-word',
                            }}
                          >
                            {message.content}
                          </Typography>
                          <Typography
                            variant="caption"
                            sx={{
                              display: 'block',
                              mt: 1,
                              opacity: 0.7,
                            }}
                          >
                            {message.timestamp.toLocaleTimeString()}
                          </Typography>
                        </Box>

                        {/* Metrics Panel - Only for assistant messages with metrics */}
                        {message.role === 'assistant' && message.metrics && (
                          <Box
                            sx={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: 1,
                              mt: 1,
                              px: 1.5,
                              py: 0.75,
                              backgroundColor: 'rgba(0, 0, 0, 0.03)',
                              borderRadius: 1,
                              fontSize: '0.75rem',
                              color: 'text.secondary',
                            }}
                          >
                            <RocketLaunchIcon sx={{ fontSize: 14, color: 'primary.main' }} />
                            {message.metrics.tokensPerSecond !== null && (
                              <>
                                <Typography variant="caption" sx={{ fontSize: '0.75rem' }}>
                                  {message.metrics.tokensPerSecond.toFixed(1)} t/s
                                </Typography>
                                <Typography variant="caption" sx={{ fontSize: '0.75rem', mx: 0.5 }}>
                                  |
                                </Typography>
                              </>
                            )}
                            {message.metrics.totalLatency !== null && (
                              <>
                                <Typography variant="caption" sx={{ fontSize: '0.75rem' }}>
                                  {message.metrics.totalLatency.toFixed(2)}s
                                </Typography>
                                <Typography variant="caption" sx={{ fontSize: '0.75rem', mx: 0.5 }}>
                                  |
                                </Typography>
                              </>
                            )}
                            {message.metrics.timeToFirstToken !== null && (
                              <Typography variant="caption" sx={{ fontSize: '0.75rem' }}>
                                {message.metrics.timeToFirstToken.toFixed(2)}s to first token
                              </Typography>
                            )}
                          </Box>
                        )}
                      </Box>
                    </Box>
                  ))}
                  {isSending && (
                    <Box
                      sx={{
                        display: 'flex',
                        gap: 2,
                        alignItems: 'flex-start',
                      }}
                    >
                      <Box
                        sx={{
                          width: 36,
                          height: 36,
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                          backgroundColor: '#e0e0e0',
                        }}
                      >
                        <SmartToyIcon sx={{ fontSize: 20 }} />
                      </Box>
                      <Box
                        sx={{
                          p: 2,
                          borderRadius: 2,
                          backgroundColor: 'white',
                          boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
                        }}
                      >
                        <CircularProgress size={20} />
                      </Box>
                    </Box>
                  )}
                  <div ref={messagesEndRef} />
                </>
              )}
            </Box>

            {/* Input Section */}
            <Box
              sx={{
                p: 2,
                borderTop: '1px solid',
                borderColor: 'divider',
                backgroundColor: 'white',
              }}
            >
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
                <TextField
                  fullWidth
                  multiline
                  maxRows={4}
                  placeholder="Type your message..."
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={isSending}
                  variant="outlined"
                  size="small"
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 2,
                    },
                  }}
                />
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim() || isSending}
                  sx={{
                    minWidth: 50,
                    height: 40,
                    borderRadius: 2,
                  }}
                >
                  <SendIcon />
                </Button>
              </Box>
              <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                Press Enter to send, Shift+Enter for new line
              </Typography>
            </Box>
          </>
        )}

        {/* Prompt to select model if deployment selected but no model */}
        {selectedDeployment && !selectedModel && !loadingModels && availableModels.length > 0 && (
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              p: 4,
              color: 'text.secondary',
            }}
          >
            <SmartToyIcon sx={{ fontSize: 80, mb: 2, opacity: 0.2 }} />
            <Typography variant="h6" sx={{ mb: 1 }}>
              Select a model to continue
            </Typography>
            <Typography variant="body2">
              Choose a model from the dropdown above
            </Typography>
          </Box>
        )}

        {/* Prompt to select deployment if none selected */}
        {!selectedDeployment && !loading && !error && deployedBundles.length > 0 && (
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              p: 4,
              color: 'text.secondary',
            }}
          >
            <SmartToyIcon sx={{ fontSize: 80, mb: 2, opacity: 0.2 }} />
            <Typography variant="h6" sx={{ mb: 1 }}>
              Select a deployment to get started
            </Typography>
            <Typography variant="body2">
              Choose a deployed bundle from the dropdown above
            </Typography>
          </Box>
        )}
      </Paper>

      {/* View Code Dialog */}
      <ViewCodeDialog
        open={viewCodeDialogOpen}
        onClose={() => setViewCodeDialogOpen(false)}
        apiKey={apiKey}
        apiDomain={apiDomain}
        modelName={selectedModel}
      />
    </Box>
  );
}
