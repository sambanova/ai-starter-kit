'use client';

import { useState, useEffect } from 'react';
import {
  Drawer,
  IconButton,
  Box,
  Typography,
  CircularProgress,
} from '@mui/material';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import CloseIcon from '@mui/icons-material/Close';
import ReactMarkdown from 'react-markdown';

interface DocumentationPanelProps {
  docFile: string; // e.g., 'home.md', 'bundle-builder.md'
}

export default function DocumentationPanel({ docFile }: DocumentationPanelProps) {
  const [open, setOpen] = useState(false);
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const toggleDrawer = () => {
    setOpen(!open);
  };

  useEffect(() => {
    if (open && !content) {
      fetchDocumentation();
    }
  }, [open]);

  const fetchDocumentation = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/docs/${docFile}`);
      if (!response.ok) {
        throw new Error('Failed to load documentation');
      }
      const text = await response.text();
      setContent(text);
    } catch (err: any) {
      console.error('Error loading documentation:', err);
      setError('Failed to load documentation');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Book Icon Button */}
      <IconButton
        onClick={toggleDrawer}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          zIndex: 1200,
          backgroundColor: 'primary.main',
          color: 'white',
          '&:hover': {
            backgroundColor: 'primary.dark',
          },
          boxShadow: 3,
          width: 56,
          height: 56,
        }}
        aria-label="Open documentation"
      >
        <MenuBookIcon />
      </IconButton>

      {/* Side Panel Drawer */}
      <Drawer
        anchor="right"
        open={open}
        onClose={toggleDrawer}
        sx={{
          '& .MuiDrawer-paper': {
            width: { xs: '100%', sm: '400px', md: '500px' },
            p: 3,
            mt: '64px', // Offset for navbar
            height: 'calc(100vh - 64px)',
          },
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Documentation
          </Typography>
          <IconButton onClick={toggleDrawer} size="small">
            <CloseIcon />
          </IconButton>
        </Box>

        {/* Content */}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Typography color="error" sx={{ py: 2 }}>
            {error}
          </Typography>
        )}

        {content && !loading && !error && (
          <Box
            sx={{
              overflowY: 'auto',
              '& h1': {
                fontSize: '1.75rem',
                fontWeight: 700,
                mb: 2,
                mt: 1,
              },
              '& h2': {
                fontSize: '1.35rem',
                fontWeight: 600,
                mb: 1.5,
                mt: 3,
              },
              '& h3': {
                fontSize: '1.1rem',
                fontWeight: 600,
                mb: 1,
                mt: 2,
              },
              '& p': {
                mb: 2,
                lineHeight: 1.6,
              },
              '& ul, & ol': {
                pl: 3,
                mb: 2,
              },
              '& li': {
                mb: 0.5,
              },
              '& code': {
                backgroundColor: 'rgba(0, 0, 0, 0.05)',
                padding: '2px 6px',
                borderRadius: '4px',
                fontFamily: 'monospace',
                fontSize: '0.875rem',
              },
              '& pre': {
                backgroundColor: '#1e1e1e',
                color: '#d4d4d4',
                p: 2,
                borderRadius: 1,
                overflow: 'auto',
                mb: 2,
                '& code': {
                  backgroundColor: 'transparent',
                  padding: 0,
                  color: '#d4d4d4',
                },
              },
              '& strong': {
                fontWeight: 600,
              },
              '& a': {
                color: 'primary.main',
                textDecoration: 'underline',
              },
            }}
          >
            <ReactMarkdown>{content}</ReactMarkdown>
          </Box>
        )}
      </Drawer>
    </>
  );
}
