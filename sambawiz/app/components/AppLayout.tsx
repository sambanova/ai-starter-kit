'use client';

import { useState, useEffect } from 'react';
import {
  Box,
  Drawer,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Alert,
} from '@mui/material';
import KeyIcon from '@mui/icons-material/Key';
import PersonIcon from '@mui/icons-material/Person';
import BuildIcon from '@mui/icons-material/Build';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import Image from 'next/image';
import { useRouter, usePathname } from 'next/navigation';

const drawerWidth = 240;

interface AppLayoutProps {
  children: React.ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const router = useRouter();
  const pathname = usePathname();
  const [selectedItem, setSelectedItem] = useState('bundle-builder');
  const [envVersion, setEnvVersion] = useState<string | null>(null);
  const [envName, setEnvName] = useState<string | null>(null);
  const [namespace, setNamespace] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Update selected item based on current pathname
  useEffect(() => {
    if (pathname === '/') {
      setSelectedItem('bundle-builder');
    } else if (pathname === '/bundle-deployment') {
      setSelectedItem('bundle-deployment');
    }
  }, [pathname]);

  // Validate kubeconfig on component mount
  useEffect(() => {
    const validateKubeconfig = async () => {
      try {
        const response = await fetch('/api/kubeconfig-validate');
        const data = await response.json();

        if (data.success) {
          setEnvVersion(data.version);
          setEnvName(data.envName);
          setNamespace(data.namespace);
          setValidationError(null);
        } else {
          setValidationError(data.error);
          setEnvVersion(null);
          setEnvName(null);
          setNamespace(null);
        }
      } catch (error) {
        console.error('Failed to validate kubeconfig:', error);
        setValidationError('Your kubeconfig.yaml seems to be invalid. Please check it and re-run the app. Also ensure that you are on the right network/VPN to access the server.');
        setEnvVersion(null);
        setEnvName(null);
        setNamespace(null);
      }
    };

    validateKubeconfig();
  }, []);

  const drawer = (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        pt: 3,
        pb: 3,
      }}
    >
      <Box sx={{ mb: 3, px: 2 }}>
        <Image
          src="/sidebar-logo.svg"
          alt="SambaNova Logo"
          width={150}
          height={40}
          style={{ width: '150px', height: 'auto' }}
          priority
        />
      </Box>

      {/* Top menu items */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
        <ListItemButton
          selected={selectedItem === 'bundle-builder'}
          onClick={() => {
            setSelectedItem('bundle-builder');
            router.push('/');
          }}
          sx={{
            mx: 2,
            px: 1,
            py: 1.25,
            borderRadius: 2,
            gap: 2,
            '&.Mui-selected': {
              backgroundColor: 'rgb(232, 229, 234)',
              '&:hover': {
                backgroundColor: 'rgb(232, 229, 234)',
              },
            },
            '&:hover': {
              backgroundColor: 'rgb(232, 229, 234)',
              borderRadius: 2,
            },
          }}
        >
          <ListItemIcon
            sx={{
              minWidth: 'auto',
              color: selectedItem === 'bundle-builder' ? 'primary.main' : '#71717A',
            }}
          >
            <BuildIcon />
          </ListItemIcon>
          <ListItemText
            primary="SambaWiz"
            primaryTypographyProps={{
              fontSize: '0.875rem',
              fontWeight: selectedItem === 'bundle-builder' ? 600 : 500,
              fontFamily: 'var(--font-geist-sans)',
            }}
          />
        </ListItemButton>

        <ListItemButton
          selected={selectedItem === 'bundle-deployment'}
          onClick={() => {
            setSelectedItem('bundle-deployment');
            router.push('/bundle-deployment');
          }}
          sx={{
            mx: 2,
            px: 1,
            py: 1.25,
            borderRadius: 2,
            gap: 2,
            '&.Mui-selected': {
              backgroundColor: 'rgb(232, 229, 234)',
              '&:hover': {
                backgroundColor: 'rgb(232, 229, 234)',
              },
            },
            '&:hover': {
              backgroundColor: 'rgb(232, 229, 234)',
              borderRadius: 2,
            },
          }}
        >
          <ListItemIcon
            sx={{
              minWidth: 'auto',
              color: selectedItem === 'bundle-deployment' ? 'primary.main' : '#71717A',
            }}
          >
            <RocketLaunchIcon />
          </ListItemIcon>
          <ListItemText
            primary="Bundle Deployment"
            primaryTypographyProps={{
              fontSize: '0.875rem',
              fontWeight: selectedItem === 'bundle-deployment' ? 600 : 500,
              fontFamily: 'var(--font-geist-sans)',
            }}
          />
        </ListItemButton>
      </Box>

      {/* Spacer to push version display to bottom */}
      <Box sx={{ flexGrow: 1 }} />

      {/* Environment version display */}
      {envVersion && envName && (
        <Box
          sx={{
            mx: 2,
            mt: 2,
            p: 1.5,
            borderRadius: 2,
            backgroundColor: 'rgb(232, 229, 234)',
            border: '1px solid rgb(209, 204, 213)',
          }}
        >
          <Typography
            sx={{
              fontSize: '0.875rem',
              fontWeight: 600,
              color: 'primary.main',
              mb: 0.5,
              fontFamily: 'var(--font-geist-sans)',
              textAlign: 'center',
            }}
          >
            {envName}
          </Typography>
          <Typography
            sx={{
              fontSize: '0.75rem',
              fontWeight: 500,
              color: '#71717A',
              fontFamily: 'var(--font-geist-sans)',
              textAlign: 'center',
            }}
          >
            version: {envVersion}
          </Typography>
          {namespace && (
            <Typography
              sx={{
                fontSize: '0.75rem',
                fontWeight: 500,
                color: '#71717A',
                fontFamily: 'var(--font-geist-sans)',
                textAlign: 'center',
              }}
            >
              namespace: {namespace}
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', backgroundColor: 'background.default' }}>
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            border: 'none',
          },
        }}
      >
        {drawer}
      </Drawer>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          backgroundColor: 'background.default',
          minHeight: '100vh',
        }}
      >
        {validationError && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {validationError}
          </Alert>
        )}
        {children}
      </Box>
    </Box>
  );
}
