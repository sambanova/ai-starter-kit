'use client';

import { useState } from 'react';
import {
  Box,
  Drawer,
  ListItemButton,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import KeyIcon from '@mui/icons-material/Key';
import PersonIcon from '@mui/icons-material/Person';
import BuildIcon from '@mui/icons-material/Build';
import Image from 'next/image';

const drawerWidth = 240;

interface AppLayoutProps {
  children: React.ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const [selectedItem, setSelectedItem] = useState('bundle-builder');

  const topMenuItems = [
    { id: 'api-keys', label: 'API Keys', icon: <KeyIcon /> },
    { id: 'bundle-builder', label: 'Bundle Builder', icon: <BuildIcon /> },
  ];

  const bottomMenuItems = [
    { id: 'profile', label: 'Profile', icon: <PersonIcon /> },
  ];

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
        {topMenuItems.map((item) => (
          <ListItemButton
            key={item.id}
            selected={selectedItem === item.id}
            onClick={() => setSelectedItem(item.id)}
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
                color: selectedItem === item.id ? 'primary.main' : '#71717A',
              }}
            >
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{
                fontSize: '0.875rem',
                fontWeight: selectedItem === item.id ? 600 : 500,
                fontFamily: 'var(--font-geist-sans)',
              }}
            />
          </ListItemButton>
        ))}
      </Box>

      {/* Spacer to push bottom items down */}
      <Box sx={{ flexGrow: 1 }} />

      {/* Bottom menu items */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
        {bottomMenuItems.map((item) => (
          <ListItemButton
            key={item.id}
            selected={selectedItem === item.id}
            onClick={() => setSelectedItem(item.id)}
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
                color: selectedItem === item.id ? 'primary.main' : '#71717A',
              }}
            >
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{
                fontSize: '0.875rem',
                fontWeight: selectedItem === item.id ? 600 : 500,
                fontFamily: 'var(--font-geist-sans)',
              }}
            />
          </ListItemButton>
        ))}
      </Box>
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
        {children}
      </Box>
    </Box>
  );
}
