import { Link } from "react-router-dom";

import DatasetIcon from "@mui/icons-material/Dataset";
import HomeIcon from "@mui/icons-material/Home";
import LocalShippingIcon from "@mui/icons-material/LocalShipping";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import TuneIcon from "@mui/icons-material/Tune";
import {
  Box,
  Divider,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Theme,
  Toolbar,
  useMediaQuery,
} from "@mui/material";

import { sidebarWidth } from "@/Constants";

interface SidebarProps {
  mobileOpen: boolean;
  onDrawerToggle: () => void;
}

const Sidebar = ({ mobileOpen, onDrawerToggle }: SidebarProps) => {
  const sidebarElements = [
    { text: "Home", icon: <HomeIcon />, path: "/" },
    { text: "Dataset", icon: <DatasetIcon />, path: "/dataset" },
    { text: "Checkpoint", icon: <MyLocationIcon />, path: "/checkpoint" },
    { text: "Fine Tuning", icon: <TuneIcon />, path: "/fine_tuning" },
    { text: "Deployment", icon: <LocalShippingIcon />, path: "/deployment" },
  ];

  const isMobile = useMediaQuery((theme: Theme) =>
    theme.breakpoints.down("sm"),
  );

  const drawer = (
    <div>
      <Toolbar>
        <Box
          component="img"
          src="https://sambanova.ai/hubfs/sambanova-logo-black.png"
          alt="SambaNova Logo"
          sx={{
            display: { xs: "none", sm: "block" }, // Only show on larger screens
          }}
        />
      </Toolbar>
      <Divider />
      <List>
        {sidebarElements.map((item) => (
          <ListItem
            key={item.text}
            component={Link}
            to={item.path}
            onClick={isMobile ? onDrawerToggle : undefined}
          >
            <ListItemIcon sx={{ minWidth: 0, mr: 1.5 }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <Box
      component="nav"
      sx={{
        width: { sm: sidebarWidth },
        flexShrink: { sm: 0 },
      }}
      aria-label="mailbox folders"
    >
      {/* Mobile Drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onDrawerToggle}
        ModalProps={{
          keepMounted: true, // Better open performance on mobile.
        }}
        sx={{
          display: { xs: "block", sm: "none" },
          "& .MuiDrawer-paper": {
            boxSizing: "border-box",
            width: sidebarWidth,
          },
        }}
      >
        {drawer}
      </Drawer>

      {/* Permanent Drawer for Larger Screens */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: "none", sm: "block" },
          "& .MuiDrawer-paper": {
            boxSizing: "border-box",
            width: sidebarWidth,
          },
        }}
        open
      >
        {drawer}
      </Drawer>
    </Box>
  );
};

export default Sidebar;
