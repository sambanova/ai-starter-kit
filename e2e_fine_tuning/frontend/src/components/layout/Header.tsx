import MenuIcon from "@mui/icons-material/Menu";
import SettingsIcon from "@mui/icons-material/Settings";
import {
  AppBar,
  Box,
  IconButton,
  Theme,
  Toolbar,
  Typography,
  useMediaQuery,
} from "@mui/material";

import { sidebarWidth } from "@/Constants";

interface HeaderProps {
  onDrawerToggle: () => void;
}

const Header = ({ onDrawerToggle }: HeaderProps) => {
  const isMobile = useMediaQuery((theme: Theme) =>
    theme.breakpoints.down("sm"),
  );

  return (
    <AppBar
      position="fixed"
      color="inherit"
      sx={{
        width: {
          sm: `calc(100% - ${sidebarWidth}px)`,
          xs: "100%",
        },
        ml: { sm: `${sidebarWidth}px` },
        zIndex: (theme) => theme.zIndex.drawer + 1,
      }}
    >
      <Toolbar>
        {isMobile && (
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={onDrawerToggle}
            sx={{ mr: 2, display: { sm: "none" } }}
          >
            <MenuIcon />
          </IconButton>
        )}
        <Box
          component="img"
          src="https://sambanova.ai/hubfs/logotype_sambanova_orange.png"
          alt="SambaNova Logo"
          sx={{
            height: 35,
            display: { xs: "block", sm: "none" }, // Only show on mobile
            mr: 2,
          }}
        />
        <Typography variant="h6" noWrap component="div">
          Bring your own Checkpoint
        </Typography>

        <SettingsIcon />
      </Toolbar>
    </AppBar>
  );
};

export default Header;
