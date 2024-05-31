import React from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  ThemeProvider,
} from "@mui/material";
import PetsIcon from "@mui/icons-material/Pets"; // Import the paw icon
import SearchIcon from "@mui/icons-material/Search";
import { Search, SearchIconWrapper, StyledInputBase } from "./NavBarSearch";
import { navBarTheme } from "./styles/commonStyles";

function NavBar({ isLoggedIn, handleLogout }) {

  const navigate = useNavigate();
  
  const handleLogoutClick = () => {
    handleLogout(); // Call the logout function passed as prop
    navigate('/'); // Navigate to the home page
  };

  return (
    <ThemeProvider theme={navBarTheme}>
      <AppBar position="sticky">
        <Toolbar>
          {/* Paw icon */}
          <IconButton edge="start" color="inherit" aria-label="menu">
            <PetsIcon />
          </IconButton>

          {/* Company Name */}
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            <Link to="/" style={{ textDecoration: "none", color: "inherit" }}>
              FETCH MATE
            </Link>
          </Typography>

          {/* Search Bar */}
          <Search>
            <SearchIconWrapper>
              <SearchIcon />
            </SearchIconWrapper>
            <StyledInputBase
              placeholder="Search…"
              inputProps={{ "aria-label": "search" }}
            />
          </Search>
          <Button color="inherit" component={Link} to="/about">
            About Us
          </Button>
          <Button color="inherit" component={Link} to="/search">
            Find a Dog
          </Button>
          <Button color="inherit" component={Link} to="/encyclopedia">
            Encyclopedia
          </Button>
          <Button color="inherit">Forum</Button>

          {/* Render "Logout" button if user is logged in, otherwise render "Login" button */}
          {isLoggedIn ? (
            <Button color="inherit" onClick={handleLogoutClick}>
              Logout
            </Button>
          ) : (
            <Button color="inherit" component={Link} to="/login">
              Login
            </Button>
          )}

        </Toolbar>
      </AppBar>
    </ThemeProvider>
  );
}

export default NavBar;
