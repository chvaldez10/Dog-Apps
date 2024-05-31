import "./App.css";
import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Home from "./components/pages/Home";
import About from './components/pages/About';
import Confirm from "./components/pages/Confirm";
import Encyclopedia from "./components/pages/Encyclopedia";
import Register from "./components/pages/Register";
import Consultation from "./components/pages/Consultation";
import Profile from "./components/pages/Profile";
import BreedDetail from "./components/pages/BreedDetail";
import NavBar from "./components/NavBar";
import LoginPage from "./components/pages/LoginPage";
import BreederPostings from "./components/pages/BreederPostings";
import Testimonials from './components/pages/Testimonials';
import Search from './components/pages/Search';

function App() {

  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const handleLogin = () => {
    // Perform login logic (e.g., validate credentials)
    // If login is successful, set isLoggedIn to true
    setIsLoggedIn(true);
  };

  const handleLogout = () => {
    // Perform logout logic (e.g., clear session, reset state)
    // Set isLoggedIn to false
    setIsLoggedIn(false);
  };

  return (
    <div className="App">
      <Router>
        <NavBar isLoggedIn={isLoggedIn} handleLogout={handleLogout} />
        <Routes>
          <Route path="/" exact element={<Home />} />
          <Route path="/about" exact element={<About />} />
          <Route path="/search" exact element={<Search />} />
          <Route path="/confirmation" exact element={<Confirm />} />
          <Route path="/register" exact element={<Register />} />
          <Route path="/consultation" exact element={<Consultation />} />
          <Route path="/profile" exact element={<Profile />} />
          <Route path="/mypostings" exact element={<BreederPostings />} />
          <Route path="/testimonials" exact element={<Testimonials />} />

          {/* Routes for login page*/}
          <Route
            path="/login"
            exact
            element={<LoginPage handleLogin={handleLogin} />} // Pass handleLogin as a prop
          />

          {/* Routes for encyclopedia page*/}
          <Route path="/encyclopedia" exact element={<Encyclopedia />} />

          {/* Routes for dog breed fact*/}
          <Route path="/encyclopedia/:dogID" element={<BreedDetail />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;
