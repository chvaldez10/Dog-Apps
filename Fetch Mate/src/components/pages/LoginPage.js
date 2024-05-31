import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useHistory from React Router
import './LoginPage.css'; // Import CSS file for styling

const LoginPage = ({ handleLogin }) => { // Receive handleLogin function as prop
    const navigate = useNavigate();
  
  // State variables to hold username, password, and error message
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  
  // Function to handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    // Check if the username and password match the expected values
    if (username === 'user' && password === 'password') {
      // Call the handleLogin function passed as prop
      handleLogin();
      // If they match, redirect to the home page
      navigate('/');
      // Reset the form and clear any previous error message
      setUsername('');
      setPassword('');
      setError('');
    } else {
      // If they don't match, display an error message
      setError('Invalid username or password. Please try again.');
    }
  };
  
  return (
    <div className="login-container"> {/* Container for centering the form */}
      <form onSubmit={handleSubmit} className="login-form"> {/* Form with class for styling */}
        <div>
          <label>Username:</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div>
          <label>Password:</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit">Login</button>
        {error && <p className="error-message">{error}</p>} {/* Error message with class for styling */}
      </form>
    </div>
  );
};

export default LoginPage;