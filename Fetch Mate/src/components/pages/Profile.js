import React from 'react';
import { Link } from 'react-router-dom';
import Footer from '../Footer';
import './Profile.css';

const breedImages = {
  "Pharaoh Hound": "/pharaoh-hound.jpg",
  "Silken Windhound": "/silken-windhound.jpg",
  "Greyhound": "/greyhound.jpg",
  "Whippet": "/whippet.jpg",
  "Border Collie": "/border-collie.jpg"
};

const Profile = () => {
  const matchedBreeds = ["Pharaoh Hound", "Silken Windhound", "Greyhound", "Whippet", "Border Collie"];

  return (
    <>
    <div className="profile-container">
      <h3 className="sub-heading" style={{marginTop: '20px'}}>Here are the Top 5 Dog Breeds You Matched With:</h3>
      <div className="breed-container">
        {matchedBreeds.map((breed, index) => (
          <div key={index} className="breed-box">
            <img
              src={breedImages[breed]}
              alt={breed}
              className="breed-image"
            />
            <Link to={`/breeds/${breed}`} className="breed-name-link">
              <p className="breed-name">{breed}</p>
            </Link>
          </div>
        ))}
      </div>
    </div>
    <Footer/>
    </>
  );
};

export default Profile;
