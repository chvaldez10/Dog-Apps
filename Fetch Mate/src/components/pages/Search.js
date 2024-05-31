import React, { useState } from 'react';
import { Button } from '../Button';
import './Search.css';

const Search = () => {
  const [data] = useState([
    { name: 'Breeder 1', location: 'New York', breeds: ['Labrador Retriever', 'Golden Retriever'] },
    { name: 'Adoption Center 1', location: 'Los Angeles', breeds: ['German Shepherd', 'Poodle'] },
    { name: 'Breeder 2', location: 'Toronto', breeds: ['Bulldog', 'Boxer'] },
    { name: 'Adoption Center 2', location: 'Vancouver', breeds: ['Siberian Husky', 'Border Collie'] },
    { name: 'Breeder 3', location: 'Chicago', breeds: ['Beagle', 'Rottweiler'] },
    { name: 'Adoption Center 3', location: 'Seattle', breeds: ['Shih Tzu', 'Chihuahua'] },
    { name: 'Breeder 4', location: 'Montreal', breeds: ['Great Dane', 'Dachshund'] },
    { name: 'Adoption Center 4', location: 'Houston', breeds: ['Golden Retriever', 'Labrador Retriever'] },
    { name: 'Breeder 5', location: 'Miami', breeds: ['Pharaoh Hound', 'Siberian Husky'] },
    { name: 'Adoption Center 5', location: 'Dallas', breeds: ['Boxer', 'Shih Tzu'] },
    { name: 'Breeder 6', location: 'Calgary', breeds: ['Australian Shepherd', 'Cavalier King Charles Spaniel'] },
    { name: 'Adoption Center 6', location: 'San Francisco', breeds: ['French Bulldog', 'Pug'] },
    { name: 'Breeder 7', location: 'Ottawa', breeds: ['Border Collie', 'Bernese Mountain Dog'] },
    { name: 'Adoption Center 7', location: 'Phoenix', breeds: ['Yorkshire Terrier', 'Pomeranian'] },
    { name: 'Breeder 8', location: 'Boston', breeds: ['Labrador Retriever', 'Golden Retriever'] },
    { name: 'Adoption Center 8', location: 'Atlanta', breeds: ['German Shepherd', 'Poodle'] },
    { name: 'Breeder 9', location: 'Edmonton', breeds: ['Boxer', 'Bulldog'] },
    { name: 'Adoption Center 9', location: 'Las Vegas', breeds: ['Siberian Husky', 'Shih Tzu'] },
    { name: 'Breeder 10', location: 'Washington D.C.', breeds: ['Pharaoh Hound', 'Cavalier King Charles Spaniel'] },
  ]);

  const [location, setLocation] = useState('');
  const [breed, setBreed] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [locationSuggestions, setLocationSuggestions] = useState([]);
  const [breedSuggestions, setBreedSuggestions] = useState([]);

  const handleLocationChange = (event) => {
    const value = event.target.value;
    setLocation(value);
    const filteredSuggestions = data.reduce((acc, curr) => {
      if (curr.location.toLowerCase().includes(value.toLowerCase()) && !acc.includes(curr.location)) {
        acc.push(curr.location);
      }
      return acc;
    }, []);
    setLocationSuggestions(filteredSuggestions);
  };

  const handleBreedChange = (event) => {
    const value = event.target.value;
    setBreed(value);
    const filteredSuggestions = data.reduce((acc, curr) => {
      const matchedBreeds = curr.breeds.filter(b => b.toLowerCase().includes(value.toLowerCase()));
      if (matchedBreeds.length > 0) {
        matchedBreeds.forEach(suggestion => {
          if (!acc.includes(suggestion)) {
            acc.push(suggestion);
          }
        });
      }
      return acc;
    }, []);
    setBreedSuggestions(filteredSuggestions);
  };

  const handleLocationSuggestionClick = (suggestion) => {
    setLocation(suggestion);
    setLocationSuggestions([]);
  };

  const handleBreedSuggestionClick = (suggestion) => {
    setBreed(suggestion);
    setBreedSuggestions([]);
  };

  const handleSearch = () => {
    const filteredResults = data.filter(item => {
      const isLocationMatch = location === '' || item.location.toLowerCase().includes(location.toLowerCase());
      const isBreedMatch = breed === '' || item.breeds.some(b => b.toLowerCase().includes(breed.toLowerCase()));
      return isLocationMatch && isBreedMatch;
    });
    setSearchResults(filteredResults);
    setLocationSuggestions([]);
    setBreedSuggestions([]);
  };

  return (
    <div className="search-page-container">
      <h1>Find Breeders and Adoption Places</h1>
      <div className="search-form">
        <div className="dropdown">
          <input
            type="text"
            placeholder="Enter location"
            value={location}
            onChange={handleLocationChange}
          />
          {locationSuggestions.length > 0 && (
            <div className="dropdown-content">
              {locationSuggestions.map((suggestion, index) => (
                <div key={index} onClick={() => handleLocationSuggestionClick(suggestion)}>
                  {suggestion}
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="dropdown">
          <input
            type="text"
            placeholder="Enter dog breed"
            value={breed}
            onChange={handleBreedChange}
          />
          {breedSuggestions.length > 0 && (
            <div className="dropdown-content">
              {breedSuggestions.map((suggestion, index) => (
                <div key={index} onClick={() => handleBreedSuggestionClick(suggestion)}>
                  {suggestion}
                </div>
              ))}
            </div>
          )}
        </div>
        <Button onClick={handleSearch} buttonStyle="btn--primary">Search</Button>
      </div>
      <div className="search-results">
        {searchResults.map((result, index) => (
          <div key={index} className="result-item">
            <h2>{result.name}</h2>
            <p>Location: {result.location}</p>
            <p>Breeds: {result.breeds.join(', ')}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Search;
