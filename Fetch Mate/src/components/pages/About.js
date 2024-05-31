import React from 'react';
import './About.css';
import Footer from '../Footer';

const About = () => {
  return (
    <>
    <div className="about-us-container">
      <div className="about-us-content">
        <h1>About Us</h1>
        <p>Welcome to Fetch Mate! We are passionate about reducing the number of abandoned dogs in shelters and on the streets by providing potential dog owners with valuable information before they commit to adopting a dog.</p>
        <p>Our website utilizes a detailed questionnaire to understand the user's preferences, lifestyle, and needs. Based on the responses, we recommend the top 5 best dog breeds that are most suitable for the user.</p>
        <p>How It Works:</p>
        <ol>
          <li>Users start by completing a detailed questionnaire covering various aspects such as living environment, activity level, family composition, and preferences in dog size, temperament, and grooming needs.</li>
          <li>Our advanced algorithm analyzes the user's responses to match them with the most compatible dog breeds from our extensive database.</li>
          <li>Users receive personalized recommendations along with detailed profiles of each recommended breed, including information on temperament, exercise needs, grooming requirements, and potential health concerns.</li>
          <li>Armed with this knowledge, users can make informed decisions about which dog breed best fits their lifestyle and preferences.</li>
          <li>We aim to empower users with the information they need to become responsible and educated dog owners, thereby reducing the likelihood of dogs being abandoned or surrendered to shelters.</li>
        </ol>
        <p>By promoting responsible dog ownership through education and informed decision-making, we strive to create happier homes for dogs and reduce the number of dogs in need of rescue.</p>
      </div>
      <div className="about-us-video">
        <video src='/video/about_us.mp4' autoPlay loop muted />
      </div>
    </div>
    <Footer />
    </>
  );
}

export default About;