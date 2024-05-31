import React from 'react';
import '../../App.css';
import './Home.css';
import { Button } from '../Button';
import Footer from '../Footer';
import { Link } from 'react-router-dom';

function Home() {
    return (
        <>
        <div className='main-container'>
            <video src='/video/happy_dog.mp4' autoPlay loop muted />
            <h1>PAWSOME ADVENTURES AWAIT</h1>
            <h3 style={{ color: '#fff' }}>What are you waiting for?</h3>
            <div className='main-btns'>
                <Link to='/confirmation' className='btn-mobile'>
                    <Button 
                        className='btns'
                        buttonStyle='btn--outline'
                        buttonSize='btn--large'
                    >
                        GET STARTED
                    </Button>
                </Link>
            </div>
        </div>
        <Footer />
        </>
    )
}

export default Home;