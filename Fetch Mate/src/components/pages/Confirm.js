import React from 'react';
import '../../App.css';
import './Confirm.css';
import { Button } from '../Button';
import Footer from '../Footer';
import { Link } from 'react-router-dom';

function Confirm() {
    return (
        <>
        <div className='main-container'>
            <video src='/video/high_five.mp4' autoPlay loop muted />
            <h2 style={{ color: '#fff' }}>ARE YOU LOOKING FOR A DOG?</h2>
            {/* <p style={{ color: '#fff' }}>What are you waiting for?</p> */}
            <div className='main-btns'>
                <Link to='/consultation' className='btn-mobile'>
                    <Button 
                        className='btns'
                        buttonStyle='btn--outline'
                        buttonSize='btn--large'
                    >
                        YES
                    </Button>
                </Link>
                <Link to='/register' className='btn-mobile'>
                    <Button 
                        className='btns'
                        buttonStyle='btn--outline'
                        buttonSize='btn--large'
                    >
                        NO
                    </Button>
                </Link>
            </div>
        </div>
        <Footer />
        </>
    )
}

export default Confirm;