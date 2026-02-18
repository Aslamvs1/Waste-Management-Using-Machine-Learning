import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="logo">
        <i className="fas fa-recycle"></i>
        <span>Waste Segregation</span>
      </div>
      <ul className="nav-links">
        <li><Link to="/" aria-label="Home">Home</Link></li>
        <li><Link to="/about" aria-label="About">About</Link></li>
        <li><Link to="/upload" aria-label="Upload">Upload</Link></li>
        <li><Link to="/contact" aria-label="Contact">Contact</Link></li>
      </ul>
    </nav>
  );
};

export default Navbar;