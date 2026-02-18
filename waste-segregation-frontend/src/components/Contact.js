import React from 'react';
import './Contact.css';
import { FaInstagram, FaWhatsapp } from 'react-icons/fa';

const Contact = () => {
  return (
    <>
      <nav className="navbar">
        <div className="logo">🌿 Waste Segregation</div>
        <ul className="nav-links">
          <li><a href="/home">Home</a></li>
          <li><a href="/about">About</a></li>
          <li><a href="/upload">Upload</a></li>
          <li><a href="/contact">Contact</a></li>
        </ul>
      </nav>

      <div className="contact-container">
        <h2>Contact Us</h2>
        <form className="contact-form">
          <input type="text" placeholder="Your Name" required />
          <input type="tel" placeholder="Phone Number" required />
          <input type="email" placeholder="Your Email" required />
          <textarea placeholder="Your Feedback"></textarea>
          <button type="submit" className="send-button">Send Response</button>
        </form>
        <div className="social-icons">
          <a href="https://www.instagram.com" target="_blank" rel="noopener noreferrer" title="Instagram">
            <FaInstagram />
          </a>
          <a href="https://wa.me/" target="_blank" rel="noopener noreferrer" title="WhatsApp">
            <FaWhatsapp />
          </a>
        </div>
      </div>
    </>
  );
};

export default Contact;
