// import React from 'react';
import './Home.css'; // Import the CSS file
import heroImage from '../assets/a1.png'; // Adjust the path to your image

const Home = () => {
  return (
    <>
      {/* Navbar */}
      <nav className="navbar">
        <div className="logo">
          <i className="fas fa-recycle"></i>
          <span>EcoSort</span>
        </div>
        <ul className="nav-links">
          <li><a href="/">Home</a></li>
          <li><a href="/about">About</a></li>
          <li><a href="/upload">Upload</a></li>
          <li><a href="/features">Features</a></li>
          <li><a href="/contact">Contact</a></li>
        </ul>
        <a href="/upload" className="get-started-btn">Get Started</a>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-subtitle">AI-Powered Waste Management</div>
          <h1 className="hero-title">Smart Waste Segregation Made Simple</h1>
          <p className="hero-text">
            Harness the power of advanced AI to properly identify and categorize waste. Our cutting-edge image recognition technology helps you segregate waste correctly, reducing environmental pollution and promoting sustainable waste management for a cleaner world.
          </p>
          <div className="hero-buttons">
            <a href="/upload" className="btn">Try It Now <i className="fas fa-arrow-right"></i></a>
            <a href="/about" className="btn btn-secondary">Learn More</a>
          </div>
        </div>
        <div className="hero-image">
          <img src={heroImage} alt="Waste Segregation Illustration" />
        </div>
      </section>

      {/* Benefits Section */}
      <section className="benefits-section">
        <h2 className="section-title">Why Choose EcoSort</h2>
        <div className="card-container">
          <div className="card">
            <div className="card-icon">
              <i className="fas fa-brain"></i>
            </div>
            <h3>AI-Powered</h3>
            <p>Our advanced machine learning algorithms provide accurate waste classification with 98% accuracy rate.</p>
            <a href="/about" className="card-btn">Learn More</a>
          </div>

          <div className="card">
            <div className="card-icon">
              <i className="fas fa-leaf"></i>
            </div>
            <h3>Eco-Friendly</h3>
            <p>Help reduce landfill waste and promote recycling to create a more sustainable future for our planet.</p>
            <a href="/about" className="card-btn">Learn More</a>
          </div>

          <div className="card">
            <div className="card-icon">
              <i className="fas fa-mobile-alt"></i>
            </div>
            <h3>Easy to Use</h3>
            <p>Simply take a photo of your waste item and our system instantly tells you how to dispose of it properly.</p>
            <a href="/upload" className="card-btn">Try Now</a>
          </div>

          <div className="card">
            <div className="card-icon">
              <i className="fas fa-chart-line"></i>
            </div>
            <h3>Real-Time Analytics</h3>
            <p>Track your waste segregation habits and get insights to improve your recycling efforts over time.</p>
            <a href="/features" className="card-btn">Explore</a>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats-section">
        <h2 className="section-title">Our Impact</h2>
        <div className="stats-container">
          <div className="stat-item">
            <span className="stat-number">98%</span>
            <span className="stat-icon"><i className="fas fa-percentage"></i></span>
            <p className="stat-text">Accuracy in Waste Classification</p>
          </div>
          <div className="stat-item">
            <span className="stat-number">500K+</span>
            <span className="stat-icon"><i className="fas fa-users"></i></span>
            <p className="stat-text">Users Worldwide</p>
          </div>
          <div className="stat-item">
            <span className="stat-number">1M+</span>
            <span className="stat-icon"><i className="fas fa-recycle"></i></span>
            <p className="stat-text">Items Segregated</p>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="how-it-works">
        <h2 className="section-title">How It Works</h2>
        <div className="process-container">
          <div className="process-step">
            <div className="step-number">1</div>
            <h3 className="step-title">Upload Image</h3>
            <p className="step-desc">Take a photo of the waste item or upload an image from your device.</p>
          </div>
          <div className="process-step">
            <div className="step-number">2</div>
            <h3 className="step-title">AI Analysis</h3>
            <p className="step-desc">Our AI analyzes the image and identifies the type of waste.</p>
          </div>
          <div className="process-step">
            <div className="step-number">3</div>
            <h3 className="step-title">Get Results</h3>
            <p className="step-desc">Receive instant feedback on how to properly segregate the waste.</p>
          </div>
        </div>
      </section>

      {/* Call-to-Action Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h2 className="cta-title">Join the Movement for a Cleaner Planet</h2>
          <p className="cta-text">Start segregating your waste responsibly today. Together, we can make a difference.</p>
          <div className="cta-buttons">
            <a href="/upload" className="btn">Get Started</a>
            <a href="/about" className="btn btn-secondary">Learn More</a>
          </div>
        </div>
      </section>

      {/* Footer
      <footer>
        <div className="footer-content">
          <div className="footer-info">
            <div className="footer-logo">
              <i className="fas fa-recycle"></i>
              <span>EcoSort</span>
            </div>
            <p className="footer-description">
              EcoSort is an AI-powered waste management system designed to help individuals and businesses segregate waste effectively and promote a sustainable future.
            </p>
            <div className="social-icons">
              <a href="https://www.facebook.com" target="_blank"><i className="fab fa-facebook-f"></i></a>
              <a href="https://www.twitter.com" target="_blank"><i className="fab fa-twitter"></i></a>
              <a href="https://www.instagram.com" target="_blank"><i className="fab fa-instagram"></i></a>
              <a href="https://www.linkedin.com" target="_blank"><i className="fab fa-linkedin-in"></i></a>
            </div>
          </div>
          <div className="footer-nav">
            <div className="footer-nav-section">
              <h4 className="footer-nav-title">Quick Links</h4>
              <ul className="footer-links">
                <li><a href="/">Home</a></li>
                <li><a href="/about">About</a></li>
                <li><a href="/upload">Upload</a></li>
                <li><a href="/features">Features</a></li>
                <li><a href="/contact">Contact</a></li>
              </ul>
            </div>
            <div className="footer-nav-section">
              <h4 className="footer-nav-title">Resources</h4>
              <ul className="footer-links">
                <li><a href="/blog">Blog</a></li>
                <li><a href="/faq">FAQ</a></li>
                <li><a href="/privacy">Privacy Policy</a></li>
                <li><a href="/terms">Terms of Service</a></li>
              </ul>
            </div>
          </div>
        </div>
        <div className="divider"></div>
        <div className="copyright">
          &copy; 2025 EcoSort. All rights reserved.
        </div>
      </footer> */}
    </>
  );
};

export default Home;