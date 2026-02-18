// import React from 'react';
import './About.css'; // Import the CSS file
import aboutImage from '../assets/a.png'; // Adjust the path to your image

const About = () => {
  return (
    <>
      {/* Particle Background */}
      <div className="particles" id="particles"></div>

      {/* Navigation Bar */}
      <nav className="navbar">
        <div className="logo"><i className="fas fa-recycle"></i> Waste Segregation</div>
        <ul className="nav-links" id="navLinks">
          <li><a href="/">Home</a></li>
          <li><a href="/about">About</a></li>
          <li><a href="/upload">Upload</a></li>
          <li><a href="/contact">Contact</a></li>
        </ul>
        <div className="mobile-menu" id="mobileMenu">
          <i className="fas fa-bars"></i>
        </div>
      </nav>

      {/* About Section */}
      <section className="about-container">
        <h1 className="about-title">About Us</h1>
        <div className="section-divider"></div>
        <p className="about-subtitle">
          We are committed to creating a sustainable future through proper waste management and innovative solutions.
        </p>

        <div className="about-content">
          <div className="about-text">
            <p>
              At <span className="highlight">Waste Segregation</span>, we believe in the power of responsible waste management to protect our planet. 
              Our mission is to educate and empower individuals and communities to segregate waste effectively, 
              ensuring that recyclable materials are reused and harmful waste is disposed of safely.
            </p>
            <p>
              With innovative solutions and a dedicated team of environmental experts, we aim to reduce environmental pollution and promote 
              a circular economy where waste becomes a valuable resource rather than a problem. Join us in making a difference for a cleaner, greener future.
            </p>
            <p>
              Through education, technology, and community involvement, we're creating sustainable solutions that have a real impact on our environment and future generations.
            </p>
          </div>
          <div className="about-image">
            <img src={aboutImage} alt="About Us Image" />
          </div>
        </div>

        {/* Stats Section */}
        <div className="stats-section">
          <div className="stat-item">
            <div className="stat-value" id="communityCount">5K+</div>
            <div className="stat-label">Community Members</div>
          </div>
          <div className="stat-item">
            <div className="stat-value" id="wasteCount">200+</div>
            <div className="stat-label">Tons of Waste Recycled</div>
          </div>
          <div className="stat-item">
            <div className="stat-value" id="projectCount">50+</div>
            <div className="stat-label">Awareness Programs</div>
          </div>
          <div className="stat-item">
            <div className="stat-value" id="volunteerCount">300+</div>
            <div className="stat-label">Active Volunteers</div>
          </div>
        </div>

        {/* Mission & Vision Section */}
        <div className="mission-vision">
          <div className="mission">
            <h3><i className="fas fa-bullseye"></i> Our Mission</h3>
            <p>
              To promote sustainable waste management practices and reduce environmental pollution through 
              education, innovation, and community engagement. We strive to create awareness about proper waste segregation 
              and implement effective solutions that make recycling accessible to everyone.
            </p>
          </div>
          <div className="vision">
            <h3><i className="fas fa-eye"></i> Our Vision</h3>
            <p>
              A world where waste is seen as a resource, and every individual contributes to a cleaner, 
              healthier planet. We envision communities where proper waste management is second nature, 
              recycling rates approach 100%, and the concept of "waste" is replaced by "resource" in our collective mindset.
            </p>
          </div>
        </div>

        {/* Team Section */}
        <div className="team-section">
          <h2 className="team-title">Our Team</h2>
          <div className="section-divider"></div>
          <div className="team-grid">
            <div className="team-member">
              {/* <img className="team-photo" src="/api/placeholder/400/320" alt="Team Member" /> */}
              <div className="team-info">
                <h3 className="team-name">Muhammed Aslam</h3>
                <p className="team-role">Founder & CEO</p>
                <p className="team-bio">Environmental engineer with 15+ years of experience in waste management and sustainability initiatives.</p>
                <div className="social-links">
                  <a href="#"><i className="fab fa-linkedin"></i></a>
                  <a href="#"><i className="fab fa-twitter"></i></a>
                  <a href="#"><i className="fas fa-envelope"></i></a>
                </div>
              </div>
            </div>
            <div className="team-member">
              {/* <img className="team-photo" src="/api/placeholder/400/320" alt="Team Member" /> */}
              <div className="team-info">
                <h3 className="team-name">Aysha Narmin</h3>
                <p className="team-role">Technical Director</p>
                <p className="team-bio">Technology expert specializing in developing waste segregation systems and recycling technologies.</p>
                <div className="social-links">
                  <a href="#"><i className="fab fa-linkedin"></i></a>
                  <a href="#"><i className="fab fa-twitter"></i></a>
                  <a href="#"><i className="fas fa-envelope"></i></a>
                </div>
              </div>
            </div>
            <div className="team-member">
              {/* <img className="team-photo" src="/api/placeholder/400/320" alt="Team Member" /> */}
              <div className="team-info">
                <h3 className="team-name">Sulekha Hannath&Jalwa Jannath</h3>
                <p className="team-role">Community Outreach</p>
                <p className="team-bio">Dedicated to building relationships and creating educational programs for communities and schools.</p>
                <div className="social-links">
                  <a href="#"><i className="fab fa-linkedin"></i></a>
                  <a href="#"><i className="fab fa-twitter"></i></a>
                  <a href="#"><i className="fas fa-envelope"></i></a>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="cta-section">
          <h2 className="cta-title">Join Our Mission</h2>
          <p className="cta-text">
            Be part of the solution for a sustainable future. Together, we can make a significant impact 
            on waste management and environmental conservation.
          </p>
          <a href="/contact" className="cta-button">Get Involved</a>
        </div>
      </section>
    </>
  );
};

export default About;