
// import React, { useRef, useState } from 'react';
// import axios from 'axios';
// import './Upload.css';

// const Upload = () => {
//   const fileInputRef = useRef(null);
//   const [previewSrc, setPreviewSrc] = useState('');
//   const [progress, setProgress] = useState(0);
//   const [showProgress, setShowProgress] = useState(false);
//   const [prediction, setPrediction] = useState(null);
//   const [isLoading, setIsLoading] = useState(false);

//   const handleFileChange = (e) => {
//     const file = e.target.files[0];
//     if (file) {
//       const reader = new FileReader();
//       reader.onload = (event) => setPreviewSrc(event.target.result);
//       reader.readAsDataURL(file);
//       setPrediction(null); // Reset previous prediction when new file is selected
//     }
//   };

//   const handleDrop = (e) => {
//     e.preventDefault();
//     const file = e.dataTransfer.files[0];
//     if (file) {
//       fileInputRef.current.files = e.dataTransfer.files;
//       handleFileChange({ target: { files: e.dataTransfer.files } });
//     }
//   };

//   const handleUpload = async () => {
//     if (!fileInputRef.current.files.length) {
//       alert('Please select an image to upload.');
//       return;
//     }

//     setShowProgress(true);
//     setIsLoading(true);
//     setProgress(0);

//     // Simulate progress (keeping your existing progress animation)
//     const progressInterval = setInterval(() => {
//       setProgress((prev) => (prev >= 90 ? prev : prev + 10));
//     }, 150);

//     try {
//       const formData = new FormData();
//       formData.append('file', fileInputRef.current.files[0]);

//       // Send to FastAPI backend
//       const response = await axios.post(
//         'http://localhost:8000/classify',
//         formData,
//         {
//           headers: {
//             'Content-Type': 'multipart/form-data',
//           },
//         }
//       );

//       clearInterval(progressInterval);
//       setProgress(100);
//       setPrediction(response.data.waste_type);
//     } catch (error) {
//       console.error('Classification error:', error);
//       alert('Classification failed. Please try again.');
//     } finally {
//       setTimeout(() => {
//         setShowProgress(false);
//         setIsLoading(false);
//       }, 500);
//     }
//   };

//   return (
//     <>
//       <nav className="navbar">
//         <div className="logo">🌿 Waste Segregation</div>
//         <ul className="nav-links">
//           <li><a href="/">Home</a></li>
//           <li><a href="/about">About</a></li>
//           <li><a href="/upload">Upload</a></li>
//           <li><a href="/contact">Contact</a></li>
//         </ul>
//       </nav>

//       <div className="upload-container">
//         <h2>Upload Your Image</h2>

//         <div 
//           className="drop-area"
//           onDragOver={(e) => e.preventDefault()}
//           onDrop={handleDrop}
//           onClick={() => fileInputRef.current.click()}
//         >
//           <p>Drag & Drop or Click to Upload</p>
//           <input 
//             type="file" 
//             ref={fileInputRef} 
//             onChange={handleFileChange} 
//             accept="image/*" 
//             hidden 
//           />
//         </div>

//         {showProgress && (
//           <div className="progress-bar">
//             <div className="progress" style={{ width: `${progress}%` }}></div>
//           </div>
//         )}

//         <button 
//           className="upload-button" 
//           onClick={handleUpload}
//           disabled={isLoading}
//         >
//           {isLoading ? 'Classifying...' : 'Upload'}
//         </button>

//         {previewSrc && (
//           <div className="fade-in">
//             <img 
//               src={previewSrc} 
//               alt="Preview" 
//               className="image-preview" 
//             />
//             {prediction && (
//               <div className="fade-in" style={{
//                 marginTop: '15px',
//                 padding: '10px',
//                 borderRadius: '8px',
//                 background: 'rgba(0, 0, 0, 0.3)'
//               }}>
//                 <h3 style={{ color: '#2DFDB3', marginBottom: '5px' }}>Classification Result:</h3>
//                 <p style={{ 
//                   fontSize: '18px', 
//                   fontWeight: 'bold',
//                   color: 'white',
//                   textTransform: 'uppercase'
//                 }}>{prediction}</p>
//               </div>
//             )}
//           </div>
//         )}
//       </div>

//       {/* <footer>
//         <p>© 2025 Waste Segregation. All rights reserved.</p>
//       </footer> */}
//     </>
//   );
// };

// export default Upload;

import React, { useRef, useState } from 'react';
import axios from 'axios';
import './Upload.css';

const Upload = () => {
  const fileInputRef = useRef(null);
  const [previewSrc, setPreviewSrc] = useState('');
  const [progress, setProgress] = useState(0);
  const [showProgress, setShowProgress] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Additional state for extra info returned by FastAPI
  const [description, setDescription] = useState('');
  const [energyConversion, setEnergyConversion] = useState([]);
  const [recyclable, setRecyclable] = useState(null);
  const [handling, setHandling] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => setPreviewSrc(event.target.result);
      reader.readAsDataURL(file);
      // Reset any previous prediction info
      setPrediction(null);
      setDescription('');
      setEnergyConversion([]);
      setRecyclable(null);
      setHandling('');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      fileInputRef.current.files = e.dataTransfer.files;
      handleFileChange({ target: { files: e.dataTransfer.files } });
    }
  };

  const handleUpload = async () => {
    if (!fileInputRef.current.files.length) {
      alert('Please select an image to upload.');
      return;
    }

    setShowProgress(true);
    setIsLoading(true);
    setProgress(0);

    // Simulate progress (you can also use real progress events if available)
    const progressInterval = setInterval(() => {
      setProgress((prev) => (prev >= 90 ? prev : prev + 10));
    }, 150);

    try {
      const formData = new FormData();
      formData.append('file', fileInputRef.current.files[0]);

      // Send image to FastAPI endpoint
      const response = await axios.post(
        'http://localhost:8000/classify',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );

      clearInterval(progressInterval);
      setProgress(100);

      // Update state with response data
      setPrediction(response.data.waste_type);
      setDescription(response.data.description);
      setEnergyConversion(response.data.energy_conversion || []);
      setRecyclable(response.data.recyclable ? 'Yes' : 'No');
      setHandling(response.data.handling);
      
    } catch (error) {
      console.error('Classification error:', error);
      alert('Classification failed. Please try again.');
    } finally {
      setTimeout(() => {
        setShowProgress(false);
        setIsLoading(false);
      }, 500);
    }
  };

  return (
    <>
      <nav className="navbar">
        <div className="logo">🌿 Waste Segregation</div>
        <ul className="nav-links">
          <li><a href="/">Home</a></li>
          <li><a href="/about">About</a></li>
          <li><a href="/upload">Upload</a></li>
          <li><a href="/contact">Contact</a></li>
        </ul>
      </nav>

      <div className="upload-container">
        <h2>Upload Your Image</h2>

        <div 
          className="drop-area"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current.click()}
        >
          <p>Drag & Drop or Click to Upload</p>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept="image/*" 
            hidden 
          />
        </div>

        {showProgress && (
          <div className="progress-bar">
            <div className="progress" style={{ width: `${progress}%` }}></div>
          </div>
        )}

        <button 
          className="upload-button" 
          onClick={handleUpload}
          disabled={isLoading}
        >
          {isLoading ? 'Classifying...' : 'Upload'}
        </button>

        {previewSrc && (
          <div className="fade-in">
            <img 
              src={previewSrc} 
              alt="Preview" 
              className="image-preview" 
            />
            {prediction && (
              <div className="info-box">
                <h3>Classification Result:</h3>
                <p className="predicted-class">{prediction}</p>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Recyclable:</strong> {recyclable}</p>
                <p><strong>Handling:</strong> {handling}</p>
                <div>
                  <strong>Energy Conversion:</strong>
                  {energyConversion.length > 0 ? (
                    <ul>
                      {energyConversion.map((method) => (
                        <li key={method}>{method}</li>
                      ))}
                    </ul>
                  ) : (
                    <span> None</span>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
};

export default Upload;
