import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");

    if (savedTheme === "dark") {
       setDarkMode(true);
       document.body.classList.add("dark-theme");
    }
  }, []);

  const toggleTheme = () => {
    const newTheme = !darkMode;

    setDarkMode(newTheme);

    document.body.classList.toggle("dark-theme", newTheme);

    localStorage.setItem("theme", newTheme ? "dark" : "light");
  };

  const handleImage = (e) => {
    const file = e.target.files[0];

    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const predictDisease = async () => {
    if (!image) {
      alert("Please select an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    setLoading(true);

    try {
      const response = await axios.post(
        "http://34.193.36.46:8000/predict",
        formData
      );

      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Prediction Failed");
    }

    setLoading(false);
  };

  return (
    <div className="app">

      {/* NAVBAR */}

      <nav className="navbar">

        <div className="logo">
          🌿 <span>LeafGuard AI</span>
        </div>

        <button
          className="theme-btn"
          onClick={toggleTheme}
          aria-label="Toggle Theme"
        >
          {darkMode ? "🌿" : "🍃"}
        </button>

      </nav>

      {/* HERO */}

      <section className="hero">

        <div className="hero-left">

          <span className="tag">
            AI Powered Plant Health Analysis
          </span>

          <h1>
            Detect Plant Diseases
            <br />
            Instantly
          </h1>

          <p>
            Upload a clear leaf image and let our AI model detect
            plant diseases within seconds using TensorFlow MobileNetV2.
          </p>

        </div>

        <div className="hero-right">

          <div className="placeholder">

            {preview ? (
              <img
                src={preview}
                alt="Leaf Preview"
                className="hero-image"
              />
            ) : (
              <span className="hero-leaf">🍃</span>
            )}

          </div>

        </div>

      </section>

      {/* UPLOAD */}

    <section className="upload-card">
            <section className="upload-card">

         <h2>🌿 Analyze Your Plant</h2>

<p className="upload-subtitle">
  Upload a healthy or infected leaf image and let AI detect the disease instantly.
</p>

<div className="button-group">

  <label htmlFor="upload" className="upload-btn">
    📁 Upload Leaf
  </label>

  <button
    className="predict-btn"
    onClick={predictDisease}
    disabled={loading}
  >
    {loading ? (
      <>
        <span className="spinner"></span>
        Analyzing...
      </>
    ) : (
      "🌿 Analyze Plant"
    )}
  </button>

</div>

<input
  id="upload"
  type="file"
  accept="image/*"
  onChange={handleImage}
  hidden
/>

{image && (
  <div className="selected-file">
    ✅ Selected: {image.name}
  </div>
)}

{preview && (
  <div className="preview-container">
    <img
      src={preview}
      alt="Preview"
      className="preview-image"
    />
  </div>
)}
 </section>
</section>
      {/* RESULTS */}

      {result && (

        <section className="result-card">

          <h2>Prediction Result</h2>

          <div className="result-grid">

            <div className="result-box">

              <h4>Disease</h4>

              <p>{result.disease}</p>

            </div>

            <div className="result-box">

              <h4>Confidence</h4>

              <p>{result.confidence}%</p>

            </div>

          </div>

        </section>

      )}

    </div>
  );
}

export default App;