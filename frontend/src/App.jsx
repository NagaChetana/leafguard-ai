import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
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
    setErrorMsg(null);
  };

  const predictDisease = async () => {
    if (!image) {
      alert("Please select an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    setLoading(true);
    setResult(null);
    setErrorMsg(null);

    const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";

    try {
      const response = await axios.post(`${apiUrl}/predict`, formData);

      if (response.data.is_valid === false || response.data.error) {
        setErrorMsg(response.data.error || "Invalid image. Please upload a valid plant leaf image.");
        setResult(null);
      } else {
        setResult(response.data);
        setErrorMsg(null);
      }
    } catch (error) {
      console.error(error);
      setErrorMsg("Prediction server connection failed. Please ensure backend is running.");
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

        <h2>🌿 Analyze Your Plant</h2>

        <p className="upload-subtitle">
          Upload a clear, close-up photo of one healthy or infected leaf. Avoid pots, soil, and wide plant photos.
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

      {/* ERROR DISPLAY */}

      {errorMsg && (
        <section className="error-card">
          <div className="error-icon">⚠️</div>
          <div className="error-content">
            <h4>Validation Error</h4>
            <p>{errorMsg}</p>
          </div>
        </section>
      )}

      {/* RESULTS */}

      {result && result.is_valid !== false && (

        <section className={`result-card ${result.is_healthy ? "healthy-card" : "diseased-card"}`}>

          <h2>{result.needs_review ? "🔎 Plant Analysis Needs Review" : result.is_healthy ? "🌿 Plant Analysis Result" : "⚠️ Disease Detection Result"}</h2>

          <div className="result-grid">

            <div className="result-box">

              <h4>Plant Status</h4>

              <p className={result.is_healthy ? "status-healthy" : "status-diseased"}>
                {result.needs_review ? "Needs Review 🔎" : result.is_healthy ? "Healthy Plant 🌿" : "Diseased ⚠️"}
              </p>

            </div>

            <div className="result-box">

              <h4>{result.needs_review ? "Result" : result.is_healthy ? "Plant Type" : "Detected Disease"}</h4>

              <p className="result-disease-text">
                {result.is_healthy ? (result.plant || "Plant") : result.disease}
              </p>

            </div>

            <div className="result-box">

              <h4>Confidence</h4>

              <p>{result.confidence}%</p>

            </div>

          </div>

          {result.is_healthy && (
            <p className="healthy-note">
              ✅ No disease detected! Your plant appears healthy and vibrant.
            </p>
          )}

          {result.needs_review && (
            <p className="healthy-note">
              ℹ️ The image looks like a leaf, but the model cannot distinguish a disease confidently. Try a closer, well-lit photo of one leaf.
            </p>
          )}

        </section>

      )}

    </div>
  );
}

export default App;
