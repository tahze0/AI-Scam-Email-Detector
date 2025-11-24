import { useState } from "react";
import { predictEmail } from "./api";
import "./App.css"; 

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    setError("");
    setResult(null);

    const trimmed = text.trim();
    if (!trimmed) {
      setError("Please paste an email to analyze.");
      return;
    }

    try {
      setLoading(true);
      const res = await predictEmail(trimmed);
      if (res.error) {
        setError(res.error);
      } else {
        setResult(res);
      }
    } catch (err) {
      console.error(err);
      setError("Could not reach the server. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="card">
        <div className="header">
          <h1>AI Scam Email Detector</h1>
          <p>Paste an email below to check for phishing indicators</p>
        </div>

        <textarea
          className="email-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Subject: Urgent Request..."
        />

        <button 
          className="analyze-btn" 
          onClick={handleAnalyze} 
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Analyze Email"}
        </button>

        {error && (
          <div className="error-msg">{error}</div>
        )}

        {result && !error && (
          <div className={`result-container ${result.is_scam ? "scam" : "safe"}`}>
            <div className="result-header">
              <div>
                <span style={{color: '#a1a1aa', fontSize: '0.9rem'}}>Prediction</span>
                <div className="result-badge">{result.prediction}</div>
              </div>
              <div style={{textAlign: 'right'}}>
                <span style={{color: '#a1a1aa', fontSize: '0.9rem'}}>Combined Confidence</span>
                <div style={{fontSize: '1.25rem', fontWeight: 'bold', color: '#fff'}}>
                  {result.confidence_percent}
                </div>
              </div>
            </div>

            <hr style={{ borderColor: '#3f3f46', margin: "1.5rem 0" }} />

            {/* Breakdown Section */}
            <div className="breakdown-grid">
              {/* TF-IDF Bar */}
              <div>
                <span className="progress-label">Keyword Analysis (TF-IDF)</span>
                <div className="progress-bg">
                  <div 
                    className="progress-fill" 
                    style={{ width: result.breakdown.tfidf_score, backgroundColor: "#3b82f6" }}
                  ></div>
                </div>
                <span className="meta-text">{result.breakdown.tfidf_score} Suspicious</span>
              </div>

              {/* SBERT Bar */}
              <div>
                <span className="progress-label">Context Analysis (SBERT)</span>
                <div className="progress-bg">
                  <div 
                    className="progress-fill" 
                    style={{ width: result.breakdown.sbert_score, backgroundColor: "#a855f7" }}
                  ></div>
                </div>
                <span className="meta-text">{result.breakdown.sbert_score} Suspicious</span>
              </div>
            </div>

            {result.flagged_words && result.flagged_words.length > 0 && (
              <div style={{marginTop: '1.5rem'}}>
                <span className="progress-label" style={{color: '#ef4444'}}>🚩 Flagged words: </span>
                <p style={{ marginTop: "0.25rem", color: "#d4d4d8" }}>
                  {result.flagged_words.join(", ")}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;