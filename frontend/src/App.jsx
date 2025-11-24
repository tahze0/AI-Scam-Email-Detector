import { useState } from "react";
import { predictEmail } from "./api";

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
    <div style={{ maxWidth: 800, margin: "0 auto", padding: "2rem" }}>
      <h1>AI Scam Email Detector</h1>
      <p>Paste an email below and click Analyze to see if it looks like a scam.</p>

      <textarea
        style={{ width: "100%", height: "200px", marginTop: "1rem" }}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste email text here..."
      />

      <div style={{ marginTop: "1rem" }}>
        <button onClick={handleAnalyze} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Email"}
        </button>
      </div>

      {error && (
        <p style={{ color: "red", marginTop: "1rem" }}>{error}</p>
      )}

      {result && !error && (
        <div
          style={{
            marginTop: "2rem",
            padding: "1.5rem",
            border: "1px solid #ccc",
            borderRadius: "8px",
            background: result.is_scam ? "#fff5f5" : "#f0fff4"
          }}
        >
          <h2>Result: <span style={{ color: result.is_scam ? "red" : "green" }}>{result.prediction}</span></h2>
          <p>Combined Confidence: <strong>{result.confidence_percent}</strong></p>

          <hr style={{ margin: "1.5rem 0", opacity: 0.3 }} />

          {/*TF-IDF vs SBERT Breakdown */}
          <div style={{ display: "flex", gap: "20px", marginBottom: "20px" }}>
            
            {/* TF-IDF Model */}
            <div style={{ flex: 1 }}>
              <strong>Keyword Analysis (TF-IDF)</strong>
              <div style={{ background: "#ddd", height: "10px", borderRadius: "5px", marginTop: "5px" }}>
                <div style={{ 
                    width: result.breakdown.tfidf_score, 
                    background: "#2196f3", height: "100%", borderRadius: "5px" 
                }}></div>
              </div>
              <small>{result.breakdown.tfidf_score} Suspicious</small>
            </div>

            {/* SBERT Model */}
            <div style={{ flex: 1 }}>
              <strong>Context Analysis (SBERT)</strong>
              <div style={{ background: "#ddd", height: "10px", borderRadius: "5px", marginTop: "5px" }}>
                <div style={{ 
                    width: result.breakdown.sbert_score, 
                    background: "#9c27b0", height: "100%", borderRadius: "5px" 
                }}></div>
              </div>
              <small>{result.breakdown.sbert_score} Suspicious</small>
            </div>
          </div>

          {result.flagged_words && result.flagged_words.length > 0 && (
            <p>🚩 Flagged words: {result.flagged_words.join(", ")}</p>
          )}
        </div>
      )}
    </div>
  );
}

export default App;