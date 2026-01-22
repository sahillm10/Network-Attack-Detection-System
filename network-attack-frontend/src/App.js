import React, { useState, useRef } from 'react';
import { Upload, Send, Moon, Sun, Shield, AlertTriangle, CheckCircle, TrendingUp, X, FileText } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

const API_BASE = 'http://localhost:8000';

export default function NetworkAttackDashboard() {
  const [darkMode, setDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState('manual');
  const [manualInput, setManualInput] = useState('');
  
  // File state management
  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null); 

  const [prediction, setPrediction] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showMitigation, setShowMitigation] = useState(false);
  const [mitigation, setMitigation] = useState(null);

  const attackColors = {
    'spoofing': '#ef4444',
    'denial of service': '#f59e0b',
    'deauthentication': '#8b5cf6',
    'jamming': '#ec4899',
    'other': '#6b7280'
  };

  const handleManualPredict = async () => {
    if (!manualInput.trim()) return;
    
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict/manual?values=${encodeURIComponent(manualInput)}`, {
        method: 'POST'
      });
      
      if (!res.ok) {
        throw new Error(`Server Error: ${res.status} ${res.statusText}`);
      }

      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      alert('Error: ' + err.message);
    }
    setLoading(false);
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleBatchPredict = async () => {
    if (!file) {
      alert("Please select a file first.");
      return;
    }
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    // DEBUG: Log the file details explicitly
    console.log("Uploading file:", file.name, file.size, file.type);

    try {
      // NOTE: Do NOT set 'Content-Type' header manually when using FormData.
      // The browser automatically sets it with the correct boundary.
      const res = await fetch(`${API_BASE}/predict/batch`, {
        method: 'POST',
        body: formData
      });

      // 1. Check if response is OK before parsing JSON
      if (!res.ok) {
        const errorText = await res.text(); // Get raw text if JSON fails
        throw new Error(`Server returned ${res.status}: ${errorText}`);
      }

      const data = await res.json();
      setBatchResults(data.predictions || []);
      
      // Optional: Clear file after success
      // setFile(null);
      // if (fileInputRef.current) fileInputRef.current.value = "";

    } catch (err) {
      console.error("Upload Error:", err);
      alert('Upload Failed: ' + err.message);
    }
    setLoading(false);
  };

  const handleGetMitigation = async () => {
    if (!prediction) return;
    
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/mitigation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          attack_type: prediction.predicted_attack,
          top_features: prediction.lime_features?.slice(0, 5) || []
        })
      });
      
      if (!res.ok) throw new Error('Failed to fetch mitigation strategies');
      
      const data = await res.json();
      setMitigation(data);
      setShowMitigation(true);
    } catch (err) {
      alert('Error: ' + err.message);
    }
    setLoading(false);
  };

  const bgClass = darkMode ? 'bg-gray-900' : 'bg-gray-50';
  const cardClass = darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200';
  const textClass = darkMode ? 'text-gray-100' : 'text-gray-900';
  const textMuted = darkMode ? 'text-gray-400' : 'text-gray-600';

  return (
    <div className={`min-h-screen ${bgClass} ${textClass} transition-colors duration-300`}>
      {/* Header */}
      <header className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b sticky top-0 z-50`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold">Intelligent Network Attack Detection</h1>
                <p className={`text-sm ${textMuted}`}>Upload data or input features to detect, explain, and mitigate network attacks</p>
              </div>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'}`}
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveTab('manual')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'manual'
                ? 'bg-blue-500 text-white shadow-lg'
                : `${cardClass} border hover:border-blue-500`
            }`}
          >
            Manual Input
          </button>
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'upload'
                ? 'bg-blue-500 text-white shadow-lg'
                : `${cardClass} border hover:border-blue-500`
            }`}
          >
            Upload Dataset
          </button>
        </div>

        {/* Manual Input Tab */}
        {activeTab === 'manual' && (
          <div className="space-y-6">
            <div className={`${cardClass} border rounded-xl p-6 shadow-lg`}>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Send className="w-5 h-5 text-blue-500" />
                Enter Feature Values
              </h2>
              <p className={`text-sm ${textMuted} mb-4`}>
                Paste comma-separated feature values (e.g., 1.2, 0.5, -0.3, 2.1, ...)
              </p>
              <textarea
                value={manualInput}
                onChange={(e) => setManualInput(e.target.value)}
                placeholder="0.5, 1.2, -0.8, 3.4, 0.1, ..."
                className={`w-full h-32 px-4 py-3 rounded-lg border ${
                  darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-300'
                } focus:ring-2 focus:ring-blue-500 focus:border-transparent`}
              />
              <button
                onClick={handleManualPredict}
                disabled={loading || !manualInput.trim()}
                className="mt-4 px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {loading ? 'Analyzing...' : 'Predict Attack'}
              </button>
            </div>

            {/* Prediction Results */}
            {prediction && (
              <div className="space-y-6">
                {/* Attack Type Card */}
                <div className={`${cardClass} border rounded-xl p-6 shadow-lg`}>
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold">Detection Result</h2>
                    <AlertTriangle className="w-6 h-6" style={{ color: attackColors[prediction.predicted_attack] }} />
                  </div>
                  <div className="space-y-3">
                    <div>
                      <p className={`text-sm ${textMuted} mb-1`}>Detected Attack Type</p>
                      <p className="text-3xl font-bold capitalize" style={{ color: attackColors[prediction.predicted_attack] }}>
                        {prediction.predicted_attack}
                      </p>
                    </div>
                    <div>
                      <p className={`text-sm ${textMuted} mb-1`}>Confidence Score</p>
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-3 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-500"
                            style={{ width: `${prediction.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-2xl font-bold">{(prediction.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Class Probabilities */}
                {prediction.class_probabilities && (
                  <div className={`${cardClass} border rounded-xl p-6 shadow-lg`}>
                    <h3 className="text-lg font-semibold mb-4">Class Probabilities</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={Object.entries(prediction.class_probabilities).map(([k, v]) => ({ name: k, probability: v * 100 }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#374151' : '#e5e7eb'} />
                        <XAxis dataKey="name" tick={{ fill: darkMode ? '#9ca3af' : '#6b7280' }} />
                        <YAxis tick={{ fill: darkMode ? '#9ca3af' : '#6b7280' }} />
                        <Tooltip contentStyle={{ backgroundColor: darkMode ? '#1f2937' : '#fff', border: 'none', borderRadius: '8px' }} />
                        <Bar dataKey="probability" radius={[8, 8, 0, 0]}>
                          {Object.keys(prediction.class_probabilities).map((key, index) => (
                            <Cell key={index} fill={attackColors[key] || '#6b7280'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* LIME Explanation */}
                {prediction.lime_features && prediction.lime_features.length > 0 && (
                  <div className={`${cardClass} border rounded-xl p-6 shadow-lg`}>
                    <h3 className="text-lg font-semibold mb-4">Feature Impact (LIME Explanation)</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={prediction.lime_features} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#374151' : '#e5e7eb'} />
                        <XAxis type="number" tick={{ fill: darkMode ? '#9ca3af' : '#6b7280' }} />
                        <YAxis dataKey="feature" type="category" width={100} tick={{ fill: darkMode ? '#9ca3af' : '#6b7280' }} />
                        <Tooltip contentStyle={{ backgroundColor: darkMode ? '#1f2937' : '#fff', border: 'none', borderRadius: '8px' }} />
                        <Bar dataKey="contribution" radius={[0, 8, 8, 0]}>
                          {prediction.lime_features.map((entry, index) => (
                            <Cell key={index} fill={entry.contribution > 0 ? '#10b981' : '#ef4444'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Top Original Features */}
                {prediction.top_original_features && prediction.top_original_features.length > 0 && (
                  <div className={`${cardClass} border rounded-xl p-6 shadow-lg`}>
                    <h3 className="text-lg font-semibold mb-4">Top Original Features</h3>
                    <div className="space-y-3">
                      {prediction.top_original_features.slice(0, 5).map((feat, idx) => (
                        <div key={idx} className="flex items-center gap-3">
                          <TrendingUp className="w-5 h-5 text-blue-500" />
                          <span className="flex-1 font-medium">{feat.feature}</span>
                          <span className={`${textMuted} text-sm`}>{feat.importance.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Mitigation Button */}
                <button
                  onClick={handleGetMitigation}
                  disabled={loading}
                  className="w-full px-6 py-4 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white rounded-xl font-medium shadow-lg disabled:opacity-50 transition-all"
                >
                  {loading ? 'Generating Mitigation Plan...' : 'View Mitigation Steps'}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Upload Dataset Tab */}
        {activeTab === 'upload' && (
          <div className="space-y-6">
            <div className={`${cardClass} border rounded-xl p-6 shadow-lg`}>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5 text-blue-500" />
                Upload Dataset
              </h2>
              
              <div className="space-y-3">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx"
                  onChange={handleFileChange}
                  className={`w-full px-4 py-3 rounded-lg border ${
                    darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-300'
                  }`}
                />
                
                {/* Visual Feedback for selected file */}
                {file && (
                  <div className={`flex items-center gap-2 p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-blue-50 text-blue-800'}`}>
                    <FileText className="w-5 h-5" />
                    <span className="text-sm font-medium">Ready to upload: {file.name}</span>
                  </div>
                )}
              </div>

              <button
                onClick={handleBatchPredict}
                disabled={loading || !file}
                className="mt-4 px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {loading ? 'Processing...' : 'Run Batch Prediction'}
              </button>
            </div>

            {/* Batch Results */}
            {batchResults.length > 0 && (
              <div className={`${cardClass} border rounded-xl p-6 shadow-lg overflow-x-auto`}>
                <h3 className="text-lg font-semibold mb-4">Batch Results ({batchResults.length} rows)</h3>
                <table className="w-full">
                  <thead>
                    <tr className={`border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                      <th className="text-left py-3 px-4">Row #</th>
                      <th className="text-left py-3 px-4">Predicted Attack</th>
                      <th className="text-left py-3 px-4">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResults.map((result, idx) => (
                      <tr key={idx} className={`border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <td className="py-3 px-4">{idx + 1}</td>
                        <td className="py-3 px-4 capitalize font-medium" style={{ color: attackColors[result.predicted_attack] }}>
                          {result.predicted_attack}
                        </td>
                        <td className="py-3 px-4">{(result.confidence * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Mitigation Modal */}
      {showMitigation && mitigation && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className={`${cardClass} rounded-2xl max-w-2xl w-full max-h-[80vh] overflow-y-auto shadow-2xl`}>
            <div className="sticky top-0 bg-gradient-to-r from-purple-500 to-pink-500 p-6 rounded-t-2xl">
              <div className="flex items-center justify-between text-white">
                <h2 className="text-2xl font-bold">Mitigation Plan for {mitigation.attack_type}</h2>
                <button onClick={() => setShowMitigation(false)} className="p-2 hover:bg-white/20 rounded-lg transition-colors">
                  <X className="w-6 h-6" />
                </button>
              </div>
            </div>
            <div className="p-6 space-y-4">
              {mitigation.mitigations && mitigation.mitigations.length > 0 ? (
                mitigation.mitigations.map((m, idx) => (
                  <div key={idx} className={`p-4 rounded-lg border ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'}`}>
                    <div className="flex items-start gap-3">
                      <CheckCircle className="w-5 h-5 text-green-500 mt-1" />
                      <div>
                        <p className="font-semibold text-lg">{m.feature}</p>
                        <p className={textMuted}>{m.recommended_change}</p>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="whitespace-pre-wrap">{mitigation.raw_text || 'No mitigation steps available'}</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}