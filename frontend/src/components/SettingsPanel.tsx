import React, { useState, useEffect } from 'react';
import { AppSettings } from '../utils/storage';
import { FaCog, FaVideo, FaEye, FaSquare, FaHistory, FaChartLine, FaUndo, FaExclamationTriangle, FaKey } from 'react-icons/fa';
import '../styles/animations.css';

interface SettingsPanelProps {
  settings: AppSettings;
  onSettingsChange: (settings: Partial<AppSettings>) => void;
  onReset: () => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = React.memo(({
  settings,
  onSettingsChange,
  onReset,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleFpsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onSettingsChange({ fps: parseInt(e.target.value, 10) });
  };

  const handleToggle = (key: keyof AppSettings) => {
    onSettingsChange({ [key]: !settings[key] });
  };

  return (
    <div className={`settings-panel ${isExpanded ? 'expanded' : ''}`}>
      <div className="settings-header" onClick={() => setIsExpanded(!isExpanded)}>
        <h3>
          <FaCog className="header-icon" />
          Settings
        </h3>
        <button className="expand-button">
          {isExpanded ? '−' : '+'}
        </button>
      </div>

      {isExpanded && (
        <div className="settings-content animate-fade-in">
          <div className="setting-group">
            <label className="setting-label">
              <FaVideo />
              Frame Rate: {settings.fps} FPS
            </label>
            <input
              type="range"
              min="5"
              max="30"
              value={settings.fps}
              onChange={handleFpsChange}
              className="setting-slider"
            />
            <div className="setting-hint">
              Lower FPS = less network usage, Higher FPS = smoother detection
            </div>
          </div>

          <div className="setting-group">
            <label className="setting-toggle">
              <span>
                <FaEye />
                Show Keypoints Overlay
              </span>
              <input
                type="checkbox"
                checked={settings.showKeypoints}
                onChange={() => handleToggle('showKeypoints')}
                className="toggle-input"
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="setting-group">
            <label className="setting-toggle">
              <span>
                <FaSquare />
                Show Bounding Box
              </span>
              <input
                type="checkbox"
                checked={settings.showBoundingBox}
                onChange={() => handleToggle('showBoundingBox')}
                className="toggle-input"
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="setting-group">
            <label className="setting-toggle">
              <span>
                <FaHistory />
                Record Detection History
              </span>
              <input
                type="checkbox"
                checked={settings.recordHistory}
                onChange={() => handleToggle('recordHistory')}
                className="toggle-input"
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="setting-group">
            <label className="setting-label">
              <FaChartLine />
              Confidence Threshold: {(settings.confidenceThreshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.3"
              max="0.95"
              step="0.05"
              value={settings.confidenceThreshold}
              onChange={(e) =>
                onSettingsChange({ confidenceThreshold: parseFloat(e.target.value) })
              }
              className="setting-slider"
            />
            <div className="setting-hint">
              Minimum confidence level to display detected words
            </div>
          </div>

          <div className="setting-group" style={{ borderTop: '1px solid var(--glass-border)', paddingTop: '16px', marginTop: '8px' }}>
            <label className="setting-toggle">
              <span>
                <FaExclamationTriangle />
                Emergency Mode (Gemini AI)
              </span>
              <input
                type="checkbox"
                checked={settings.emergencyMode}
                onChange={() => handleToggle('emergencyMode')}
                className="toggle-input"
              />
              <span className="toggle-slider"></span>
            </label>
            <div className="setting-hint" style={{ marginTop: '8px' }}>
              Uses Google Gemini for real-time translation (requires API key)
            </div>
          </div>

          {settings.emergencyMode && (
            <div className="setting-group">
              <label className="setting-label">
                <FaKey />
                Gemini API Key
              </label>
              <input
                type="password"
                value={settings.geminiApiKey}
                onChange={(e) => onSettingsChange({ geminiApiKey: e.target.value })}
                placeholder="Enter your Gemini API key"
                style={{
                  width: '100%',
                  padding: '10px',
                  borderRadius: '8px',
                  border: '1px solid var(--glass-border)',
                  fontSize: '0.9rem',
                  fontFamily: 'monospace',
                  marginTop: '8px'
                }}
              />
              <div className="setting-hint" style={{ marginTop: '4px' }}>
                Get your key from: https://makersuite.google.com/app/apikey
              </div>
            </div>
          )}

          <button className="reset-button" onClick={onReset}>
            <FaUndo />
            Reset to Defaults
          </button>
        </div>
      )}
    </div>
  );
});

