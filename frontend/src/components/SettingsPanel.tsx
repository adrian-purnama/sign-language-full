import React, { useState, useEffect } from 'react';
import { AppSettings } from '../utils/storage';
import { FaCog, FaVideo, FaEye, FaHistory, FaChartLine, FaUndo, FaUser } from 'react-icons/fa';
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

          <div className="setting-group">
            <label className="setting-toggle">
              <span>
                <FaUser />
                I am a disabled user (sign language user)
              </span>
              <input
                type="checkbox"
                checked={settings.isDisabled}
                onChange={() => handleToggle('isDisabled')}
                className="toggle-input"
              />
              <span className="toggle-slider"></span>
            </label>
            <div className="setting-hint">
              {settings.isDisabled 
                ? 'You will receive messages from others' 
                : 'You can send messages to disabled users'}
            </div>
          </div>

          <button className="reset-button" onClick={onReset}>
            <FaUndo />
            Reset to Defaults
          </button>
        </div>
      )}
    </div>
  );
});

