import React, { useMemo } from 'react';
import { HistoryItem } from '../utils/storage';
import { FaHistory, FaTrash, FaClock, FaCheckCircle } from 'react-icons/fa';
import '../styles/animations.css';

interface DetectionHistoryProps {
  history: HistoryItem[];
  onClear: () => void;
}

export const DetectionHistory: React.FC<DetectionHistoryProps> = React.memo(({ history, onClear }) => {
  const recentHistory = useMemo(() => history.slice(0, 20), [history]);

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);

    if (diffSecs < 60) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.7) return '#00e676';
    if (confidence > 0.4) return '#ffb300';
    return '#ff5252';
  };

  return (
    <div className="history-panel">
      <div className="history-header">
        <h3>
          <FaHistory className="header-icon" />
          Detection History
        </h3>
        {history.length > 0 && (
          <button className="clear-button" onClick={onClear} title="Clear history">
            <FaTrash />
          </button>
        )}
      </div>

      <div className="history-content">
        {recentHistory.length === 0 ? (
          <div className="history-empty">
            <FaClock className="empty-icon" />
            <p>No detections yet</p>
            <span>Recent detections will appear here</span>
          </div>
        ) : (
          <div className="history-list">
            {recentHistory.map((item, index) => (
              <div
                key={item.id}
                className="history-item animate-fade-in"
                style={{ animationDelay: `${index * 0.05}s` }}
              >
                <div className="history-item-main">
                  <div className="history-word">
                    <FaCheckCircle
                      className="history-icon"
                      style={{ color: getConfidenceColor(item.confidence) }}
                    />
                    <span className="history-word-text">{item.detected_word.toUpperCase()}</span>
                  </div>
                  <div className="history-meta">
                    <span
                      className="history-confidence"
                      style={{ color: getConfidenceColor(item.confidence) }}
                    >
                      {Math.round(item.confidence * 100)}%
                    </span>
                    <span className="history-time">
                      <FaClock />
                      {formatTime(item.timestamp)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {history.length > 20 && (
        <div className="history-footer">
          Showing 20 of {history.length} detections
        </div>
      )}
    </div>
  );
});

