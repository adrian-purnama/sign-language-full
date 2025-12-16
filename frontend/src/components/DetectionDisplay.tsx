import React, { useEffect, useRef } from 'react';
import { DetectionResult } from '../types';
import { FaCheckCircle, FaSpinner } from 'react-icons/fa';
import '../styles/animations.css';

interface DetectionDisplayProps {
  result: DetectionResult | null;
}

export const DetectionDisplay: React.FC<DetectionDisplayProps> = React.memo(({ result }) => {
  const word = result?.detected_word || '...';
  const confidence = result?.confidence || 0;
  const sequenceLength = result?.sequence_length || 0;
  const prevWordRef = useRef<string>('...');
  const wordRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (word !== prevWordRef.current && word !== '...') {
      prevWordRef.current = word;
      if (wordRef.current) {
        wordRef.current.classList.add('word-pop');
        setTimeout(() => {
          wordRef.current?.classList.remove('word-pop');
        }, 500);
      }
    }
  }, [word]);

  const getConfidenceColor = () => {
    if (confidence > 0.7) return '#00e676';
    if (confidence > 0.4) return '#ffb300';
    return '#ff5252';
  };

  const getConfidenceGradient = () => {
    if (confidence > 0.7) return 'linear-gradient(90deg, #00e676 0%, #00ff88 100%)';
    if (confidence > 0.4) return 'linear-gradient(90deg, #ffb300 0%, #ffc947 100%)';
    return 'linear-gradient(90deg, #ff5252 0%, #ff7979 100%)';
  };

  return (
    <div className="detection-display">
      <div className="detection-header">
        <h2>
          <FaCheckCircle className="header-icon" />
          Detection Results
        </h2>
      </div>

      <div className="detection-content">
        <div className="word-section">
          <div className="word-label">Detected Word</div>
          <div
            ref={wordRef}
            className="detected-word"
            style={{
              color: word !== '...' ? getConfidenceColor() : '#888',
            }}
          >
            {word === '...' ? (
              <div className="waiting-state">
                <FaSpinner className="spinner" />
                <span>Waiting for detection...</span>
              </div>
            ) : (
              word.toUpperCase()
            )}
          </div>
        </div>

        <div className="confidence-section">
          <div className="confidence-header">
            <span className="confidence-label">Confidence</span>
            <span className="confidence-value">{Math.round(confidence * 100)}%</span>
          </div>
          <div className="progress-container">
            <div
              className="progress-bar"
              style={{
                width: `${confidence * 100}%`,
                background: getConfidenceGradient(),
              }}
            >
              <div className="progress-shine"></div>
            </div>
          </div>
        </div>

        <div className="sequence-section">
          <div className="sequence-header">
            <span className="sequence-label">Sequence Buffer</span>
            <span className="sequence-value">{sequenceLength} / 30</span>
          </div>
          <div className="sequence-container">
            <div
              className="sequence-bar"
              style={{
                width: `${(sequenceLength / 30) * 100}%`,
                backgroundColor: sequenceLength === 30 ? '#4CAF50' : '#2196F3',
              }}
            />
          </div>
          {sequenceLength < 30 && (
            <div className="sequence-hint">
              Collecting frames... ({30 - sequenceLength} remaining)
            </div>
          )}
        </div>
      </div>
    </div>
  );
});
