import React from 'react';
import { FaChartBar, FaTrophy, FaBullseye, FaList } from 'react-icons/fa';
import '../styles/animations.css';

interface StatsPanelProps {
  stats: {
    total: number;
    averageConfidence: number;
    mostFrequent: string | null;
    wordCounts: Record<string, number>;
  };
}

export const StatsPanel: React.FC<StatsPanelProps> = React.memo(({ stats }) => {
  const topWords = Object.entries(stats.wordCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return (
    <div className="stats-panel">
      <div className="stats-header">
        <h3>
          <FaChartBar className="header-icon" />
          Statistics
        </h3>
      </div>

      <div className="stats-content">
        <div className="stat-card">
          <div className="stat-icon" style={{ backgroundColor: 'rgba(0, 212, 255, 0.2)' }}>
            <FaList style={{ color: '#00d4ff' }} />
          </div>
          <div className="stat-info">
            <div className="stat-value">{stats.total}</div>
            <div className="stat-label">Total Detections</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ backgroundColor: 'rgba(0, 230, 118, 0.2)' }}>
            <FaBullseye style={{ color: '#00e676' }} />
          </div>
          <div className="stat-info">
            <div className="stat-value">{stats.averageConfidence.toFixed(1)}%</div>
            <div className="stat-label">Avg Confidence</div>
          </div>
        </div>

        {stats.mostFrequent && (
        <div className="stat-card">
          <div className="stat-icon" style={{ backgroundColor: 'rgba(255, 179, 0, 0.2)' }}>
            <FaTrophy style={{ color: '#ffb300' }} />
          </div>
            <div className="stat-info">
              <div className="stat-value">{stats.mostFrequent.toUpperCase()}</div>
              <div className="stat-label">Most Frequent</div>
            </div>
          </div>
        )}

        {topWords.length > 0 && (
          <div className="top-words">
            <div className="top-words-title">Top Words</div>
            <div className="top-words-list">
              {topWords.map(([word, count], index) => (
                <div key={word} className="top-word-item">
                  <span className="top-word-rank">#{index + 1}</span>
                  <span className="top-word-name">{word.toUpperCase()}</span>
                  <span className="top-word-count">{count}x</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {stats.total === 0 && (
          <div className="stats-empty">
            <FaChartBar className="empty-icon" />
            <p>No statistics available yet</p>
          </div>
        )}
      </div>
    </div>
  );
});

