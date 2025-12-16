import { useState, useEffect, useCallback } from 'react';
import { DetectionResult } from '../types';
import { storage, HistoryItem } from '../utils/storage';

export const useDetectionHistory = (enabled: boolean) => {
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    // Load initial history
    setHistory(storage.getHistory());
  }, []);

  const addDetection = useCallback(
    (result: DetectionResult) => {
      if (!enabled) return;
      storage.addToHistory(result);
      setHistory(storage.getHistory());
    },
    [enabled]
  );

  const clearHistory = useCallback(() => {
    storage.clearHistory();
    setHistory([]);
  }, []);

  const getStats = useCallback(() => {
    if (history.length === 0) {
      return {
        total: 0,
        averageConfidence: 0,
        mostFrequent: null as string | null,
        wordCounts: {} as Record<string, number>,
      };
    }

    const wordCounts: Record<string, number> = {};
    let totalConfidence = 0;

    history.forEach((item) => {
      wordCounts[item.detected_word] = (wordCounts[item.detected_word] || 0) + 1;
      totalConfidence += item.confidence;
    });

    const mostFrequent = Object.entries(wordCounts).reduce((a, b) =>
      wordCounts[a[0]] > wordCounts[b[0]] ? a : b
    )[0];

    return {
      total: history.length,
      averageConfidence: totalConfidence / history.length,
      mostFrequent,
      wordCounts,
    };
  }, [history]);

  return {
    history,
    addDetection,
    clearHistory,
    stats: getStats(),
  };
};


