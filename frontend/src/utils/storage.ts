import { DetectionResult } from '../types';

export interface HistoryItem extends DetectionResult {
  timestamp: number;
  id: string;
}

const HISTORY_KEY = 'sign-language-history';
const SETTINGS_KEY = 'sign-language-settings';
const MAX_HISTORY_ITEMS = 50;

export interface AppSettings {
  fps: number;
  showKeypoints: boolean;
  showBoundingBox: boolean;
  recordHistory: boolean;
  confidenceThreshold: number;
  emergencyMode: boolean;
  geminiApiKey: string;
}

const defaultSettings: AppSettings = {
  fps: 15,
  showKeypoints: true,
  showBoundingBox: true,
  recordHistory: true,
  confidenceThreshold: 0.7,
  emergencyMode: false,
  geminiApiKey: '',
};

export const storage = {
  // History Management
  getHistory(): HistoryItem[] {
    try {
      const stored = localStorage.getItem(HISTORY_KEY);
      if (!stored) return [];
      const history = JSON.parse(stored);
      return Array.isArray(history) ? history : [];
    } catch (error) {
      console.error('Error loading history:', error);
      return [];
    }
  },

  addToHistory(result: DetectionResult): void {
    try {
      if (result.detected_word === '...' || result.confidence < 0.5) {
        return; // Don't store low confidence or placeholder results
      }

      const history = this.getHistory();
      const newItem: HistoryItem = {
        ...result,
        timestamp: Date.now(),
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      };

      const updatedHistory = [newItem, ...history].slice(0, MAX_HISTORY_ITEMS);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(updatedHistory));
    } catch (error) {
      console.error('Error saving to history:', error);
    }
  },

  clearHistory(): void {
    try {
      localStorage.removeItem(HISTORY_KEY);
    } catch (error) {
      console.error('Error clearing history:', error);
    }
  },

  // Settings Management
  getSettings(): AppSettings {
    try {
      const stored = localStorage.getItem(SETTINGS_KEY);
      if (!stored) return defaultSettings;
      const settings = JSON.parse(stored);
      return { ...defaultSettings, ...settings };
    } catch (error) {
      console.error('Error loading settings:', error);
      return defaultSettings;
    }
  },

  saveSettings(settings: Partial<AppSettings>): void {
    try {
      const currentSettings = this.getSettings();
      const updatedSettings = { ...currentSettings, ...settings };
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(updatedSettings));
    } catch (error) {
      console.error('Error saving settings:', error);
    }
  },

  resetSettings(): void {
    try {
      localStorage.removeItem(SETTINGS_KEY);
    } catch (error) {
      console.error('Error resetting settings:', error);
    }
  },
};


