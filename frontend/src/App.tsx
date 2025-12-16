import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { VideoCapture } from './components/VideoCapture';
import { DetectionDisplay } from './components/DetectionDisplay';
import { DetectionHistory } from './components/DetectionHistory';
import { SettingsPanel } from './components/SettingsPanel';
import { StatsPanel } from './components/StatsPanel';
import { VideoCanvas } from './components/VideoCanvas';
import { WebSocketClient } from './websocket';
import { DetectionResult } from './types';
import { useDetectionHistory } from './hooks/useDetectionHistory';
import { storage, AppSettings } from './utils/storage';
import { translateWithGemini } from './utils/gemini';
import './App.css';

const WS_URL = 'http://localhost:5000';

function App() {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const [settings, setSettings] = useState<AppSettings>(storage.getSettings());
  const wsClientRef = useRef<WebSocketClient | null>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 640, height: 480 });
  const geminiProcessingRef = useRef(false);
  const lastGeminiCallRef = useRef<number>(0);

  const { history, addDetection, clearHistory, stats } = useDetectionHistory(
    settings.recordHistory
  );

  // Handle settings changes
  const handleSettingsChange = useCallback((newSettings: Partial<AppSettings>) => {
    const updated = { ...settings, ...newSettings };
    setSettings(updated);
    storage.saveSettings(updated);
  }, [settings]);

  const handleResetSettings = useCallback(() => {
    storage.resetSettings();
    const defaults = storage.getSettings();
    setSettings(defaults);
  }, []);

  // Handle detection results
  const handleResult = useCallback((data: DetectionResult) => {
    setResult(data);
    setError(null);
    
    // Only add to history if confidence meets threshold and word is detected
    if (
      settings.recordHistory &&
      data.detected_word !== '...' &&
      data.confidence >= settings.confidenceThreshold
    ) {
      addDetection(data);
    }
  }, [settings.recordHistory, settings.confidenceThreshold, addDetection]);

  // Handle frame capture with settings
  const handleFrame = useCallback(async (frameData: string) => {
    // Emergency mode: use Gemini API
    if (settings.emergencyMode) {
      // if (!settings.geminiApiKey || settings.geminiApiKey.trim() === '') {
      //   setError('Please enter your Gemini API key in Settings');
      //   return;
      // }

      // Throttle Gemini calls to avoid rate limits (max 1 call per 6 seconds for 10 RPM limit)
      const now = Date.now();
      if (now - lastGeminiCallRef.current < 6000 || geminiProcessingRef.current) {
        return;
      }
      
      geminiProcessingRef.current = true;
      lastGeminiCallRef.current = now;
      
      try {
        const geminiResult = await translateWithGemini(frameData, settings.geminiApiKey.trim());
        
        // Calculate bounding box for center area of video (where sign language typically happens)
        // Use 70% of width and 80% of height, centered
        const videoWidth = videoDimensions.width || 640;
        const videoHeight = videoDimensions.height || 480;
        const boxWidth = videoWidth * 0.7;
        const boxHeight = videoHeight * 0.8;
        const x1 = (videoWidth - boxWidth) / 2;
        const y1 = (videoHeight - boxHeight) / 2;
        const x2 = x1 + boxWidth;
        const y2 = y1 + boxHeight;
        
        const emergencyResult: DetectionResult = {
          detected_word: geminiResult.translation,
          confidence: geminiResult.confidence || 0.85,
          bounding_box: [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)],
          keypoints: { pose: null, left_hand: null, right_hand: null },
          sequence_length: 30, // Always ready in emergency mode
        };
        
        handleResult(emergencyResult);
      } catch (error) {
        console.error('Gemini translation error:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        setError(`Gemini Error: ${errorMessage}`);
        // Show error result
        handleResult({
          detected_word: '...',
          confidence: 0.0,
          bounding_box: [0, 0, 0, 0],
          keypoints: { pose: null, left_hand: null, right_hand: null },
          sequence_length: 0,
        });
      } finally {
        geminiProcessingRef.current = false;
      }
      return;
    }
    
    // Normal mode: use WebSocket
    if (wsClientRef.current && wsClientRef.current.isConnected()) {
      wsClientRef.current.sendFrame(frameData);
    }
  }, [settings.emergencyMode, settings.geminiApiKey, handleResult]);

  const handleError = useCallback((err: string) => {
    setError(err);
    console.error('WebSocket error:', err);
  }, []);

  const handleConnect = useCallback(() => {
    setIsConnected(true);
    setError(null);
  }, []);

  const handleDisconnect = useCallback(() => {
    setIsConnected(false);
  }, []);

  const handleVideoReady = useCallback((video: HTMLVideoElement) => {
    setVideoElement(video);
    setVideoDimensions({
      width: video.videoWidth || 640,
      height: video.videoHeight || 480,
    });
  }, []);

  // Initialize WebSocket only when emergency mode is OFF
  useEffect(() => {
    // Don't connect WebSocket if emergency mode is enabled
    if (settings.emergencyMode) {
      // Disconnect if already connected
      if (wsClientRef.current) {
        wsClientRef.current.disconnect();
        wsClientRef.current = null;
      }
      setIsConnected(false);
      return;
    }

    // Connect WebSocket for normal mode
    const client = new WebSocketClient();
    wsClientRef.current = client;

    client.connect(WS_URL, handleResult, handleError, handleConnect, handleDisconnect);

    return () => {
      client.disconnect();
    };
  }, [settings.emergencyMode, handleResult, handleError, handleConnect, handleDisconnect]);

  // Update video dimensions
  useEffect(() => {
    if (!videoElement) return;

    const updateDimensions = () => {
      const container = videoContainerRef.current;
      if (container && videoElement.videoWidth > 0) {
        const containerWidth = container.clientWidth;
        const aspectRatio = videoElement.videoHeight / videoElement.videoWidth;
        setVideoDimensions({
          width: containerWidth,
          height: containerWidth * aspectRatio,
        });
      }
    };

    videoElement.addEventListener('loadedmetadata', updateDimensions);
    window.addEventListener('resize', updateDimensions);
    updateDimensions();

    return () => {
      videoElement.removeEventListener('loadedmetadata', updateDimensions);
      window.removeEventListener('resize', updateDimensions);
    };
  }, [videoElement]);

  // Memoize video capture component props
  const videoCaptureProps = useMemo(() => ({
    onFrame: handleFrame,
    onVideoReady: handleVideoReady,
    fps: settings.fps,
  }), [handleFrame, handleVideoReady, settings.fps]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Sign Language Translation</h1>
        <div className="connection-status">
          {settings.emergencyMode ? (
            <>
              <span className="status-indicator connected" />
              <span>Connected</span>
            </>
          ) : (
            <>
              <span
                className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}
              />
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </>
          )}
        </div>
      </header>

      <main className="App-main">
        <div className="video-section">
          <div ref={videoContainerRef} className="video-container">
            <VideoCapture {...videoCaptureProps} />
            {videoElement && result && (
              <VideoCanvas
                result={result}
                width={videoDimensions.width}
                height={videoDimensions.height}
              />
            )}
          </div>
        </div>

        <div className="results-section">
          <DetectionDisplay result={result} />
          
          <StatsPanel stats={stats} />
          
          <DetectionHistory history={history} onClear={clearHistory} />
          
          <SettingsPanel
            settings={settings}
            onSettingsChange={handleSettingsChange}
            onReset={handleResetSettings}
          />

          {error && (
            <div className="error-message">
              Error: {error}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
