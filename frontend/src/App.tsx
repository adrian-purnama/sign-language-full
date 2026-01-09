import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { VideoCapture } from './components/VideoCapture';
import { DetectionDisplay } from './components/DetectionDisplay';
import { DetectionHistory } from './components/DetectionHistory';
import { SettingsPanel } from './components/SettingsPanel';
import { StatsPanel } from './components/StatsPanel';
import { VideoCanvas } from './components/VideoCanvas';
import { ChatPanel, ChatMessage } from './components/ChatPanel';
import { WebSocketClient } from './websocket';
import { DetectionResult } from './types';
import { useDetectionHistory } from './hooks/useDetectionHistory';
import { storage, AppSettings } from './utils/storage';
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
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);

  const { history, addDetection, clearHistory, stats } = useDetectionHistory(
    settings.recordHistory
  );

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

  const handleResult = useCallback((data: DetectionResult) => {
    setResult(data);
    setError(null);
    
    if (
      settings.recordHistory &&
      data.detected_word !== '...' &&
      data.confidence >= settings.confidenceThreshold
    ) {
      addDetection(data);
    }
  }, [settings.recordHistory, settings.confidenceThreshold, addDetection]);

  const handleFrame = useCallback((frameData: string) => {
    if (wsClientRef.current && wsClientRef.current.isConnected()) {
      wsClientRef.current.sendFrame(frameData);
    }
  }, []);

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

  const handleChatMessage = useCallback((data: { text: string; sender: string; timestamp: number }) => {
    const newMessage: ChatMessage = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      text: data.text,
      timestamp: data.timestamp || Date.now(),
      sender: data.sender === 'disabled' ? 'disabled' : 'user',
    };
    setChatMessages((prev) => [...prev, newMessage]);
  }, []);

  const handleSendMessage = useCallback((message: string) => {
    if (wsClientRef.current && wsClientRef.current.isConnected() && !settings.isDisabled) {
      wsClientRef.current.sendMessage(message);
    }
  }, [settings.isDisabled]);

  const handleVideoReady = useCallback((video: HTMLVideoElement) => {
    setVideoElement(video);
    setVideoDimensions({
      width: video.videoWidth || 640,
      height: video.videoHeight || 480,
    });
  }, []);

  useEffect(() => {
    const client = new WebSocketClient();
    wsClientRef.current = client;

    client.connect(WS_URL, handleResult, handleError, handleConnect, handleDisconnect, handleChatMessage);

    return () => {
      client.disconnect();
    };
  }, [handleResult, handleError, handleConnect, handleDisconnect, handleChatMessage]);

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
          <span
            className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}
          />
          <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
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
                videoElement={videoElement}
              />
            )}
          </div>
        </div>

        <div className="results-section">
          <DetectionDisplay result={result} />
          
          <ChatPanel
            isDisabled={settings.isDisabled}
            onSendMessage={handleSendMessage}
            messages={chatMessages}
          />
          
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
