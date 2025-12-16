import { io, Socket } from 'socket.io-client';
import { DetectionResult } from './types';

export class WebSocketClient {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  connect(
    url: string,
    onResult: (result: DetectionResult) => void,
    onError: (error: string) => void,
    onConnect: () => void,
    onDisconnect: () => void
  ): void {
    this.socket = io(url, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: 1000,
    });

    this.socket.on('connect', () => {
      console.log('Connected to server');
      this.reconnectAttempts = 0;
      onConnect();
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from server');
      onDisconnect();
    });

    this.socket.on('result', (data: DetectionResult) => {
      onResult(data);
    });

    this.socket.on('error', (data: { message: string }) => {
      console.error('Server error:', data.message);
      onError(data.message);
    });

    this.socket.on('connect_error', (error) => {
      console.error('Connection error:', error);
      this.reconnectAttempts++;
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        onError('Failed to connect to server after multiple attempts');
      }
    });
  }

  sendFrame(frameData: string): void {
    if (this.socket && this.socket.connected) {
      this.socket.emit('frame', { data: frameData });
    }
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  isConnected(): boolean {
    return this.socket?.connected ?? false;
  }
}


