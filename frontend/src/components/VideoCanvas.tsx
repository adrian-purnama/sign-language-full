import React, { useEffect, useRef, useMemo } from 'react';
import { DetectionResult } from '../types';

interface VideoCanvasProps {
  result: DetectionResult | null;
  width?: number;
  height?: number;
  videoElement?: HTMLVideoElement | null;
}

export const VideoCanvas: React.FC<VideoCanvasProps> = React.memo(({ 
  result, 
  width = 640, 
  height = 480,
  videoElement
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    if (result?.bounding_box && result.bounding_box[2] > 0 && result.bounding_box[3] > 0) {
      const [x1, y1, x2, y2] = result.bounding_box;
      
      let scaleX = 1;
      let scaleY = 1;
      
      if (result.frame_width && result.frame_height && videoElement) {
        const displayWidth = videoElement.clientWidth || width;
        const displayHeight = videoElement.clientHeight || height;
        
        scaleX = displayWidth / result.frame_width;
        scaleY = displayHeight / result.frame_height;
      }
      
      const scaledX1 = x1 * scaleX;
      const scaledY1 = y1 * scaleY;
      const scaledX2 = x2 * scaleX;
      const scaledY2 = y2 * scaleY;
      
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 3;
      ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);

      const label = result.detected_word !== '...'
        ? `${result.detected_word} (${Math.round(result.confidence * 100)}%)`
        : `... (${Math.round(result.confidence * 100)}%)`;
      
      ctx.font = 'bold 18px Arial';
      const metrics = ctx.measureText(label);
      const labelWidth = metrics.width + 10;
      const labelHeight = 25;
      
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(scaledX1, scaledY1 - labelHeight - 5, labelWidth, labelHeight);
      
      ctx.fillStyle = '#00FF00';
      ctx.fillText(label, scaledX1 + 5, scaledY1 - 10);
    }
  }, [result, width, height, videoElement]);

  const canvasStyle = useMemo(() => ({
    position: 'absolute' as const,
    top: 0,
    left: 0,
    pointerEvents: 'none' as const,
    borderRadius: '8px',
  }), []);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={canvasStyle}
    />
  );
});

