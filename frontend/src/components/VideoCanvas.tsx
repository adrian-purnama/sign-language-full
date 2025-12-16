import React, { useEffect, useRef, useMemo } from 'react';
import { DetectionResult } from '../types';

interface VideoCanvasProps {
  result: DetectionResult | null;
  width?: number;
  height?: number;
}

export const VideoCanvas: React.FC<VideoCanvasProps> = React.memo(({ 
  result, 
  width = 640, 
  height = 480 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw bounding box
    if (result?.bounding_box) {
      const [x1, y1, x2, y2] = result.bounding_box;
      
      // Draw bounding box rectangle
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Draw label background
      if (result.detected_word !== '...') {
        const label = `${result.detected_word} (${Math.round(result.confidence * 100)}%)`;
        ctx.font = 'bold 18px Arial';
        const metrics = ctx.measureText(label);
        const labelWidth = metrics.width + 10;
        const labelHeight = 25;
        
        // Draw background rectangle for text
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x1, y1 - labelHeight - 5, labelWidth, labelHeight);
        
        // Draw text
        ctx.fillStyle = '#00FF00';
        ctx.fillText(label, x1 + 5, y1 - 10);
      }
    }
  }, [result, width, height]);

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

