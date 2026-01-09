import React, { useEffect, useRef, useMemo } from 'react';
import { DetectionResult, Keypoint } from '../types';

interface KeypointsOverlayProps {
  result: DetectionResult | null;
  width: number;
  height: number;
  boundingBox: [number, number, number, number] | null;
}

const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10],
    [11, 12],
    [11, 13], [13, 15],
    [15, 17], [15, 19], [15, 21], [17, 19],
    [12, 14], [14, 16],
    [16, 18], [16, 20], [16, 22], [18, 20],
    [11, 23], [12, 24],
    [23, 24],
    [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
    [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
];

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
];

const drawKeypoints = (
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[] | null,
  connections: number[][],
  color: string,
  scaleX: number,
  scaleY: number,
  offsetX: number,
  offsetY: number
) => {
  if (!keypoints || keypoints.length === 0) return;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.fillStyle = color;


  connections.forEach(([start, end]) => {
    if (start < keypoints.length && end < keypoints.length) {
      const startPoint = keypoints[start];
      const endPoint = keypoints[end];
      
      const startX = startPoint.x * scaleX + offsetX;
      const startY = startPoint.y * scaleY + offsetY;
      const endX = endPoint.x * scaleX + offsetX;
      const endY = endPoint.y * scaleY + offsetY;

      if (startPoint.visibility === undefined || startPoint.visibility > 0.5) {
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      }
    }
  });


  keypoints.forEach((kp, idx) => {
    if (kp.visibility === undefined || kp.visibility > 0.5) {
      const x = kp.x * scaleX + offsetX;
      const y = kp.y * scaleY + offsetY;
      
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
};

export const KeypointsOverlay: React.FC<KeypointsOverlayProps> = React.memo(({
  result,
  width,
  height,
  boundingBox,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !result) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    let scaleX = width;
    let scaleY = height;
    let offsetX = 0;
    let offsetY = 0;

    if (boundingBox && boundingBox[2] > 0 && boundingBox[3] > 0) {
      const [x1, y1, x2, y2] = boundingBox;
      const boxWidth = x2 - x1;
      const boxHeight = y2 - y1;
      
      scaleX = boxWidth;
      scaleY = boxHeight;
      offsetX = x1;
      offsetY = y1;
    }

    if (result.keypoints.pose) {
      drawKeypoints(
        ctx,
        result.keypoints.pose,
        POSE_CONNECTIONS,
        '#00FF00',
        scaleX,
        scaleY,
        offsetX,
        offsetY
      );
    }

    if (result.keypoints.left_hand) {
      drawKeypoints(
        ctx,
        result.keypoints.left_hand,
        HAND_CONNECTIONS,
        '#FF0000',
        scaleX,
        scaleY,
        offsetX,
        offsetY
      );
    }

    if (result.keypoints.right_hand) {
      drawKeypoints(
        ctx,
        result.keypoints.right_hand,
        HAND_CONNECTIONS,
        '#0000FF',
        scaleX,
        scaleY,
        offsetX,
        offsetY
      );
    }
  }, [result, width, height, boundingBox]);

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

