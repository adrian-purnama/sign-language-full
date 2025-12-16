import React, { useEffect, useRef, useCallback } from 'react';

interface VideoCaptureProps {
  onFrame: (frameData: string) => void;
  onVideoReady?: (video: HTMLVideoElement) => void;
  fps?: number;
}

export const VideoCapture: React.FC<VideoCaptureProps> = React.memo(({ onFrame, onVideoReady, fps = 15 }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number>();

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || video.readyState !== video.HAVE_ENOUGH_DATA) {
      animationFrameRef.current = requestAnimationFrame(captureFrame);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Convert to JPEG with quality 0.8 to reduce payload size
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    onFrame(frameData);

    // Limit frame rate
    setTimeout(() => {
      animationFrameRef.current = requestAnimationFrame(captureFrame);
    }, 1000 / fps);
  }, [onFrame, fps]);

  useEffect(() => {
    let isMounted = true;

    const startCapture = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
        });

        if (!isMounted) {
          stream.getTracks().forEach(track => track.stop());
          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          
          // Wait for video to be ready before playing
          const playPromise = videoRef.current.play();
          
          if (playPromise !== undefined) {
            playPromise
              .then(() => {
                if (!isMounted) return;
                // Start capturing frames once video is playing
                if (onVideoReady && videoRef.current) {
                  onVideoReady(videoRef.current);
                }
                animationFrameRef.current = requestAnimationFrame(captureFrame);
              })
              .catch((error) => {
                // Ignore play errors if component unmounted
                if (!isMounted) return;
                console.error('Error playing video:', error);
              });
          }

          videoRef.current.onloadedmetadata = () => {
            if (!isMounted || !videoRef.current) return;
            // Start capturing once metadata is loaded
            if (!animationFrameRef.current) {
              animationFrameRef.current = requestAnimationFrame(captureFrame);
            }
          };
        }
      } catch (error) {
        if (!isMounted) return;
        console.error('Error accessing webcam:', error);
        alert('Failed to access webcam. Please allow camera permissions.');
      }
    };

    startCapture();

    return () => {
      isMounted = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = undefined;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [captureFrame, onVideoReady]);

  return (
    <div style={{ position: 'relative', display: 'block', width: '100%', margin: 0, padding: 0 }}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          display: 'block',
          width: '100%',
          height: 'auto',
          borderRadius: '8px',
          margin: 0,
          padding: 0,
        }}
      />
      <canvas
        ref={canvasRef}
        style={{ display: 'none' }}
      />
    </div>
  );
});

