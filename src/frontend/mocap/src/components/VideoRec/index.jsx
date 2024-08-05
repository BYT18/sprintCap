import { useState, useRef, useEffect } from "react";
import Button from 'react-bootstrap/Button';

const mimeType = 'video/webm; codecs="opus,vp8"';

const VideoRecorder = () => {
  const [permission, setPermission] = useState(false);
  const mediaRecorder = useRef(null);
  const liveVideoFeed = useRef(null);
  const [recordingStatus, setRecordingStatus] = useState("inactive");
  const [stream, setStream] = useState(null);
  const [recordedVideo, setRecordedVideo] = useState(null);
  const [videoChunks, setVideoChunks] = useState([]);

  const getCameraPermission = async () => {
    setRecordedVideo(null);
    try {
      const videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
      const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const combinedStream = new MediaStream([
        ...videoStream.getVideoTracks(),
        ...audioStream.getAudioTracks(),
      ]);

      setStream(combinedStream);
      setPermission(true);

      if (liveVideoFeed.current) {
        liveVideoFeed.current.srcObject = combinedStream;
      }
    } catch (err) {
      alert(err.message);
    }
  };

  const startRecording = () => {
    if (!stream) return;

    setRecordingStatus("recording");
    const media = new MediaRecorder(stream, { mimeType });
    mediaRecorder.current = media;

    let localVideoChunks = [];

    mediaRecorder.current.ondataavailable = (event) => {
      if (event.data.size > 0) {
        localVideoChunks.push(event.data);
      }
    };

    mediaRecorder.current.onstop = () => {
      const videoBlob = new Blob(localVideoChunks, { type: mimeType });
      const videoUrl = URL.createObjectURL(videoBlob);
      setRecordedVideo(videoUrl);
      setVideoChunks([]);
    };

    mediaRecorder.current.start();
  };

  const stopRecording = () => {
    setRecordingStatus("inactive");
    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
    }
  };

  useEffect(() => {
    if (liveVideoFeed.current && stream) {
      liveVideoFeed.current.srcObject = stream;
    }
  }, [stream]);

  return (
    <div>
      <main>
        <div className="video-controls">
          {!permission && (
            <Button variant="primary" className="mb-3" onClick={getCameraPermission}>
              Open Camera
            </Button>
          )}
          {permission && recordingStatus === "inactive" && (
            <Button variant="primary" className="mb-2" onClick={startRecording}>
              Start Recording
            </Button>
          )}
          {recordingStatus === "recording" && (
            <Button variant="primary" className="mb-2"  onClick={stopRecording} >
              Stop Recording
            </Button>
          )}
        </div>
      </main>

      <div className="video-player">
        {permission && !recordedVideo && (
          <video ref={liveVideoFeed} autoPlay muted className="live-player"></video>
        )}
        {recordedVideo && (
          <div className="recorded-player">
            <video className="recorded" src={recordedVideo} controls></video>
            <a download href={recordedVideo} download="recording.webm">
              Download Recording
            </a>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoRecorder;
