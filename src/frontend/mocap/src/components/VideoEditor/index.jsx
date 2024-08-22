
import React, { useEffect, useRef, useState } from 'react';
import Nouislider from 'nouislider-react';
import 'nouislider/distribute/nouislider.css';
import './style.css';

let ffmpeg;

function App({setParentVid}) {
  const [videoDuration, setVideoDuration] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [videoSrc, setVideoSrc] = useState('');
  const [videoFileValue, setVideoFileValue] = useState(null);
  const [isScriptLoaded, setIsScriptLoaded] = useState(false);
  const [videoTrimmedUrl, setVideoTrimmedUrl] = useState('');
  const videoRef = useRef();
  let initialSliderValue = 0;

  const loadScript = (src) => {
    return new Promise((onFulfilled) => {
      const script = document.createElement('script');
      script.async = 'async';
      script.defer = 'defer';
      script.setAttribute('src', src);
      script.onload = () => onFulfilled(script);
      script.onerror = () => console.log('Script failed to load');
      document.head.appendChild(script);
    });
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    const blobURL = URL.createObjectURL(file);
    setVideoFileValue(file);
    setVideoSrc(blobURL);
    setParentVid(file)
  };

  // Convert time in seconds to HH:MM:SS.MS format
  const convertToHHMMSSWithMilliseconds = (val) => {
  const hours = Math.floor(val / 3600);
  const minutes = Math.floor((val % 3600) / 60);
  const seconds = Math.floor(val % 60);
  const milliseconds = Math.floor((val * 1000) % 1000);

  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
};



  useEffect(() => {
    loadScript('https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.11.2/dist/ffmpeg.min.js').then(() => {
      if (typeof window !== 'undefined') {
        ffmpeg = window.FFmpeg.createFFmpeg({ log: true });
        ffmpeg.load().then(() => setIsScriptLoaded(true));
      }
    });
  }, []);

  useEffect(() => {
    if (videoRef.current) {
      const currentVideo = videoRef.current;
      currentVideo.onloadedmetadata = () => {
        setVideoDuration(currentVideo.duration);
        setEndTime(currentVideo.duration);
      };
    }
  }, [videoSrc]);

  const updateOnSliderChange = (values, handle) => {
    setVideoTrimmedUrl('');
    let readValue = values[handle] | 0;
    if (handle === 1) {
      if (endTime !== readValue) setEndTime(readValue);
    } else {
      if (initialSliderValue !== readValue) {
        initialSliderValue = readValue;
        if (videoRef.current) {
          videoRef.current.currentTime = readValue;
          setStartTime(readValue);
        }
      }
    }
  };

  const handlePlay = () => {
    if (videoRef.current) videoRef.current.play();
  };

  const handlePauseVideo = (e) => {
    const currentTime = Math.floor(e.currentTarget.currentTime);
    if (currentTime === endTime) e.currentTarget.pause();
  };

  const handleTrim = async () => {
    if (isScriptLoaded) {
      try {
        if (!videoFileValue) throw new Error("No video file selected");
        const { name, type } = videoFileValue;
        const fileData = await window.FFmpeg.fetchFile(videoFileValue);

        if (!fileData || fileData.length === 0) throw new Error("Failed to load video file");

        ffmpeg.FS('writeFile', name, fileData);
        console.log(startTime)
        //console.log( convertToHHMMSS(startTime))
        console.log(endTime)
        const outputFileName = 'out.mov';
        await ffmpeg.run(
          '-ss', `${convertToHHMMSSWithMilliseconds(startTime)}`,
          '-i', name,
          '-to', `${convertToHHMMSSWithMilliseconds(endTime)}`,
          '-c', 'copy',
           '-movflags', '+faststart', // Ensure the moov atom is at the beginning
          outputFileName
        );

        const data = ffmpeg.FS('readFile', outputFileName);
        if (!data || data.length === 0) throw new Error(`Output file ${outputFileName} not found`);

        const trimmedBlob = new Blob([data.buffer], { type: 'video/quicktime' });
        const url = URL.createObjectURL(trimmedBlob);
        setVideoTrimmedUrl(url);

        // Pass the trimmed video file back to the parent component
        setParentVid(trimmedBlob);

      } catch (error) {
        console.error("Error during video trimming:", error);
        alert("There was an error processing the video. Please try again.");
      }
    }
  };

  return (
    <div className="App" style={{fontFamily:"Quicksand, sans-serif"}}>
      <input class="form-control mb-2" type="file" id="formFileMultiple" accept="video/*" onChange={handleFileUpload}/>
      <br />
      {videoSrc && (
        <>
          <video src={videoSrc} ref={videoRef} onTimeUpdate={handlePauseVideo} controls>
            <source src={videoSrc} type={videoFileValue?.type || 'video/mp4'} />
          </video>
          <br />
          <Nouislider
            behaviour="tap-drag"
            step={1}
            margin={2}
            limit={30}
            range={{ min: 0, max: videoDuration || 2 }}
            start={[0, videoDuration || 2]}
            connect
            onUpdate={updateOnSliderChange}
          />
          <br />
          <p style = {{color:"black", fontFamily:"Quicksand, sans-serif"}}>
          Start duration: {convertToHHMMSSWithMilliseconds(startTime)} &nbsp; End duration:{' '}
          {convertToHHMMSSWithMilliseconds(endTime)}
          </p>
          <button  onClick={handlePlay}>Play</button> &nbsp;
          <button onClick={handleTrim}>Trim</button>
          <br />
          {videoTrimmedUrl && (
            <video controls>
              <source src={videoTrimmedUrl} type="video/mp4" />
            </video>
          )}
        </>
      )}
    </div>
  );
}

export default App;


{/*import { createFFmpeg } from "@ffmpeg/ffmpeg";
import { useEffect, useState, useCallback } from "react";
import { Slider, Spin } from "antd";
import { VideoPlayer } from "../VideoPlayer/index.jsx";
import VideoUpload from "../VideoUpload/index.jsx";
import VideoConversionButton from "../VideoConversion/index.jsx";
import './style.css';

const ffmpeg = createFFmpeg({ log: true });

function VideoEditor({ VideoUrl, onVideoEdit }) {
    const [ffmpegLoaded, setFFmpegLoaded] = useState(false);
    const [videoFile, setVideoFile] = useState(null);
    const [videoPlayerState, setVideoPlayerState] = useState(null);
    const [videoPlayer, setVideoPlayer] = useState(null);
    const [gifUrl, setGifUrl] = useState(null);
    const [sliderValues, setSliderValues] = useState([0, 1000]);
    const [processing, setProcessing] = useState(false);

    useEffect(() => {
        // loading ffmpeg on startup
        ffmpeg.load().then(() => {
            setFFmpegLoaded(true);
        });
    }, []);

    useEffect(() => {
        // Set videoFile from VideoUrl if provided
        if (VideoUrl) {
            fetch(VideoUrl)
                .then(response => response.blob())
                .then(blob => {
                    const file = new File([blob], 'video.mp4', { type: 'video/mp4' });
                    setVideoFile(file);
                })
                .catch(err => console.error('Error fetching video:', err));
        }
    }, [VideoUrl]);

    // Mapping slider values to video time
    const sliderValueToVideoTime = (duration, value) => {
        const time = (value / 1000) * duration;
        console.log(`Slider Value: ${value}, Mapped Time: ${time}`);
        return time;
    };

    const handleSliderChange = useCallback((values) => {
        setSliderValues(values);

        const minTime = sliderValueToVideoTime(videoPlayerState.duration, values[0]);
        videoPlayer.seek(minTime);
        //handleTrim();
    }, [videoPlayer, videoPlayerState]);

    // Function to format time as HH:MM:SS.mmm
const formatTime = (timeInSeconds) => {
    //const milliseconds = Math.floor((timeInSeconds % 1) * 1000);
    const milliseconds = 0

    const seconds = Math.floor(timeInSeconds);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    const formattedSeconds = seconds % 60;
    const formattedMinutes = minutes % 60;
    const formattedHours = hours;

    return `${formattedHours.toString().padStart(2, '0')}:${formattedMinutes.toString().padStart(2, '0')}:${formattedSeconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
};
    const handleTrim = async () => {
    setProcessing(true);

    const start0 = sliderValueToVideoTime(videoPlayerState.duration, sliderValues[0]);
    const end0 = sliderValueToVideoTime(videoPlayerState.duration, sliderValues[1]);

            // Implement logic to find nearest keyframe for more stable seeking
    const start1 = parseFloat(start0.toFixed(3));
    const end1 = parseFloat(end0.toFixed(3));

    const start = formatTime(start1); // Format start time
    const end= formatTime(end1); // Format end time
    console.log(start1)
      console.log(start)

    const inputFileName = "input.mp4";
    const outputFileName = "output.mp4";

    ffmpeg.FS("writeFile", inputFileName, await fetchFile(videoFile));

    await ffmpeg.run(
      "-i",
      inputFileName,
      "-ss",
      `${0}`,
      "-to",
      `${end}`,
      "-c",
      "copy",
      outputFileName
    );

    const data = ffmpeg.FS("readFile", outputFileName);
    const trimmedBlob = new Blob([data.buffer], { type: "video/mp4" });
    // Download the trimmed video
    const blobURL = URL.createObjectURL(trimmedBlob);
    const link = document.createElement('a');
    link.href = blobURL;
    link.download = 'trimmed_video.mp4'; // Specify the filename
    document.body.appendChild(link); // Append the link to the body
    link.click(); // Trigger the download
    document.body.removeChild(link); // Remove the link from the document
    URL.revokeObjectURL(blobURL); // Revoke the URL to free up memory


    setProcessing(false);
    onVideoEdit(trimmedBlob);
  };
  // Helper function to fetch the video file
async function fetchFile(file) {
  const response = await fetch(URL.createObjectURL(file));
  const buffer = await response.arrayBuffer();
  return new Uint8Array(buffer);
}


    useEffect(() => {
    if (videoPlayer && videoPlayerState) {
        const [min, max] = sliderValues;

        const minTime = sliderValueToVideoTime(videoPlayerState.duration, min);
        const maxTime = sliderValueToVideoTime(videoPlayerState.duration, max);

        // Implement logic to find nearest keyframe for more stable seeking
        const nearestMinTime = parseFloat(minTime.toFixed(3));
        const nearestMaxTime = parseFloat(maxTime.toFixed(3));


            console.log(`Seeking to nearest keyframes: minTime = ${nearestMinTime}, maxTime = ${nearestMaxTime}`);

            if (videoPlayerState.currentTime < nearestMinTime) {
                videoPlayer.seek(nearestMinTime);
            }
            if (videoPlayerState.currentTime > nearestMaxTime) {
                videoPlayer.seek(nearestMinTime);  // Loop back to start of range if overshooting
            }
        }
    }, [videoPlayerState, sliderValues, videoPlayer]);

    // Function to find nearest keyframe
    const findNearestKeyframe = (time) => {
        // This function should return the nearest keyframe time, you can use FFmpeg or any other method to determine keyframes
        // For now, assuming it's a dummy implementation
        return Math.floor(time);  // Just an example, should be based on actual keyframes
    };

    useEffect(() => {
        if (!videoFile) {
            setVideoPlayerState(null);
            setSliderValues([0, 1000]);
            setGifUrl(null);
        }
    }, [videoFile]);

    return (
        <div className="video-container">
            <Spin
                spinning={processing || !ffmpegLoaded}
                tip={!ffmpegLoaded ? "Waiting for FFmpeg to load..." : "Processing..."}
                style={{ minWidth: '100vw'}}
            >
                <div className="video-player">
                    {videoFile ? (
                        <VideoPlayer
                            src={URL.createObjectURL(videoFile)}
                            onPlayerChange={setVideoPlayer}
                            onChange={setVideoPlayerState}
                        />
                    ) : (
                        <h1>No video loaded</h1>
                    )}
                </div>
                <div className="upload-div">
                    <VideoUpload
                        disabled={!!videoFile}
                        onChange={setVideoFile}
                        onRemove={() => setVideoFile(null)}
                    />
                </div>
                <button type="button" class="btn btn-secondary" onClick={handleTrim}>
            Trim
          </button>
                <div className="slider-div">
                    <h3>Trim Video</h3>
                    <Slider
                        disabled={!videoPlayerState}
                        value={sliderValues}
                        range
                        max={1000} // Increase the max value for finer control
                        onChange={handleSliderChange}
                        tooltip={{
                            formatter: value => {
                                const time = sliderValueToVideoTime(videoPlayerState.duration, value);
                                return new Date(time * 1000).toISOString().substr(11, 8); // Display HH:MM:SS
                            }
                        }}
                    />
                </div>

            </Spin>
        </div>
    );
}

export default VideoEditor;*/}


{/*import React, { useEffect, useRef, useState } from 'react';
import Nouislider from 'nouislider-react';
import 'nouislider/distribute/nouislider.css';
import './style.css';

let ffmpeg; //Store the ffmpeg instance
function App(url) {
  const [videoDuration, setVideoDuration] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [videoSrc, setVideoSrc] = useState(url);
  const [videoFileValue, setVideoFileValue] = useState('');
  const [isScriptLoaded, setIsScriptLoaded] = useState(false);
  const [videoTrimmedUrl, setVideoTrimmedUrl] = useState('');
  const videoRef = useRef();
  let initialSliderValue = 0;

  //Created to load script by passing the required script and append in head tag
  const loadScript = (src) => {
    return new Promise((onFulfilled, _) => {
      const script = document.createElement('script');
      let loaded;
      script.async = 'async';
      script.defer = 'defer';
      script.setAttribute('src', src);
      script.onreadystatechange = script.onload = () => {
        if (!loaded) {
          onFulfilled(script);
        }
        loaded = true;
      };
      script.onerror = function () {
        console.log('Script failed to load');
      };
      document.getElementsByTagName('head')[0].appendChild(script);
    });
  };

  //Handle Upload of the video
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    const blobURL = URL.createObjectURL(file);
    setVideoFileValue(file);
    setVideoSrc(blobURL);
  };

  //Convert the time obtained from the video to HH:MM:SS format
  const convertToHHMMSS = (val) => {
    const secNum = parseInt(val, 10);
    let hours = Math.floor(secNum / 3600);
    let minutes = Math.floor((secNum - hours * 3600) / 60);
    let seconds = secNum - hours * 3600 - minutes * 60;

    if (hours < 10) {
      hours = '0' + hours;
    }
    if (minutes < 10) {
      minutes = '0' + minutes;
    }
    if (seconds < 10) {
      seconds = '0' + seconds;
    }
    let time;
    // only mm:ss
    if (hours === '00') {
      time = minutes + ':' + seconds;
    } else {
      time = hours + ':' + minutes + ':' + seconds;
    }
    return time;
  };

  useEffect(() => {
    //Load the ffmpeg script
    loadScript(
      'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.11.2/dist/ffmpeg.min.js',
    ).then(() => {
      if (typeof window !== 'undefined') {
        // creates a ffmpeg instance.
        ffmpeg = window.FFmpeg.createFFmpeg({ log: true });
        //Load ffmpeg.wasm-core script
        ffmpeg.load();
        //Set true that the script is loaded
        setIsScriptLoaded(true);
      }
    });
  }, []);

  //Get the duration of the video using videoRef
  useEffect(() => {
    if (videoRef && videoRef.current) {
      const currentVideo = videoRef.current;
      currentVideo.onloadedmetadata = () => {
        setVideoDuration(currentVideo.duration);
        setEndTime(currentVideo.duration);
      };
    }
  }, [videoSrc]);

  //Called when handle of the nouislider is being dragged
  const updateOnSliderChange = (values, handle) => {
    setVideoTrimmedUrl('');
    let readValue;
    if (handle) {
      readValue = values[handle] | 0;
      if (endTime !== readValue) {
        setEndTime(readValue);
      }
    } else {
      readValue = values[handle] | 0;
      if (initialSliderValue !== readValue) {
        initialSliderValue = readValue;
        if (videoRef && videoRef.current) {
          videoRef.current.currentTime = readValue;
          setStartTime(readValue);
        }
      }
    }
  };

  //Play the video when the button is clicked
  const handlePlay = () => {
    if (videoRef && videoRef.current) {
      videoRef.current.play();
    }
  };

  //Pause the video when then the endTime matches the currentTime of the playing video
  const handlePauseVideo = (e) => {
    const currentTime = Math.floor(e.currentTarget.currentTime);

    if (currentTime === endTime) {
      e.currentTarget.pause();
    }
  };

  //Trim functionality of the video
  const handleTrim = async () => {
    if (isScriptLoaded) {
      const { name, type } = videoFileValue;
      //Write video to memory
      ffmpeg.FS(
        'writeFile',
        name,
        await window.FFmpeg.fetchFile(videoFileValue),
      );
      const videoFileType = type.split('/')[1];
      //Run the ffmpeg command to trim video
      await ffmpeg.run(
        '-i',
        name,
        '-ss',
        `${convertToHHMMSS(startTime)}`,
        '-to',
        `${convertToHHMMSS(endTime)}`,
        '-acodec',
        'copy',
        '-vcodec',
        'copy',
        `out.${videoFileType}`,
      );
      //Convert data to url and store in videoTrimmedUrl state
      const data = ffmpeg.FS('readFile', `out.${videoFileType}`);
      const url = URL.createObjectURL(
        new Blob([data.buffer], { type: videoFileValue.type }),
      );
      setVideoTrimmedUrl(url);
    }
  };

  return (
    <div className="App">
      <input type="file" onChange={handleFileUpload} />
      <br />
      {videoSrc.length ? (
        <React.Fragment>
          <video src={videoSrc} ref={videoRef} onTimeUpdate={handlePauseVideo}>
            <source src={videoSrc} type={videoFileValue.type} />
          </video>
          <br />
          <Nouislider
            behaviour="tap-drag"
            step={0.5}
            margin={1}
            limit={2}
            range={{ min: 0, max: 2 }}
            start={[0,  2]}
            connect
            onUpdate={updateOnSliderChange}
          />
          <br />
          Start duration: {convertToHHMMSS(startTime)} &nbsp; End duration:{' '}
          {convertToHHMMSS(endTime)}
          <br />
          <button onClick={handlePlay}>Play</button> &nbsp;
          <button onClick={handleTrim}>Trim</button>
          <br />
          {videoTrimmedUrl && (
            <video controls>
              <source src={videoTrimmedUrl} type={videoFileValue.type} />
            </video>
          )}
        </React.Fragment>
      ) : (
        ''
      )}
    </div>
  );
}

export default App;*/}

