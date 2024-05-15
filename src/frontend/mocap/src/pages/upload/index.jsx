import React, { useState, useEffect, useRef } from 'react';

const VidComp = () => {
  const [video1, setVideo1] = useState(null);
  const [file1,setFile1] = useState(null);
  const [video2, setVideo2] = useState(null);
  const [gif1, setGif1] = useState("");
  const [key1, setKey1] = useState(1); // Add key state for video 1
  const [key2, setKey2] = useState(1); // Add key state for video 2

   const canvasRef = useRef(null);
    const contextRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);

  const handleVideo1Change = (event) => {
    setVideo1(URL.createObjectURL(event.target.files[0]));
    const v = event.target.files[0];
    setFile1(v)
    setKey1((prevKey) => prevKey + 1);
    analyze1(v);
  };

  const handleVideo2Change = (event) => {
    setVideo2(URL.createObjectURL(event.target.files[0]));
    setKey2((prevKey) => prevKey + 1);
  };

   const startDrawing = ({ nativeEvent }) => {
        const { offsetX, offsetY } = nativeEvent;
        contextRef.current.beginPath();
        contextRef.current.moveTo(offsetX, offsetY);
        setIsDrawing(true);
    };

    const finishDrawing = () => {
        contextRef.current.closePath();
        setIsDrawing(false);
    };

    const draw = ({ nativeEvent }) => {
        if (!isDrawing) {
            return;
        }
        const { offsetX, offsetY } = nativeEvent;
        contextRef.current.lineTo(offsetX, offsetY);
        contextRef.current.stroke();
    };

  const analyze1 = async (e) => {
        const formData = new FormData();
        formData.append("vid", e);
        console.log(e)
        try {
            // Create the POST request using the fetch API
            const response = await fetch('http://127.0.0.1:8000/test/', {
                method: 'POST',
                headers: {

                },
                body: formData,
            });
            // Check if the request was successful (status code in the range 200-299)
            if (response.ok) {
                const data = await response.json();
                console.log(data.pic);
                setGif1(data.pic)
                console.log(gif1)
            } else {
                // Handle error responses
                console.error('Error:', response.statusText);
            }
        } catch (error) {
            // Handle network errors
            console.error('Network error:', error.message);
        }
    };


  useEffect(() => {
    console.log('upload successful')

    const canvas = canvasRef.current;
    canvas.width = window.innerWidth * 2;
    canvas.height = window.innerHeight * 2;
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    const context = canvas.getContext('2d');
    context.scale(2, 2);
    context.lineCap = 'round';
    context.strokeStyle = 'red';
    context.lineWidth = 5;
    contextRef.current = context;
  }, [video1]);

  return (
    <div className="container" style={{ height: '80vh', marginTop:'25px', marginBottom:'60px',textAlign: "center"}}>
      {/*<input type="file" accept="video/*" onChange={handleVideo1Change} />
      <input type="file" accept="video/*" onChange={handleVideo2Change} />*/}
        <div class="mb-3 container">
            <div class="row">
                <div class="col-9">
                    <input class="form-control" type="file" id="formFileMultiple" accept="video/*" onChange={handleVideo1Change}/>
                </div>
                <div class="col">
                     <button type="button" class="btn btn-primary w-100">Overlay</button>
                </div>
            </div>
        </div>
        <div class="mb-3 container">
            <div class="row">
                <div class="col-9">
                    <input class="form-control" type="file" id="formFileMultiple" accept="video/*" onChange={handleVideo2Change}/>
                </div>
                <div class="col">
                    <button type="button" class="btn btn-primary w-100">Overlay</button>
                </div>
            </div>
        </div>
      {/*<div class="mb-3">
        <input class="form-control" type="file" id="formFileMultiple" multiple/>
      </div>*/}

      <div style={{ display: 'flex', maxHeight: '90%', overflow: 'hidden', marginTop:'25px'}}>
        {video1 && (
          <div style={{ flex: '1 1 50%', position: 'relative', maxHeight: '100%', overflow: 'hidden' }}>
            <video key={key1} controls width="100%" height="100%" style={{ objectFit: 'contain' }}>
              <source src={video1} type="video/mp4" />
              Your browser does not support HTML5 video.
            </video>
          </div>
        )}
<canvas onMouseDown={startDrawing} onMouseUp={finishDrawing} onMouseMove={draw} ref={canvasRef} />
        {video2 && (
          <div style={{ flex: '1 1 50%', position: 'relative', maxHeight: '90%', overflow: 'hidden' }}>
            <video key={key2} controls width="100%" height="100%" style={{ objectFit: 'contain' }}>
              <source src={video2} type="video/mp4" />
              Your browser does not support HTML5 video.
            </video>
          </div>
        )}

        {gif1 && (
          <div style={{ flex: '1 1 50%', position: 'relative', maxHeight: '90%', overflow: 'hidden' }}>
            <img src={gif1}/>
          </div>
        )}
      </div>
    </div>
  );
};

export default VidComp;
