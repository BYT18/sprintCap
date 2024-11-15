// VideoUploader.js
import React, { useState, useRef, useEffect} from 'react';
import { Tooltip } from 'react-tooltip'
import './VideoUploader.css';
import { BarChart } from '@mui/x-charts/BarChart';
import { LineChart, lineElementClasses } from '@mui/x-charts/LineChart';
import { ScatterChart } from '@mui/x-charts/ScatterChart';
import {
  motion,
  useScroll,
  useSpring,
  useTransform,
  MotionValue
} from "framer-motion";
import { saveAs } from 'file-saver'
import myImage1 from "../../assets/touchdown_R2.png";
import myImage2 from "../../assets/max_ver3.png";
import myImage3 from "../../assets/strike_R2.png";
import myImage4 from "../../assets/toe_off_L2.png";
import myImage5 from "../../assets/full_sup_L2.png";

import Loader from '../../components/Loader/index.jsx';
import PasteImage from '../../components/Dropdown/index.jsx';
import Slider from '../../components/Slider/index.jsx';
import VideoRec from '../../components/VideoRec/index.jsx';
import VideoCrop from '../../components/VideoEditor/index.jsx';

import 'bootstrap/dist/css/bootstrap.min.css';
import "bootstrap-icons/font/bootstrap-icons.css";
import 'bootstrap/dist/js/bootstrap.bundle.min';
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap/dist/js/bootstrap.bundle.min.js";
import * as Icon from 'react-bootstrap-icons';

import Modal from 'react-bootstrap/Modal';
import Button from 'react-bootstrap/Button';
import Carousel from 'react-bootstrap/Carousel';
import Form from 'react-bootstrap/Form';
import Offcanvas from 'react-bootstrap/Offcanvas';

const data = []

const uData = [40, 30, 20, 27, 18, 23, 34];
const xLabels = [
  'Page A',
  'Page B',
  'Page C',
  'Page D',
  'Page E',
  'Page F',
  'Page G',
];


const VideoUploader = () => {
  const [show, setShow] = useState(false);
  const handleClose = () => setShow(false);

  const ref = useRef(null);
  const { scrollXProgress } = useScroll({ container: ref });
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState('https://mocapltd.xyz/api/media/pics/adam.mov');
  const [images, setImages] = useState([]);
  const [displayedImages, setDisplayedImages] = useState([]);
  const [allImages, setAllImages] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [selectedSaveIndex, setSelectedSaveIndex] = useState(null);
  const [testImg, setTestImg] = useState(null);
  const [resultVid, setResultVid] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pasteImage, setPasteImage] = useState(null);
  const [isToggled, setIsToggled] = useState(false);
  const [isSlowToggled, setIsSlowToggled] = useState(false);
  const [analType, setAnalType] = useState(0);

  const [datas, setDatas] = useState([]);
  const [chartData, setChartData] = useState({});

  const [xLineVals, setXLineVals] = useState([]);
  const [yLineVals, setYLineVals] = useState([]);
  const [velDataL, setVelDataL] = useState([]);
  const [velDataR, setVelDataR] = useState([]);

  const [itemsToe, setItemsToe] = useState([]);
  const [textAreaToe, setTextAreaToe] = useState("");
  const [itemsV, setItemsV] = useState([]);
  const [textAreaV, setTextAreaV] = useState("");
  const [itemsS, setItemsS] = useState([]);
  const [textAreaS, setTextAreaS] = useState("");
  const [itemsT, setItemsT] = useState([]);
  const [textAreaT, setTextAreaT] = useState("");
  const [itemsFS, setItemsFS] = useState([]);
  const [textAreaFS, setTextAreaFS] = useState("");

  const [analData, setAnalData] = useState([]);
  const [analData2, setAnalData2] = useState([]);
  const [avgG, setAvgG] = useState(null);
  const [avgF, setAvgF] = useState(null);
  const [contactLabels, setContactLabels] = useState(null);
  const [angLabels, setAngLabels] = useState(null);

  const [showG1, setShowG1] = useState(false);
  const handleCloseG1 = () => setShowG1(false);
  const handleShowG1 = () => setShowG1(true);
  const [showFeed, setShowFeed] = useState(false);
  const [feedInd, setFeedInd] = useState(null);
  const handleCloseFeed = () => setShowFeed(false);
  const handleShowFeed = (i) => {
    setFeedInd(i);
    setShowFeed(true);
  }

  const getFeedbackMessage = (index) => {
        switch (index) {
            case 0:
                return 'Thigh ROM is crucial to generate power. Lack of ROM could be due weakness in glutes and quad.';
            case 1:
                return 'Knee SPARC measures the smoothness of movment in knees. It is highly dependent on quad to hamstring strength ratio.';
            case 2:
                return 'Nice';
            case 3:
                return 'Bad';
            default:
                return 'No feedback';
        }
    };

   // Handler function for carousel image click
  const handleCarouselClick = (index) => {
    setSelectedSaveIndex(index)
    const start = index * 5;
    const end = start + 5;
    setDisplayedImages(allImages.slice(start, end));
    console.log(displayedImages)
    setShow(true);
  };

  const handleSaveImage = () => {
    console.log(images[0])
    console.log(selectedIndex)
    images[selectedSaveIndex] = displayedImages[selectedIndex]
    console.log(images[0])
    setShow(false);
  };

  const handleToggleChange = (e) => {
        setIsToggled(e.target.checked);
  };

  const handleSlowToggleChange = (e) => {
        setIsSlowToggled(e.target.checked);
  };

  const handleAnalType = (event) => {
    setAnalType(event.target.value);
    console.log(analType)
  };

  const addImage = (newImage) => {
      setImages(prevImages => [...prevImages, newImage]);
  };


  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoFile(file);
      setVideoURL(url);
      //analyze(file);
    }
  };

  const handleAnalyze = () => {
    analyze(videoFile,pasteImage)
  };

  const handleImageUpload = (event) => {
    const files = Array.from(event.target.files);
    const newImages = files.map((file) => URL.createObjectURL(file));
    setImages([...images, ...newImages]);
  };

 const createCombinedImage = async (imageUrls, texts) => {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
 // Function to load images and calculate dimensions
  const loadImage = async (url) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'Anonymous'; // Set crossOrigin attribute
      img.onload = () => {
        resolve(img);
      };
      img.onerror = reject;
      img.src = url;
    });
  };

  // Load all images
  const images = await Promise.all(imageUrls.map(url => loadImage(url)));

  // Ensure all images are loaded before continuing
  await Promise.all(images.map(img => {
    return new Promise((resolve, reject) => {
      if (img.complete) {
        resolve(img);
      } else {
        img.onload = resolve;
        img.onerror = reject;
      }
    });
  }));

  // Calculate canvas dimensions based on images and text
  const imageWidth = images[0].width*1.5; // Assuming all images have the same width
  const imageHeight = images[0].height*1.5; // Assuming all images have the same height
  const canvasWidth = imageUrls.length * imageWidth + 150; // Add space between images
  const canvasHeight = imageHeight + 400; // Adjust height as needed

  canvas.width = canvasWidth;
  canvas.height = canvasHeight;

  let xOffset = 10;

  for (let i = 0; i < imageUrls.length; i++) {
    const imageUrl = imageUrls[i];
    const img = await loadImage(imageUrl);

    // Draw image
    ctx.drawImage(img, xOffset, 50, imageWidth, imageHeight);

     // Format and wrap text as bullet points below the image
    const lines = wrapText(ctx, texts[i], imageWidth - 10); // Example max width for wrapped text

    const lineHeight = 25; // Example line height
    ctx.font = '20px Arial';
    ctx.fillStyle = 'white'; // Set text color
    ctx.textAlign = 'left';

    let yOffset = 50 + imageHeight + 50; // Initial y-offset for text below image

    for (const line of lines) {
      // Draw bullet point (circle)
      //ctx.beginPath();
      //ctx.arc(xOffset + 10, yOffset - 10, 5, 0, Math.PI * 2);
      //ctx.fill();

      // Draw wrapped text line with indentation
      ctx.fillText(line.trim(), xOffset + 5, yOffset);
      yOffset += lineHeight; // Increment y-offset for next line
    }

    xOffset += imageWidth + 20; // Adjust spacing between images
  }

  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error('Canvas toBlob failed'));
      }
    }, 'image/jpeg', 0.9); // Adjust quality if needed
  });
};

const wrapText = (ctx, text, maxWidth) => {
  if (typeof text !== 'string') {
    console.error('Expected text to be a string');
    return [];
  }

  // Split the text by newline characters
  const paragraphs = text.split('\n');
  let lines = [];

  paragraphs.forEach(paragraph => {
    const words = paragraph.split(' ');
    let currentLine = words[0];

    for (let i = 1; i < words.length; i++) {
      const word = words[i];
      const width = ctx.measureText(currentLine + ' ' + word).width;
      if (width < maxWidth) {
        currentLine += ' ' + word;
      } else {
        lines.push(currentLine);
        currentLine = word;
      }
    }
    lines.push(currentLine); // Add the last line of the paragraph
  });

  return lines;
};

const fetchVideoBlob = async (videoURL) => {
    const response = await fetch(videoURL);
    const blob = await response.blob();
    return blob;
};

const loadImage = (src) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'Anonymous'; // Handle cross-origin issues if necessary
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
};

   const analyze = async (e,p) => {


        //const video = new File([e], "blob.mov", { type: e.type });


        const videoBlob = await fetchVideoBlob(videoURL);

        // Create a File object from the blob
        const video = new File([videoBlob], "blob.mov", { type: videoBlob.type });

        console.log(video)


        setLoading(true);
        const formData = new FormData();
        formData.append("vid", video);
        //formData.append("vid", e);
        const blob = await fetch(p).then(res => res.blob()); // Convert base64 to Blob
        formData.append('pic', blob, 'image.png'); // Add blob to formData with filename
        formData.append('height', height);
        formData.append('image_urls',[])
        console.log(analType)
        formData.append('analysis_type', analType);

        // Check the value of slwomo and append the appropriate value to formData
        if (isSlowToggled) {
            formData.append('slowmo', 1);
        } else {
            formData.append('slowmo', 0);
        }

        if (isToggled) {
            formData.append('step', 1);
        } else {
            formData.append('step', 0);
        }
        try {
            // Create the POST request using the fetch API
            const token = localStorage.getItem('access_token');
            console.log(token)
            //const response = await fetch('http://127.0.0.1:8000/api/demo/', {
            //const response = await fetch('http://3.131.119.69:8000/test/', {
            const response = await fetch('/api/demo/', {
                method: 'POST',
                headers: {

                },
                body: formData,
            });
            // Check if the request was successful (status code in the range 200-299)
            if (response.ok) {
                const data = await response.json();
                console.log(data)
                console.log(data.pic);

                setDatas(data.x_vals);

                //if (isToggled) {
                // Generate xAxis labels
                const xAxisLabels = data.x_vals["ground"].map((_, index) => `C${index + 1}`);
                setContactLabels(xAxisLabels)
                //}


                const xAxisData = Array.from({ length: data.x_vals["ang"].length }, (_, i) => i + 1);
                setAngLabels(xAxisData)


                /*const xValues = data.map(point => point.x_value);
                const yValues = data.map(point => point.y_value);*/
                /*const xValues = data.x_vals;
                const yValues = data.x_vals;
                setDatas(data.x_vals)

                setChartData({
                  labels: xValues,
                  datasets: [
                    {
                      label: 'Data Points',
                      data: yValues,
                      borderColor: 'rgba(75,192,192,1)',
                      backgroundColor: 'rgba(75,192,192,0.2)',
                    }
                  ]
                });*/

                setResultVid(data.pic)
                //setImages(data.pic)
                setTestImg(data.pic)
                setImages([])
                //console.log("After clearing:", images);
                setDisplayedImages([]);
                setAllImages([]);
                setSelectedIndex(null);
                setSelectedSaveIndex(null);

                addImage(data.kin1);
                addImage(data.kin2);
                addImage(data.kin3);
                addImage(data.kin4);
                addImage(data.kin5);
                const imageUrls = Array.from({ length: 25 }, (_, index) =>
                    `https://mocapltd.xyz/api/media/pics/out_${index + 1}.png`
                  );
                setAllImages(imageUrls)
                                    //`http://127.0.0.1:8000/media/pics/out_${index + 1}.png`

                const augmentedData = [
                  { title: 'Time between steps', value: '0.3', unit: 'SECONDS', color:data.x_vals["colors"][0] },
                  { title: 'Mean step length', value: Number(data.x_vals["maxSL"]).toFixed(3), unit: 'METERS',color:data.x_vals["colors"][1]},
                  { title: 'Mean Ground Time', value: Number(data.x_vals["avg_g"]).toFixed(3), unit: 'SECONDS',color:data.x_vals["colors"][2]},
                  { title: 'Mean Flight Time', value: Number(data.x_vals["avg_f"]).toFixed(3), unit: 'SECONDS',color:data.x_vals["colors"][3]}
                ];
                setAnalData(augmentedData);

                const augmentedData2 = [
                  { title: 'Thigh ROM', value: '2', unit: 'SECONDS', color:data.x_vals["colors"][4]},
                  { title: 'Knee SPARC', value: '0.5', unit: 'METERS',color:data.x_vals["colors"][5]},
                  { title: 'Shoulder AV', value: '0.8', unit: 'SECONDS',color:data.x_vals["colors"][6]},
                  { title: 'Hip SE', value: '1.2', unit: 'SECONDS',color:data.x_vals["colors"][7]}
                ];
                setAnalData2(augmentedData2);


                // Convert kneePos array to an array of objects with x and y properties
                const xValues = data.x_vals["kneePos"].map(coords => coords[0]);
                const yValues = data.x_vals["kneePos"].map(coords => coords[1]);

                setXLineVals(xValues);
                setYLineVals(yValues);

                setItemsToe(data.x_vals["feedback"]["TO"])
                formatItemsForTextarea(data.x_vals["feedback"]["TO"],setTextAreaToe);
                setItemsV(data.x_vals["feedback"]["MV"])
                formatItemsForTextarea(data.x_vals["feedback"]["MV"],setTextAreaV);
                setItemsS(data.x_vals["feedback"]["S"])
                formatItemsForTextarea(data.x_vals["feedback"]["S"],setTextAreaS);
                setItemsT(data.x_vals["feedback"]["TD"])
                formatItemsForTextarea(data.x_vals["feedback"]["TD"],setTextAreaT);
                setItemsFS(data.x_vals["feedback"]["FS"])
                formatItemsForTextarea(data.x_vals["feedback"]["FS"],setTextAreaFS);

                //setToeFeed(data.x_vals["feedback"]["TO"])


                //setAvgG(data.x_vals["feedback"]["avg_g"])
                //setAvgF(data.x_vals["feedback"]["avg_F"])
                //setVelDataL([data.x_vals["vR"],data.x_vals["vR"]])


                setVelDataL([data.x_vals["ang"],data.x_vals["ang"]])

                setLoading(false);

                //setGif1(data.pic)
                //console.log(gif1)
            } else {
                // Handle error responses
                console.error('Error:', response.statusText);
                alert(`Error: ${response.message} or a network error has occured`);
                setLoading(false);
            }
        } catch (error) {
            // Handle network errors
            console.error('Network error:', error.message);
            alert(`Error: ${error.message} or a network error has occured`);
            setLoading(false);
        }
    };

      // Format the items for display in the textarea
  const formatItemsForTextarea = (items,func) => {
    const formattedItems = items.map(item => `- ${item}`).join('\n');
    func(formattedItems);
  };

  // Handler to update state based on textarea changes
  const handleTextChangeT = (event) => {
    const updatedContent = event.target.value;
    setTextAreaToe(updatedContent);
    const updatedItems = updatedContent.split('\n').map(line => line.replace(/^•\s*/, ''));
    setItemsToe(updatedItems);
  };
  const handleTextChangeV = (event) => {
    const updatedContent = event.target.value;
    setTextAreaV(updatedContent);
    const updatedItems = updatedContent.split('\n').map(line => line.replace(/^•\s*/, ''));
    setItemsV(updatedItems);
  };
  const handleTextChangeS = (event) => {
    const updatedContent = event.target.value;
    setTextAreaS(updatedContent);
    const updatedItems = updatedContent.split('\n').map(line => line.replace(/^•\s*/, ''));
    setItemsS(updatedItems);
  };
  const handleTextChangeTD = (event) => {
    const updatedContent = event.target.value;
    setTextAreaT(updatedContent);
    const updatedItems = updatedContent.split('\n').map(line => line.replace(/^•\s*/, ''));
    setItemsT(updatedItems);
  };
  const handleTextChangeFS = (event) => {
    const updatedContent = event.target.value;
    setTextAreaFS(updatedContent);
    const updatedItems = updatedContent.split('\n').map(line => line.replace(/^•\s*/, ''));
    setItemsFS(updatedItems);
  };


      const fileInputRef = useRef(null);

  const handleFileChangePaste = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setPasteImage(event.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

   const handleDoubleClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    } else {
      console.error("File input ref is not set correctly.");
    }
  }

  const cardVariants: Variants = {
  offscreen: {
    opacity: 0,
    scale: 0.5
  },
  onscreen: {
    opacity: 1,
    scale: 1,
    //rotate: -10,
    transition: {
      duration: 0.8,
      delay: 0.5,
      ease: [0, 0.71, 0.2, 1.01]
      }
    }
  };

  useEffect(() => {
    console.log('upload successful')
    setImages([])
    //console.log('Images reset:', images);
    setResultVid(null)
  }, [videoFile]);

  const downloadImage = async () => {
        const feedbacks = [textAreaToe,textAreaV,textAreaS,textAreaT,textAreaFS]
        console.log(feedbacks)
        const combinedImage = await createCombinedImage(images, feedbacks);
        saveAs(combinedImage, 'kino.jpg')
       // saveAs(images[0], 'womp.jpg') // Put your image URL here.
  }

  const handlePaste = (event) => {
    const items = event.clipboardData.items;
    for (let item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        const reader = new FileReader();
        reader.onload = (e) => {
          setPasteImage(e.target.result);
        };
        reader.readAsDataURL(file);
      }
    }
  };

  const [height, setHeight] = useState(170); // Default height value (in cm)

  const handleSliderChange = (event) => {
    setHeight(event.target.value);
  };

  const steps = [
    {
      image: 'path/to/image1.jpg', // Replace with the actual image path
      description: 'Step 1: Open the camera or upload a video.',
    },
    {
      image: 'path/to/image2.jpg',
      description: 'Step 2: Crop it to desired length. Try to eliminate as much camera panning or distortion as possible.',
    },
    {
      image: 'path/to/image3.jpg',
      description: 'Step 3: Select the type of analysis to do.',
    },
    {
      image: 'path/to/image4.jpg',
      description: 'Step 4: Toggle slowmo if the video is in slowmo.',
    },
    {
      image: 'path/to/image5.jpg',
      description: 'Step 5: Toggle step analysis if you want to analyze step length and if your video includes a still frame of reference. You can then copy and paste this reference into the box.',
    },
     {
      image: 'path/to/image5.jpg',
      description: 'Step 6: Click analyze to get your results.',
    }
  ];


  return (
  <body >
   <section class="bannerCards text-sm-start text-center p-4">
    <motion.div class="container" initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{
        duration: 0.8,
        delay: 0.5,
        ease: [0, 0.71, 0.2, 1.01]
      }}>
      <div className="video-uploader">
          <h2>Upload and Display Video</h2>
          {videoURL && (
            <div className="video-container">
            <video controls src={videoURL}>
                Your browser does not support the video tag.
            </video>
              {/*<video controls src={videoURL}>
                Your browser does not support the video tag.
              </video>
              <VideoCrop VideoUrl={videoURL} onVideoEdit={handleEditedVideo}  />
              */}
            </div>
          )}
        <div className="container mt-2" data-tooltip-id="my-tooltip" data-tooltip-content="Change analysis depending on sprinting phase being recorded">
          <select class="form-select" disabled aria-label="Default select example" onChange={handleAnalType} value={analType} style={{fontFamily:"Quicksand, sans-serif"}}>
              <option  value="0">Top End Analysis</option>
              <option value="1">Acceleration Analysis</option>
              <option value="2">Tempo Stride Analysis</option>
          </select>
          <Tooltip id="my-tooltip" />
        </div>
        <div className="container mt-4" >
            <Form>
            <Form.Check
                type="switch"
                id="custom-switch"
                label="Slowmo"
                style={{color:"black", fontFamily:"Quicksand, sans-serif"}}
                checked={true}
                disabled
                onChange={handleSlowToggleChange}
                onChange={handleSlowToggleChange}
              />
             `<Form.Check
                type="switch"
                id="custom-switch"
                label="Toggle Step Length Analysis"
                style={{color:"black",fontFamily:"Quicksand, sans-serif"}}
                checked={isToggled}
                disabled
                onChange={handleToggleChange}
              />
            </Form>
          {isToggled && (<div
              className="paste-area"
              onPaste={handlePaste}
              onDoubleClick={handleDoubleClick}
              style={{ border: '2px dashed #ccc', padding: '20px', textAlign: 'center', cursor: 'pointer' }}
              data-tooltip-id="my-tooltip" data-tooltip-content="This is used to compute step length and velocity"
            >
              <p style={{color:"black"}}>Click here and press Ctrl+V to paste image of measuring device or double click to upload image</p>
                         <Tooltip id="my-tooltip" />
              {pasteImage && <img src={pasteImage} alt="Pasted" style={{ maxWidth: '100%', maxHeight: '400px', marginTop: '20px' }} />}
              <input
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                ref={fileInputRef}
                onChange={handleFileChangePaste}
             />
            </div>)}
        </div>
        {/*<div className="container mt-2">
              <div className="slider-container">
              <label style={{color:"black"}} htmlFor="height-slider">Select Height: {height} cm</label>
              <input
                type="range"
                id="height-slider"
                min="50"
                max="250"
                value={height}
                onChange={handleSliderChange}
                className="slider"
              />
            </div>
        </div>*/}
        <div className="container mt-2">
          <button type="button" class="btn btn-secondary" onClick={handleAnalyze} style={{fontFamily:"Quicksand, sans-serif"}}>
            Analyze
          </button>
        </div>
      </div>
    </motion.div>
     <button onClick={handleShowG1} className="help-button"><i className="bi bi-question-circle"></i></button>
     <Offcanvas show={showG1} onHide={handleCloseG1}>
        <Offcanvas.Header closeButton>
          <Offcanvas.Title>How to analyze your run</Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body>
          <div className="steps-container">
          {steps.map((step, index) => (
            <div key={index} className="step">
              <img src={step.image} alt={`Step ${index + 1}`} className="step-image" />
              <p className="step-description">{step.description}</p>
            </div>
          ))}
        </div>
        </Offcanvas.Body>
  </Offcanvas>
  </section>
   {loading && <Loader />}
   {/*resultVid && */}
  {resultVid &&(

  <div>
  <section class="bannerCards text-sm-start text-center p-4">
      <div class="container">
      <div className="video-uploader">
      <h2>Fitted</h2>
         <div className="video-container">
            <video controls src={resultVid}>
              Your browser does not support HTML5 video.
            </video>
            </div>
      </div>
      </div>
  </section>

  {/*<section class="bannerCards text-sm-start text-center p-4">
      <div class="container">
        <h2>Upload and Display Pictures</h2>
          <input type="file" accept="image/*" multiple onChange={handleImageUpload} />
          <div className="carder image-container">
            {images.map((url, index) => (
              <img key={index} src={url} alt={`Uploaded ${index}`} className="uploaded-image" />
            ))}
          </div>
      </div>
  </section>*/}

  <section class="bannerCards p-4">
  <motion.div className="container"
  variants={cardVariants}
   initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
  >
  <svg id="progress" width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="30" pathLength="1" className="bg" />
        <motion.circle
          cx="50"
          cy="50"
          r="30"
          pathLength="1"
          className="indicator"
          style={{ pathLength: scrollXProgress }}
        />
      </svg>
      <ul ref={ref}  class="anul">
        <motion.li class="anli"
            whileHover={{scale:1.02}}
        >
          <img src={images[0]} className="img-fluid" alt="Image" onClick={() => handleCarouselClick(0)} style={{ cursor: 'pointer' }}/>
          <div class="text-container">
              <h3 >Toe Off</h3>
              <textarea
                value={textAreaToe}
                onChange={handleTextChangeT}
                rows="5"
                cols="50"
                className="textarea-styled"
              />
          </div>
        </motion.li>
         <motion.li class="anli"
            whileHover={{scale:1.02}}
        >
            <img src={images[1]} onClick={() => handleCarouselClick(1)} style={{ cursor: 'pointer' }}/>
            <div class="text-container">
              <h3 >Max Vertical Projection</h3>
              {/*<div class="scrollable-list" contenteditable="true">
              <ul>
                <li>Neutral head</li>
                <li>>110 degrees knee angle in front leg</li>
                {vertFeed.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
              </ul>
            </div>*/}
            <textarea
                value={textAreaV}
                onChange={handleTextChangeV}
                rows="5"
                cols="50"
                 className="textarea-styled"
              />
              </div>
        </motion.li>
         <motion.li class="anli"
            whileHover={{scale:1.02}}
        >
            <img src={images[2]} onClick={() => handleCarouselClick(2)} style={{ cursor: 'pointer' }}/>
            <div class="text-container">
              <h3 >Strike</h3>
              <textarea
                value={textAreaS}
                onChange={handleTextChangeS}
                rows="5"
                cols="50"
                 className="textarea-styled"
              />
          </div>
        </motion.li>
         <motion.li class="anli"
            whileHover={{scale:1.02}}
        >
            <img src={images[3]} onClick={() => handleCarouselClick(3)} style={{ cursor: 'pointer' }}/>
            <div class="text-container">
              <h3 >Touch Down</h3>
              <textarea
                value={textAreaT}
                onChange={handleTextChangeTD}
                rows="5"
                cols="50"
                 className="textarea-styled"
              />
            </div>
        </motion.li>
         <motion.li class="anli"
            whileHover={{scale:1.02}}
        >
            <img src={images[4]} onClick={() => handleCarouselClick(4)} style={{ cursor: 'pointer' }}/>
            <div class="text-container">
              <h3 >Full Support</h3>
              <textarea
                value={textAreaFS}
                onChange={handleTextChangeFS}
                rows="5"
                cols="50"
                 className="textarea-styled"
              />
            </div>
        </motion.li>
      </ul>
       <button onClick={downloadImage} className="download-button"><i className="bi bi-download"></i></button>
  </motion.div>
  </section>

   <Modal show={show} onHide={handleClose} centered>
        <Modal.Header closeButton>
          <Modal.Title>Select Image</Modal.Title>
        </Modal.Header>
        <Modal.Body>
            <Carousel data-bs-theme="dark" interval={null} onSelect={(index) => setSelectedIndex(index)}>
              {displayedImages.map((image, index) => (
                <Carousel.Item key={index}>
                    <img
                        className="d-block w-100 carousel-image" // Apply custom CSS class
                        src={image}
                        alt={`Slide ${index}`}
                    />
                </Carousel.Item>
            ))}
            </Carousel>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={handleClose}>
            Close
          </Button>
          <Button variant="primary" onClick={handleSaveImage}>
            Save Changes
          </Button>
        </Modal.Footer>
      </Modal>

  <section class="bannerCards text-sm-start text-center p-4">
  <div class="container">
  <h2 style={{color:'white'}}>Visualizations</h2>
  <div class="row" style={{justifyContent:"center"}}>
  <motion.div onClick={handleShowG1} class="col-auto text-center"
  variants={cardVariants}
   initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
  >
      <p>Flight vs Ground Contact Times</p>
          <div className="carder chart-container" style={{cursor: "pointer"}}>
              <BarChart
              series={[
                //{ data: [0.3,0.4,0.25], label:'Ground'},
                //{ data: [0.8,0.9,0.76], label:'Flight' },
                { data: datas["ground"], label:'Ground'},
                { data: datas["flight"], label:'Flight'},
              ]}
              //height={300}
              //width={500}
              xAxis={[{ data: contactLabels, scaleType: 'band', label:'Contacts' }]}
              yAxis={[{ label: 'Time (sec)' }]}
              margin={{ top: 30, bottom: 40, left: 40, right: 10 }}
            />
          </div>
  </motion.div>
   {isToggled && (<motion.div class="col-auto text-center"
  variants={cardVariants}
   initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
  >
      <p>Relative Stride Length</p>
          <div className="carder chart-container">
          <BarChart
          series={[
            { data: [1.2,0.9,1.1], label:'Ground'},
          ]}
          xAxis={[{ data: ['C1', 'C2', 'C3'], scaleType: 'band', label:'Contacts' }]}
          //yAxis={[{ label: 'Time (sec)' }]}
          margin={{ top: 30, bottom: 40, left: 40, right: 10 }}
        />
          </div>
  </motion.div>)}
  </div>
  <div class="row" style={{justifyContent:"center"}}>
  <motion.div class="col-auto text-center"
    variants={cardVariants}
   initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
  >
      <p>Angles over Frames</p>
     <div className="carder chart-container" >
      <LineChart
  series={[{ data: datas["ang"], label: 'Hip Angle', area: true, showMark: false }]}
  xAxis={[{ scaleType: 'point', data: angLabels, label:'Frames' }]}
  yAxis={[{ label:'Degrees' }]}
  sx={{
    [`& .${lineElementClasses.root}`]: {
      display: 'none',
    },
  }}
/>
      </div>
  </motion.div>
  <motion.div class="col-auto text-center"
    variants={cardVariants}
   initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
  >
      <p>Acceleration Smoothness</p>
     <div className="carder chart-container">
      <LineChart
          //xAxis={[{ data: [1, 2, 3, 5, 8, 10] }]}
          //xAxis={[{ data: xLineVals}]}
          xAxis={[{ data: datas["vT"], label:"Time"}]}
          yAxis={[{ label:'M/s' }]}
          series={[
            {
              //data: [1, 4, 2, 5, 8, 6],
              //data: yLineVals
              data: datas["vR"], curve: "linear",label: 'Angular Velocity Right Knee',
            },
          ]}
      />
      </div>
  </motion.div>
  </div>
  <div>
  <div className="analytics-panel pb-5">
          {analData.map((item, index) => (
            <motion.div
              key={index}
              className="analytics-card"
              whileHover={{ scale: 1.05 }}
               style={{ cursor: 'pointer', backgroundColor: item.color >= 1 ? 'green' : 'red', border: item.color >= 1 ? '2px solid darkgreen' : '2px solid darkred'}}
            >
              <div className="anal-card-content">
                <h4>{item.title}</h4>
                <h1>{item.value}</h1>
                <p>{item.unit}</p>
              </div>
            </motion.div>
          ))}
        </div>
  </div>
  <div className="analytics-panel pb-5">
          {analData2.map((item, index) => (
            <motion.div
             style={{ cursor: 'pointer',backgroundColor: item.color >= 1 ? 'green' : 'red', border: item.color >= 1 ? '2px solid darkgreen' : '2px solid darkred' }}
              key={index}
              className="analytics-card"
              whileHover={{ scale: 1.05 }}
              onClick={() => handleShowFeed(index)}
            >
              <div className="anal-card-content" >
                <h4>{item.title}</h4>
                <h1>{item.value}</h1>
                <p>{item.unit}</p>
              </div>
            </motion.div>
          ))}
  </div>
  </div>
  <Modal show={showFeed} onHide={handleCloseFeed} centered>
        <Modal.Header closeButton>
          <Modal.Title>Feedback</Modal.Title>
        </Modal.Header>
        <Modal.Body>
         {getFeedbackMessage(feedInd)}
        </Modal.Body>
        <Modal.Footer>
        </Modal.Footer>
      </Modal>
  </section>
  </div>
)}


</body>
  );
};

export default VideoUploader;
