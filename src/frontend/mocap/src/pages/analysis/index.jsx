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

import 'bootstrap/dist/css/bootstrap.min.css';
import "bootstrap-icons/font/bootstrap-icons.css";
import 'bootstrap/dist/js/bootstrap.bundle.min';
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap/dist/js/bootstrap.bundle.min.js";
import * as Icon from 'react-bootstrap-icons';



/*const data = [
  { x: 100, y: 200, id: 1 },
  { x: 120, y: 100, id: 2 },
  { x: 170, y: 300, id: 3 },
  { x: 140, y: 250, id: 4 },
  { x: 150, y: 400, id: 5 },
  { x: 110, y: 280, id: 6 },
];*/
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

/*const analData = [
  { title: 'Time between steps', value: '0.24', unit: 'SECONDS'},
  { title: 'Max step length', value: '1.08', unit: 'METS'},
  { title: 'Mean Ground Time', value: avgG, unit: 'SECONDS'},
  { title: 'Mean Flight Time', value: avgF, unit: 'SECONDS'}
];*/

const VideoUploader = () => {
  const ref = useRef(null);
  const { scrollXProgress } = useScroll({ container: ref });
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState('');
  const [images, setImages] = useState([]);
  const [testImg, setTestImg] = useState(null);
  const [resultVid, setResultVid] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pasteImage, setPasteImage] = useState(null);

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
  const [avgG, setAvgG] = useState(null);
  const [avgF, setAvgF] = useState(null);
  const [contactLabels, setContactLabels] = useState(null);
  const [angLabels, setAngLabels] = useState(null);


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
        setLoading(true);
        const formData = new FormData();
        formData.append("vid", e);

        const blob = await fetch(p).then(res => res.blob()); // Convert base64 to Blob
        formData.append('pic', blob, 'image.png'); // Add blob to formData with filename
        //formData.append("pic", p);
        console.log(e)
        //console.log(p)
        try {
            // Create the POST request using the fetch API
            const response = await fetch('http://127.0.0.1:8000/test/', {
            //const response = await fetch('http://3.143.116.75:8000/test/', {
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

                // Generate xAxis labels
                const xAxisLabels = data.x_vals["ground"].map((_, index) => `C${index + 1}`);
                setContactLabels(xAxisLabels)

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
                addImage(data.kin1);
                addImage(data.kin2);
                addImage(data.kin3);
                addImage(data.kin4);
                addImage(data.kin5);


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
                setVelDataL([data.x_vals["ang"],data.x_vals["ang"]])
                //setVelDataL([data.x_vals["vL"],data.x_vals["vR"][1]])
                console.log(velDataL)

                const augmentedData = [
                  { title: 'Time between steps', value: '???', unit: 'SECONDS'},
                  { title: 'Max step length', value: Number(data.x_vals["maxSL"]).toFixed(3), unit: 'METERS'},
                  { title: 'Mean Ground Time', value: Number(data.x_vals["avg_g"]).toFixed(3), unit: 'SECONDS'},
                  { title: 'Mean Flight Time', value: Number(data.x_vals["avg_f"]).toFixed(3), unit: 'SECONDS'}
                  // Add more mappings as needed
                ];

                setAnalData(augmentedData);
                setLoading(false);
                //setGif1(data.pic)
                //console.log(gif1)
            } else {
                // Handle error responses
                console.error('Error:', response.statusText);
            }
        } catch (error) {
            // Handle network errors
            console.error('Network error:', error.message);
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
  }, [videoFile]);

  const downloadImage = async () => {
        const feedbacks = [textAreaToe,textAreaV,textAreaS,textAreaT,textAreaFS]
        console.log(feedbacks)
        const combinedImage = await createCombinedImage(images, feedbacks);
        saveAs(combinedImage, 'womp.jpg')
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

  return (
  <body>
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
          <input class="form-control mb-2" type="file" id="formFileMultiple" accept="video/*" onChange={handleVideoUpload}/>
          {/*<input style={{color:'black'}} class="pb-3" type="file" accept="video/*" onChange={handleVideoUpload} />*/}
          {videoURL && (
            <div className="video-container">
              <video controls src={videoURL}>
                Your browser does not support the video tag.
              </video>
            </div>
          )}
        <div className="container mt-2" data-tooltip-id="my-tooltip" data-tooltip-content="Change analysis depending on sprinting phase being recorded">
          <select class="form-select" aria-label="Default select example">
              <option selected>Top End Analysis</option>
              <option value="1">Acceleration Analysis</option>
              <option value="2">Tempo Stride Analysis</option>
          </select>
          <Tooltip id="my-tooltip" />
        </div>
        <div className="container mt-4">
          <div
              className="paste-area"
              onPaste={handlePaste}
              style={{ border: '2px dashed #ccc', padding: '20px', textAlign: 'center', cursor: 'pointer' }}
              data-tooltip-id="my-tooltip" data-tooltip-content="This is used to compute step length and velocity"
            >
              <p style={{color:"black"}}>Click here and press Ctrl+V to paste image of measuring device</p>
                         <Tooltip id="my-tooltip" />
              {pasteImage && <img src={pasteImage} alt="Pasted" style={{ maxWidth: '100%', maxHeight: '400px', marginTop: '20px' }} />}
            </div>
        </div>
        <div className="container mt-4">
          <button type="button" class="btn btn-secondary" onClick={handleAnalyze}>
            Analyze
          </button>
        </div>
      </div>
    </motion.div>
  </section>
   {loading && <Loader />}
  {resultVid && (
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
          <img src={images[0]} className="img-fluid" alt="Image" />
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
            <img src={images[1]}/>
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
            <img src={images[2]}/>
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
            <img src={images[3]}/>
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
            <img src={images[4]}/>
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
  <section class="bannerCards text-sm-start text-center p-4">
  <motion.div class="container"
   variants={cardVariants}
   initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
  >
  <h2 style={{color:'white'}}>Visualizations</h2>
  <div class="row" style={{justifyContent:"center"}}>
  <div class="col-auto text-center">
      <p>Flight vs Ground Contact Times</p>
          <div className="carder chart-container">
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
  </div>
  <div class="col-auto text-center">
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
  </div>
  </div>
  <div class="row" style={{justifyContent:"center"}}>
  <div class="col-auto text-center">
      <p>Angles over Frames</p>
     <div className="carder chart-container" >
      <LineChart
  series={[{ data: datas["ang"], label: 'Hip Angle', area: true, showMark: false }]}
  xAxis={[{ scaleType: 'point', data: angLabels }]}
  sx={{
    [`& .${lineElementClasses.root}`]: {
      display: 'none',
    },
  }}
/>
      </div>
  </div>
  <div class="col-auto text-center">
      <p>Acceleration Smoothness</p>
     <div className="carder chart-container">
      <LineChart
          //xAxis={[{ data: [1, 2, 3, 5, 8, 10] }]}
          //xAxis={[{ data: xLineVals}]}
          xAxis={[{ data: datas["vT"]}]}
          series={[
            {
              //data: [1, 4, 2, 5, 8, 6],
              //data: yLineVals
              data: datas["vL"], curve: "linear",label: 'Velocity Left Knee',
            },
            { data: datas["vR"], curve: "linear",label: 'Velocity Right Knee',}
          ]}
      />
      </div>
  </div>
  </div>
  <div>
  <div className="analytics-panel pb-5">
          {analData.map((item, index) => (
            <motion.div
              key={index}
              className="analytics-card"
              whileHover={{ scale: 1.05 }}
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
    </motion.div>
  </section>
   </div>
)}


</body>
  );
};

export default VideoUploader;
