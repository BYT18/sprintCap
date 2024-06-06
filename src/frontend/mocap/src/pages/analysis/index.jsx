// VideoUploader.js
import React, { useState, useRef, useEffect} from 'react';
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
import myImage1 from "../../assets/touchdown_R2.png";
import myImage2 from "../../assets/max_ver3.png";
import myImage3 from "../../assets/strike_R2.png";
import myImage4 from "../../assets/toe_off_L2.png";
import myImage5 from "../../assets/full_sup_L2.png";

import Loader from '../../components/Loader/index.jsx';

const data = [
  { x: 100, y: 200, id: 1 },
  { x: 120, y: 100, id: 2 },
  { x: 170, y: 300, id: 3 },
  { x: 140, y: 250, id: 4 },
  { x: 150, y: 400, id: 5 },
  { x: 110, y: 280, id: 6 },
];

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

const analData = [
  { title: 'Time between steps', value: '0.24', unit: 'SECONDS'},
  { title: 'Max step length', value: '1.08', unit: 'METS'},
  { title: 'Mean Ground Time', value: '0.22', unit: 'SECONDS'},
  { title: 'Mean Flight Time', value: '0.12', unit: 'SECONDS'}
];

const VideoUploader = () => {
  const ref = useRef(null);
  const { scrollXProgress } = useScroll({ container: ref });
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState('');
  const [images, setImages] = useState([]);
  const [testImg, setTestImg] = useState(null);
  const [resultVid, setResultVid] = useState(null);
  const [loading, setLoading] = useState(false);

  const [chartData, setChartData] = useState({});

  const addImage = (newImage) => {
      setImages(prevImages => [...prevImages, newImage]);
  };


  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoFile(file);
      setVideoURL(url);
      analyze(file);
    }
  };

  const handleImageUpload = (event) => {
    const files = Array.from(event.target.files);
    const newImages = files.map((file) => URL.createObjectURL(file));
    setImages([...images, ...newImages]);
  };

   const analyze = async (e) => {
        setLoading(true);
        const formData = new FormData();
        formData.append("vid", e);
        console.log(e)
        try {
            // Create the POST request using the fetch API
            //const response = await fetch('http://127.0.0.1:8000/test/', {
            const response = await fetch('http://18.216.114.197:8000/test/', {
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
                /*const xValues = data.map(point => point.x_value);
                const yValues = data.map(point => point.y_value);*/
                const xValues = data.x_vals;
                const yValues = data.x_vals;

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
                });
                setResultVid(data.pic)
                //setImages(data.pic)
                setTestImg(data.pic)
                addImage(data.kin1);
                addImage(data.kin2);
                addImage(data.kin3);
                addImage(data.kin4);
                addImage(data.kin5);
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
      </div>
    </motion.div>
  </section>
   {loading && <Loader />}
  {resultVid && (
  <div>
  <section class="bannerCards text-sm-start text-center p-4">
      <div className="video-uploader">
      <h2>Fitted</h2>
            <video controls src={resultVid}>
              Your browser does not support HTML5 video.
            </video>
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
              <ul>
                <li>90 dorsiflexion in swing foot</li>
                <li>Front shin parallel to rear thigh</li>
              </ul>
          </div>
        </motion.li>
         <motion.li class="anli"
            whileHover={{scale:1.02}}
        >
            <img src={images[1]}/>
            <div class="text-container">
              <h3 >Max Vertical Projection</h3>
              <ul>
                <li>Neutral head</li>
                <li>>110 degrees knee angle in front leg</li>
              </ul>
            </div>
        </motion.li>
         <motion.li class="anli"
            whileHover={{scale:1.02}}
        >
            <img src={images[2]}/>
            <div class="text-container">
              <h3 >Strike</h3>
              <ul>
                <li>20-40 degree gap between thighs</li>
                <li>Front foot beginning to supinate</li>
              </ul>
          </div>
        </motion.li>
         <motion.li class="anli"
            whileHover={{scale:1.02}}
        >
            <img src={images[3]}/>
            <div class="text-container">
              <h3 >Touch Down</h3>
              <ul>
                <li>Knees together</li>
                <li>Initial contact with outside of ball of foot</li>
                <li>Initial contact with outside of ball of foot</li>
              </ul>
          </div>
        </motion.li>
         <motion.li class="anli"
            whileHover={{scale:1.02}}
        >
            <img src={images[4]}/>
            <div class="text-container">
              <h3 >Full Support</h3>
              <ul>
                <li>Swing foot tucked under glutes</li>
                <li>Stance amortization occurs more at ankles than hips or knee</li>
              </ul>
          </div>
        </motion.li>
      </ul>
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
  <div class="row">
  <div class="col text-center">
      <p>Flight vs Ground Contact Times</p>
          <div className="carder chart-container">
          <BarChart
          series={[
            { data: [0.3,0.4,0.25], label:'Ground'},
            { data: [0.8,0.9,0.76], label:'Flight' },
          ]}
          height={300}
             width={500}
          xAxis={[{ data: ['C1', 'C2', 'C3'], scaleType: 'band', label:'Contacts' }]}
          yAxis={[{ label: 'Time (sec)' }]}
          margin={{ top: 30, bottom: 40, left: 40, right: 10 }}
        />
          </div>
  </div>
  <div class="col text-center">
      <p>Relative Stride Length</p>
          <div className="carder chart-container">
          <BarChart
          series={[
            { data: [1.2,0.9,1.1], label:'Ground'},
          ]}
          height={300}
             width={500}
          xAxis={[{ data: ['C1', 'C2', 'C3'], scaleType: 'band', label:'Contacts' }]}
          //yAxis={[{ label: 'Time (sec)' }]}
          margin={{ top: 30, bottom: 40, left: 40, right: 10 }}
        />
          </div>
  </div>
  </div>
  <div class="row">
  <div class="col text-center">
      <p>Angles over Frames</p>
     <div className="carder chart-container" >
      <LineChart
  width={500}
  height={300}
  series={[{ data: uData, label: 'uv', area: true, showMark: false }]}
  xAxis={[{ scaleType: 'point', data: xLabels }]}
  sx={{
    [`& .${lineElementClasses.root}`]: {
      display: 'none',
    },
  }}
/>
      </div>
  </div>
  <div class="col text-center">
      <p>Acceleration Smoothness</p>
     <div className="carder chart-container">
      <LineChart
          xAxis={[{ data: [1, 2, 3, 5, 8, 10] }]}
          series={[
            {
              data: [1, 4, 2, 5, 8, 6],
            },
          ]}
          width={500}
          height={300}
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
