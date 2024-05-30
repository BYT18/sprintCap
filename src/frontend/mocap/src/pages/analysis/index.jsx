// VideoUploader.js
import React, { useState, useRef } from 'react';
import './VideoUploader.css';
import { BarChart } from '@mui/x-charts/BarChart';
import { LineChart } from '@mui/x-charts/LineChart';
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

const VideoUploader = () => {
  const ref = useRef(null);
  const { scrollXProgress } = useScroll({ container: ref });
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState('');
  const [images, setImages] = useState([]);

  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoFile(file);
      setVideoURL(url);
    }
  };

  const handleImageUpload = (event) => {
    const files = Array.from(event.target.files);
    const newImages = files.map((file) => URL.createObjectURL(file));
    setImages([...images, ...newImages]);
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

  return (
  <body>
   <section class="banner text-sm-start text-center p-4">
    <motion.div class="container" initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{
        duration: 0.8,
        delay: 0.5,
        ease: [0, 0.71, 0.2, 1.01]
      }}>
      <div className="video-uploader">
          <h2>Upload and Display Video</h2>
          <input style={{color:'black'}} class="pb-3" type="file" accept="video/*" onChange={handleVideoUpload} />
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
          <img src={myImage1} className="img-fluid" alt="Image" />
          <div class="text-container">
              <h3 >Touch Down</h3>
              <ul>
                <li>Knees together</li>
                <li>Initial contact with outside of ball of foot</li>
              </ul>
          </div>
        </motion.li>
         <motion.li class="anli"
            whileHover={{scale:1.02}}
        >
            <img src={myImage2}/>
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
            <img src={myImage3}/>
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
            <img src={myImage4}/>
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
            <img src={myImage5}/>
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
  <h2 style={{color:'white'}}>Flight and Contact Times</h2>
      <div className="carder chart-container">
           <BarChart
      series={[
        { data: [0.3,0.4,0.25] },
        { data: [0.8,0.9,0.76] },
      ]}
      height={290}
      xAxis={[{ data: ['C1', 'C2', 'C3'], scaleType: 'band' }]}
      margin={{ top: 10, bottom: 30, left: 40, right: 10 }}
    />
      </div>
     <div className="carder chart-container">
      <LineChart
  xAxis={[{ data: [1, 2, 3, 5, 8, 10] }]}
  series={[
    {
      data: [2, 5.5, 2, 8.5, 1.5, 5],
    },
  ]}
  width={500}
  height={300}
/>
      </div>
    </motion.div>
  </section>

</body>
  );
};

export default VideoUploader;
