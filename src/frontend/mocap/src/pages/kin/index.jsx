import React, { useRef } from "react";
import {
  motion,
  useScroll,
  useSpring,
  useTransform,
  MotionValue
} from "framer-motion";
import "./styles.css"; // Import the CSS file for styling

import myImage1 from "../../assets/touchdown_R2.png";
import myImage2 from "../../assets/max_ver3.png";
import myImage3 from "../../assets/strike_R2.png";
import myImage4 from "../../assets/toe_off_L2.png";
import myImage5 from "../../assets/full_sup_L2.png";

// Custom hook to create a parallax effect
function useParallax(value, distance) {
  return useTransform(value, [0, 1], [-distance, distance]);
}

// Component to render an image with a parallax effect
function Image({ id }) {
  const ref = useRef(null); // Create a ref to track the DOM element
  const { scrollYProgress } = useScroll({ target: ref }); // Get the scroll progress for the element
  const y = useParallax(scrollYProgress, 300); // Create a parallax effect

  const images = ["https://www.questsportscanada.club/images/creatives/uncutted-img-content-1.png",myImage2,"https://cdn.britannica.com/96/129796-050-7EA034EF/Usain-Bolt-2008.jpg","https://femalefitnesssystems.com/wp-content/uploads/2023/07/online-training-for-women-2.jpg","https://melbournefootclinic.com.au/wp-content/uploads/2022/08/image1.jpg"]
  const desc = ["Capture any video of your athlete","Get instant analysis and personalized coaching feedback","Compare with pros","Train and finetune","Take care of all the details"]

  return (
    <section className="sec">
      <div ref={ref} className="secDiv">
        <img className="scrollImg" src={images[id-1]} alt={`A London skyscraper ${id}`}/>
      </div>
      <motion.h2 className="parallax-title" style={{ y }}>{`Phase ${id}`}</motion.h2>
      <motion.p className="parallax-text" style={{ y }}>{desc[id-1]}</motion.p>
    </section>
  );
}

// Main component to render a series of parallax images
const Kin = () => {
  const { scrollYProgress } = useScroll(); // Track the scroll progress of the page
  const scaleX = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001
  }); // Create a spring animation for the progress bar

  return (
    <div className="kin-container">
      {[1, 2, 3, 4, 5].map((image) => (
        <Image key={image} id={image} />
      ))}
      <motion.div className="progress" style={{ scaleX }} />
    </div>
  );
};

export default Kin;
