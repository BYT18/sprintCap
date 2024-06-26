import React, { useRef } from "react";
import {
  motion,
  useScroll,
  useSpring,
  useTransform,
  MotionValue
} from "framer-motion";
import "./styles.css"; // Import the CSS file for styling

import myImage1 from "../../assets/s1.png";
import myImage2 from "../../assets/s22.png";
import myImage3 from "../../assets/s3.png";
import myImage4 from "../../assets/s4.png";
import myImage5 from "../../assets/s5.png";
import myImage6 from "../../assets/s6.png";

// Custom hook to create a parallax effect
function useParallax(value, distance) {
  return useTransform(value, [0, 1], [-distance, distance]);
}

// Component to render an image with a parallax effect
function Image({ id }) {
  const ref = useRef(null); // Create a ref to track the DOM element
  const { scrollYProgress } = useScroll({ target: ref }); // Get the scroll progress for the element
  const y = useParallax(scrollYProgress, 300); // Create a parallax effect

  const images = [myImage1,myImage2,myImage3,myImage4, myImage5, myImage6]
  const desc = ["Capture any video of your athlete","Upload and Analyze the Video","Receive Instant Feedback","Personalized Coaching Tips","Receive Data Invisible to Naked Eye", "Compare with Pros, Share, and Win"]

  const customSizeClass1 = (id === 2) ? "custom-size1" : "";
  const customSizeClass2 = (id === 6) ? "custom-size2" : "";

  return (
    <section className="sec">
      <div ref={ref} className="secDiv">
        <img className={`scrollImg ${customSizeClass1} ${customSizeClass2}`} src={images[id-1]} alt={`A London skyscraper ${id}`}/>
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
      {[1, 2, 3, 4, 5, 6].map((image) => (
        <Image key={image} id={image} />
      ))}
      <motion.div className="progress" style={{ scaleX }} />
    </div>
  );
};

export default Kin;
