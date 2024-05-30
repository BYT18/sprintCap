import React, { useRef } from "react";
import {
  motion,
  useScroll,
  useSpring,
  useTransform,
  MotionValue
} from "framer-motion";
import "./styles.css"; // Import the CSS file for styling

// Custom hook to create a parallax effect
function useParallax(value, distance) {
  return useTransform(value, [0, 1], [-distance, distance]);
}

// Component to render an image with a parallax effect
function Image({ id }) {
  const ref = useRef(null); // Create a ref to track the DOM element
  const { scrollYProgress } = useScroll({ target: ref }); // Get the scroll progress for the element
  const y = useParallax(scrollYProgress, 300); // Create a parallax effect

  return (
    <section className="sec">
      <div ref={ref} className="secDiv">
        <img className="scrollImg" src={`https://t4.ftcdn.net/jpg/01/99/22/71/360_F_199227139_FCFEMYvJMfzebvcS7S0s3BGerCMpfIVl.jpg`} alt={`A London skyscraper ${id}`} className="parallax-image" />
      </div>
      <motion.h2 className="parallax-title" style={{ y }}>{`#00${id}`}</motion.h2>
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
