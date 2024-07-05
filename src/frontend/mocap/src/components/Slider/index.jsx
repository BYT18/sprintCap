import React, { useState } from 'react';
import './HeightSlider.css'; // Import the CSS file for styling

const HeightSlider = () => {
  const [height, setHeight] = useState(170); // Default height value (in cm)

  const handleSliderChange = (event) => {
    setHeight(event.target.value);
  };

  return (
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
  );
};

export default HeightSlider;
