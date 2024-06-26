import React, { useState } from 'react';
import { Tooltip } from 'react-tooltip'

const PasteImage = () => {
  const [image, setImage] = useState(null);

  const handlePaste = (event) => {
    const items = event.clipboardData.items;
    for (let item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        const reader = new FileReader();
        reader.onload = (e) => {
          setImage(e.target.result);
        };
        reader.readAsDataURL(file);
      }
    }
  };

  return (
    <div
      className="paste-area"
      onPaste={handlePaste}
      style={{ border: '2px dashed #ccc', padding: '20px', textAlign: 'center', cursor: 'pointer' }}
      data-tooltip-id="my-tooltip" data-tooltip-content="This is used to compute step length and velocity"
    >
      <p style={{color:"black"}}>Click here and press Ctrl+V to paste image of measuring device</p>
                 <Tooltip id="my-tooltip" />
      {image && <img src={image} alt="Pasted" style={{ maxWidth: '100%', maxHeight: '400px', marginTop: '20px' }} />}
    </div>
  );
};

export default PasteImage;
