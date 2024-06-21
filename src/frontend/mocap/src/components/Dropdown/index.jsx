import React, { useState } from 'react';

const Dropdown = ({ options }) => {
  const [selectedOption, setSelectedOption] = useState('Dropdown button');

  const handleSelect = (option) => {
    setSelectedOption(option);
  };

  return (
    <div className="dropdown">
      <button
        className="btn btn-secondary dropdown-toggle mt-3"
        type="button"
        id="dropdownMenuButton"
        data-bs-toggle="dropdown"
        aria-haspopup="true"
        aria-expanded="false"
      >
        {selectedOption}
      </button>
      <div className="dropdown-menu" aria-labelledby="dropdownMenuButton">
        {options.map((option, index) => (
          <a
            key={index}
            className="dropdown-item"
            onClick={() => handleSelect(option)}
          >
            {option}
          </a>
        ))}
      </div>
    </div>
  );
};

export default Dropdown;
