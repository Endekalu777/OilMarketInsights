import React from 'react';

const Filters = () => {
  return (
    <div>
      <label>Date Range: </label>
      <input type="date" />
      <input type="date" />
      <button>Apply</button>
    </div>
  );
};

export default Filters;
