import React from "react";
import { circularImageStyle, leftDiv } from "./styles/breedDetail";

export const LeftDetail = ({ imageUrl }) => {
  return (
    <div style={leftDiv}>
      <img src={imageUrl} alt="Breed" style={circularImageStyle} />
    </div>
  );
};
