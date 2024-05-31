// RightPart.js
import React from "react";
import Typography from "@mui/material/Typography";
import {
  rightDiv,
  rightDivTypography,
  addImageStyle,
} from "./styles/breedDetail";

export const RightDetail = ({ summary }) => {
  return (
    <div style={rightDiv}>
      <Typography variant="body1" sx={rightDivTypography}>
        {summary}
      </Typography>
      <img
        src="/images/dog_adds/dog-add-1.jpg"
        alt="dog-add-1"
        style={addImageStyle}
      ></img>
    </div>
  );
};
