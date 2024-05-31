import React from "react";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid";
// import { useNavigate } from "react-router-dom";
import { TextField } from "@mui/material";
import { BreedGridItem } from "../BreedGridItem";
import {
  encyclopediaBoxStyle,
  searchBoxStyle,
  textFieldStyle,
} from "../styles/encyclopediaStyles";

import dogData from "./../data/top_breed_data.json";

function Encyclopedia() {
  console.log(dogData);
  return (
    <Box sx={encyclopediaBoxStyle}>
      {/* Search bar component */}
      <Box sx={searchBoxStyle}>
        <TextField
          id="encyclopedia-search"
          label="Search for dog breed"
          variant="standard"
          sx={textFieldStyle}
        />
      </Box>

      {/* Featured Dogs */}
      <Grid
        container
        spacing={{ xs: 2, md: 3 }}
        columns={{ xs: 4, sm: 8, md: 12 }}
      >
        {Object.values(dogData).map((breed, index) => {
          return <BreedGridItem key={index} index={index} breed={breed} />;
        })}
      </Grid>
    </Box>
  );
}

export default Encyclopedia;
