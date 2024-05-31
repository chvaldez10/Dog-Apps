import * as React from "react";
import Grid from "@mui/material/Grid";
import Box from "@mui/material/Box";
import { useParams } from "react-router-dom";
import { breedDetailBoxStyle } from "../styles/breedDetail";
import { LeftDetail } from "../LeftDetail";
import { RightDetail } from "../RightDetail";

import testDogData from "./../data/top_breed_data.json";

function BreedDetail() {
  let { dogID } = useParams();
  let dogData = testDogData[dogID];

  return (
    <Box sx={breedDetailBoxStyle}>
      <Grid container>
        <Grid item xs={12} sm={4}>
          <LeftDetail imageUrl={dogData?.real_photo} />
        </Grid>
        <Grid item xs={12} sm={8}>
          <RightDetail summary={dogData?.summary} />
        </Grid>
      </Grid>
    </Box>
  );
}

export default BreedDetail;
