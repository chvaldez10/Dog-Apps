import Grid from "@mui/material/Grid";
import { BreedCard } from "./BreedCard";
import { customParagraphStyle } from "./styles/commonStyles";
import { gridItemStyle } from "./styles/breedGridItem";
import BreedModal from "./BreedModal";

export const BreedGridItem = ({ index, breed }) => {
  return (
    <Grid item xs={2} sm={4} md={4} key={index} style={gridItemStyle}>
      <div style={{ textAlign: "center" }}>
        <BreedCard>
          <BreedModal breed={breed} index={index + 1} />
        </BreedCard>
        <p style={customParagraphStyle}>{breed.breed}</p>
      </div>
    </Grid>
  );
};
