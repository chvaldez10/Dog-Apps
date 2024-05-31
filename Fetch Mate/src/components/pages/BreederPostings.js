import React from 'react';
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardMedia from "@mui/material/CardMedia";
import Typography from "@mui/material/Typography";
import './BreederPostings.css';

const BreederPostings = () => {
  return (
    <Box className="breeder-postings-container">
      <Grid container spacing={2} className="breeder-postings">
        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardMedia
              component="img"
              height="140"
              image="/labrador-retriever-puppies.png" // Placeholder image path
              alt="Kaya's Litter"
            />
            <CardContent>
              <Typography gutterBottom variant="h5" component="div">
                Kaya's Litter
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Labrador Retriever
              </Typography>
              {/* Placeholder for actual schedule data */}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardMedia
              component="img"
              height="140"
              image="/tibetan-mastiff-puppies.png" // Placeholder image path
              alt="Tika's Litter"
            />
            <CardContent>
              <Typography gutterBottom variant="h5" component="div">
                Tika's Litter
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Tibetan Mastiff
              </Typography>
              {/* Placeholder for actual health records */}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardMedia
              component="img"
              height="140"
              image="/pharaoh-hound-puppies.jpg" // Placeholder image path
              alt="Nef's Litter"
            />
            <CardContent>
              <Typography gutterBottom variant="h5" component="div">
                Nef's Litter
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Pharaoh Hound
              </Typography>
              {/* Placeholder for actual health records */}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default BreederPostings;
