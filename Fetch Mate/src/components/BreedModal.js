import React, { useState } from "react";
import Box from "@mui/material/Box";
// import Button from "@mui/material/Button";
import { useNavigate } from "react-router-dom";
import Typography from "@mui/material/Typography";
import Modal from "@mui/material/Modal";
import { breedImageStyle } from "./styles/breedGridItem";
import { modalStyle } from "./styles/modalStyle";

function BreedModal({ breed, index }) {
  const [selecteItem, setSelectedItem] = useState(null);
  const navigate = useNavigate();
  const [open, setOpen] = React.useState(false);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  const handleClick = (index, breedData) => {
    setSelectedItem(index);
    console.log(`Index ${index} clicked`);
    navigate(`/encyclopedia/${index}`, { state: { breedData } });
  };

  return (
    <>
      <img
        src={`${breed.svg_photo}`}
        alt="Breed"
        style={breedImageStyle}
        onClick={handleOpen}
      />
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box sx={modalStyle}>
          <Typography id="modal-modal-title" variant="h6" component="h2">
            Summary
          </Typography>
          <Typography
            id="modal-modal-description"
            sx={{ mt: 2 }}
            onClick={() => handleClick(index, breed)}
          >
            {breed.summary}
          </Typography>
        </Box>
      </Modal>
    </>
  );
}

export default BreedModal;
