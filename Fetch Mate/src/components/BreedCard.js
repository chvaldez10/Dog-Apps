import { experimentalStyled as styled } from "@mui/material/styles";
import Paper from "@mui/material/Paper";

export const BreedCard = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.mode === "dark" ? "#1A2027" : "#fffffff",
  ...theme.typography.body2,
  padding: theme.spacing(2),
  textAlign: "center",
  color: theme.palette.text.secondary,
  width: theme.spacing(12),
  height: theme.spacing(12),
  borderRadius: "50%",
  overflow: "hidden",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  position: "relative",
}));
