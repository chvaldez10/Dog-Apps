export const encyclopediaBoxStyle = {
  flexGrow: 1,
  p: 4,
  m: 4,
  maxWidth: "500px",
  margin: "auto",
  width: "100%",
  boxSizing: "border-box",
};

export const searchBoxStyle = {
  display: "flex",
  justifyContent: "center",
  mb: 2,
};

export const textFieldStyle = {
  borderRadius: "4px",
  width: "300px",
  "& label.Mui-focused": {
    // Text color when focused
    color: "black",
  },
  "& .MuiInput-underline:before": {
    // Underline color when not focused
    borderBottomColor: "#3E3232",
  },
  "& .MuiInput-underline:after": {
    // Underline color when focused
    borderBottomColor: "#ECB159",
  },
  "& .MuiInput-underline:hover:not(.Mui-disabled):before": {
    // Underline color on hover
    borderBottomColor: "#E4DEBE",
  },
  "&::selection": {
    // Text selection color
    backgroundColor: "#FFFFEC",
  },
};
