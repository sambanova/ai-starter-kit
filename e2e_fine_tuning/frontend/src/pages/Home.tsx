import { Container, Typography } from "@mui/material";

const HomePage = () => {
  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Welcome to the Home Page
      </Typography>
      <Typography variant="body1">
        This is the Home page of the application. Here will be displayed general
        information of what to do here.
      </Typography>
    </Container>
  );
};

export default HomePage;
