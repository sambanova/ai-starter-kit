import { Container, Typography } from "@mui/material";

const Dataset = () => {
  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Welcome to the Dataset page
      </Typography>
      <Typography variant="body1">
        This is the Dataset page of the application. Here you will be able to
        upload a dataset and check the status of the process.
      </Typography>
    </Container>
  );
};

export default Dataset;
