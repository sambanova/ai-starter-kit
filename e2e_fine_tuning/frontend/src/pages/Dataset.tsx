import { useEffect, useState } from "react";

import {
  Container,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  SelectChangeEvent,
  Typography,
} from "@mui/material";

import { useDatasetStore } from "@/stores/DatasetStore";

const Dataset = () => {
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const { datasets, fetchDatasets } = useDatasetStore();

  const handleSelectDataset = (event: SelectChangeEvent) => {
    setSelectedDataset(event.target.value as string);
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Welcome to the Dataset page
      </Typography>
      <Typography variant="body1" sx={{ mb: 4 }}>
        This is the Dataset page of the application. Here you will be able to
        upload a dataset and check the status of the process.
      </Typography>

      <FormControl fullWidth>
        <InputLabel>Dataset list</InputLabel>
        <Select
          label="Dataset list"
          value={selectedDataset}
          onChange={handleSelectDataset}
        >
          {datasets.map((ds) => (
            <MenuItem key={ds.id} value={ds.dataset_name}>
              {ds.dataset_name}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </Container>
  );
};

export default Dataset;
