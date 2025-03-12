import { create } from "zustand";

import { axiosClient } from "@/utils/axiosClient";

type DatasetType = {
  id: string;
  dataset_name: string;
};

type DatasetStoreType = {
  datasets: DatasetType[];
  isLoadingFetch: boolean;
  isLoadingCreate: boolean;
  error: string | null;
  fetchDatasets: () => Promise<void>;
};

const baseRoute = "/dataset";

export const useDatasetStore = create<DatasetStoreType>((set) => ({
  datasets: [],
  isLoadingFetch: false,
  isLoadingCreate: false,
  error: null,

  setError: (error: string | null) => set({ error }),

  setIsLoading: (isLoading: boolean) => set({ isLoadingFetch: isLoading }),

  fetchDatasets: async () => {
    set({ isLoadingFetch: true, error: null });

    try {
      const datasets: DatasetType[] = await axiosClient
        .get(baseRoute)
        .then((res) => res.data.datasets);
      console.log(datasets);

      set({ datasets });
    } catch (err) {
      console.error(err);
    }
  },

  createDataset: async () => {
    set({ isLoadingCreate: true, error: null });

    try {
      const response = await axiosClient.post(baseRoute, {
        dataset_name_sambastudio: "publichealth-testing-davidp",
        dataset_description: "Q&A pages and FAQs",
        dataset_split: "train",
        data_files: [
          "english.csv",
          "spanish.csv",
          "french.csv",
          "russian.csv",
          "chinese.csv",
        ],
        dataset_job_types: ["evaluation", "train"],
        dataset_source_type: "localMachine",
        dataset_language: "english",
        dataset_filetype: "hdf5",
        dataset_url: "https://huggingface.co/datasets/xhluca/publichealth-qa",
        dataset_metadata: {},
        hf_dataset: "xhluca/publichealth-qa",
        model_family: "llama3",
        tokenizer: "lightblue/suzume-llama-3-8B-multilingual",
        max_seq_length: 8192,
        shuffle: "on_RAM",
        input_packing_config: "single::truncate_right",
        prompt_keyword: "prompt",
        completion_keyword: "completion",
        num_training_splits: 8,
        apply_chat_template: false,
      });
      console.log(response);
    } catch (err) {}
  },
}));
