from datasets import load_dataset # type: ignore
import logging 
import os
import json
import uuid
import numpy as np

logging.basicConfig(level=logging.INFO)

class LlaVaData:

    "Download huggingface datasets and save to local storage."

    def download_hf_data_documentVQA(self, 
                         dataset_name: str = "HuggingFaceM4/DocumentVQA", 
                         output_dir: str = "./",
                         split: str = "train",
                         ) -> None:
        
        """
        Download a dataset from Hugging Face datasets and save to a user specified location  
        with the correct file structure expected by SambaStudio.  Expects download_hf_data_documentVQA
        data structure: https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA

        Args:
            dataset_name: Name of the dataset to download.
            output_dir: Directory to save the dataset.
            split: Which split of the dataset to download.

        Raises:
            TypeError: If `dataset_name`, `split`, or `output_dir` is not a string.
            ValueError: If `split` is not one of ['train', 'validation', 'test'].
        """
        
        # Check inputs
        assert isinstance(dataset_name, str), \
            TypeError(f"dataset_name must be a string.  Got {type(dataset_name)}")
        assert isinstance(output_dir, str), \
            TypeError(f"output_dir must be a string.  Got {type(output_dir)}")
        assert isinstance(split, str), \
            TypeError(f"split must be a string.  Got {type(split)}")
        assert split in ["train", "validation", "test"], \
            ValueError(f"split must be one of ['train', 'validation', 'test'].  Got {split}")
        
        logging.info(f"Downloading {dataset_name} dataset")
        # Load dataset from HuggingFace
        ds = load_dataset(dataset_name)

        logging.warning(f"This dataset is of length: {len(ds[split])} \
        and may take some time to process.")

        img_dir: str = os.path.join(output_dir, "data", "images")
        if split == "validation":
            json_path: str = os.path.join(output_dir, "data", f"annotations_val.json")
        else:
            json_path = os.path.join(output_dir, "data", f"annotations_{split}.json")

        # Create data directory if it doesn't exist
        if not os.path.exists(os.path.join(output_dir, "data")):
            os.makedirs(os.path.join(output_dir, "data"))
        with open(json_path, "w") as f:
            json.dump([], f)

        # Create image directory if it doesn't exist
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        logging.info(f"Saving {split} to {os.path.join(output_dir, 'data')}")

        # Create a unique identifier for each row.  
        # Name the image as the UUID for simplicity.
        for i in range(len(ds[split])):
            unique_uuid: str = str(uuid.uuid4())
            image = ds[split][i]["image"]
            
            image.save(os.path.join(img_dir, f"{unique_uuid}.png"))

            # Here we add the image token to the question, per SN guidance. \
            # Would like to use pydantic for typing, but "from" key is problematic.
            question: str = ds[split][i]["question"] + "\n<image>"
            # Lazily take one of the potential answers.
            answer: str = np.random.choice(ds[split][i]["answers"])

            new_data = {
                "id": unique_uuid,
                "image": f"{unique_uuid}.png",
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer}
                ]
            }

            # Append new data to json file
            with open(json_path, "r+") as f:
                data = json.load(f)
                data.append(new_data)
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
