# This work has been based off of the great tutorial by Fenglu Hong @SambaNova Systems.
# The workflow has been simplified for ease of use.  However, it is still under development.

import os
import cv2
import glob
import json
from langchain_core.prompts import load_prompt, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A # type: ignore
from albumentations.core.composition import Compose # type: ignore
from pdf2image import convert_from_path
import random
import subprocess
import time
from typing import Any, Dict, List, Tuple
from ultralyticsplus import YOLO # type: ignore
import uuid
import yaml # type: ignore

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '../..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

from utils.model_wrappers.api_gateway import APIGateway # type: ignore
from yoda.llava_data_prep.table_templates.sec_edgar.AAPL import synth_tables # type: ignore
from yoda.prompts.table_qa_template import tableqa_template # type: ignore


logging.basicConfig(level=logging.INFO)

DOCUMENT_PREFIX_LANDSCAPE = r'''\documentclass[8pt]{article}
\usepackage[landscape, margin={MARGIN}in]{geometry} % Sets the document to landscape mode
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{array}
\usepackage{colortbl} 

\usepackage{multirow}
\usepackage{booktabs}

\usepackage{FONT_TYPE}
\renewcommand{\familydefault}{\sfdefault}
\setlength{\arrayrulewidth}{{RULE_WIDTH}mm}
\begin{document}
'''

DOCUMENT_PREFIX = r'''\documentclass[8pt]{article}
\usepackage[margin={MARGIN}in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{array}
\usepackage{colortbl} 

\usepackage{multirow}
\usepackage{booktabs}

\usepackage{FONT_TYPE}
\renewcommand{\familydefault}{\sfdefault}
\setlength{\arrayrulewidth}{{RULE_WIDTH}mm}
\begin{document}
'''

class TableTools:

    def convert_pdf_to_images(self, data_directory: str) -> None:
            """
            This method converts a pdf file to a series of images of each page
            
            Args:
                data_directory: Base path to save images.

            Raises:
                TypeError: If `data_directory` is not a string.
                FileNotFoundError: If `data_directory` does not exist.
            """

            assert isinstance(data_directory, str), \
                TypeError(f"Expected str, got {type(data_directory)}")

            # Get all pdf files in the directory and its 
            # subdirectories recursively
            try:
                files: List[str] = glob.glob(data_directory + "**/**",
                                        recursive=True)
                files = [file for file in files if file.endswith(".pdf")]
            except FileNotFoundError:
                logging.error(f"{data_directory} not found")
                raise FileNotFoundError(f"{data_directory} does not exist.")

            for filename in files:
                logging.info(f"Converting {filename} to a folder of \
                             images in the same location")

                # Convert pdf to images and save them in the images 
                # subdirectory.
                images: list = convert_from_path(filename)
                base_folder: str = filename.split('.')[0]
                output_folder: str = os.path.join(base_folder, "images")
                os.makedirs(output_folder, exist_ok=True)
                for i, image in enumerate(images):
                    img_name=f"page_{i}.jpg"
                    output_path: str = f"{output_folder}/{img_name}"
                    image.save(output_path, 'JPEG')

    def crop_tables(self,
                    data_directory: str, 
                    conf: float = 0.25,
                    iou: float = 0.45,
                    agnostic_nms: bool = False,
                    max_det: int = 1000,
                    threshold: float = 0.8,
                    offset: int = 20) -> None:
        
        """
        This method crops tables from images using YOLOv8
        Args:
            data_directory: directory of images

        Raises:
            TypeError: If `data_directory` is not a string.
            TypeError: If `conf`, `iou`, `agnostic_nms`, `max_det`, 
                `threshold`, `offset` are not floats.
            TypeError: If `agnostic_nms`, `max_det` are not booleans.
            TypeError: If `max_det` is not an integer.
            FileNotFoundError: If `data_directory` does not exist.
        """

        assert isinstance(data_directory, str), \
            TypeError(f"Expected str, got {type(data_directory)}")
        assert isinstance(conf, float), \
            TypeError(f"Expected float, got {type(conf)}")
        assert isinstance(iou, float), \
            TypeError(f"Expected float, got {type(iou)}")
        assert isinstance(agnostic_nms, bool), \
            TypeError(f"Expected bool, got {type(agnostic_nms)}")
        assert isinstance(max_det, int), \
            TypeError(f"Expected int, got {type(max_det)}")
        assert isinstance(threshold, float), \
            TypeError(f"Expected float, got {type(threshold)}")
        assert isinstance(offset, int), \
            TypeError(f"Expected int, got {type(offset)}")
        
        # YOLO works quite well and has less issues than PaddleOCR tools.
        model = YOLO('foduucom/table-detection-and-extraction')

        # set model parameters
        model.overrides['conf'] = conf  # NMS confidence threshold
        model.overrides['iou'] = iou  # NMS IoU threshold
        model.overrides['agnostic_nms'] = agnostic_nms  # NMS class-agnostic
        model.overrides['max_det'] = max_det  # maximum number of detections per image

        # Get all files in directory recursively
        try:
            files: List[str] = glob.glob(data_directory + "**/**", recursive=True)
            files = [file for file in files if file.endswith(".jpg")]
        except FileNotFoundError:
            logging.error(f"{data_directory} not found.")
            raise FileNotFoundError(f"{data_directory} does not exist.")

        # Iterate over all files and crop tables from them
        for filename in files:
            logging.info(f"Cropping tables from {filename}")
            results: Any = model.predict(filename)
            
            # Get bounding boxes
            boxes: Any = results[0].boxes.xyxy.numpy(),
            
            # Load image.
            loaded_img: Image.Image = Image.open(filename)
            output_folder: str = filename.partition('images')[0] + "cropped_tables/"

            # Create output folder if it doesn't exist.
            os.makedirs(output_folder, exist_ok=True)

            # Crop tables and save them to output folder.
            for i, box in enumerate(boxes):
                cropped_table: Image.Image = loaded_img.crop((box[0]-offset, 
                                                box[1]-offset, 
                                                box[2]+offset, 
                                                box[3]+offset))
                page_no: str = filename.partition('images')[-1].replace("/", "").replace(".jpg", "")
                output_path: str = f"{output_folder}/{page_no}_table_{i}.jpg"
                cropped_table.save(output_path, "JPEG")
        logging.info(f"Cropped tables saved to {data_directory} \
                     in subdirectories.")

    def replace_special_to_latex(self, text: str) -> str:
        
        """
        Replace special characters with their LaTeX equivalents.

        Args:
            text: Text to be converted.

        Returns:
            Converted text.

        Raises:
            TypeError: If `text` is not a string.
        """

        assert isinstance(text, str), \
            TypeError(f"Expected str, got {type(text)}.")
        
        text = text.replace("%", "\\%")
        text = text.replace("_", "\\_")
        text = text.replace("*", "\\quad ")
        text = text.replace("$", "\\$")

        return text
    
    def _return_latex_text(self, text: str) -> str:

        """
        Return LaTeX formatted text from given text.

        Args:
            text: Text to be converted.

        Returns:
            Converted text.

        Raises:
            TypeError: If `text` is not a string.
        """

        assert isinstance(text, str), \
            TypeError(f"Expected str, got {type(text)}.")
        
        latex_text: str = ""

        for line in text.splitlines():
            line = line.replace("\t", " & ")
            line += " \\\\\n"
            latex_text += line

        return latex_text
    
    def _return_colors(self,
                       colors: List[Tuple[Any, ...]] = [
                           (150,210,255),
                           (210,210,210),
                           (175,250,180),
                           (250,250,180),
                       ]) -> List[str]:

        """
        Returns a list of RGB color codes.

        Args:
            colors: A list of RGB color codes as tuples.

        Returns:
            A list of RGB color codes in LaTeX format for table coloring.

        Raises:
            TypeError: If `colors` is not a list.
        """

        assert isinstance(colors, list), \
            TypeError(f"Expected List, got {type(colors)}.")

        colors_list: List[str] = [f"\\definecolor{{mycolor}}{{RGB}}{{{','.join(map(str, color))}}}" \
                  for color in colors]
        
        return colors_list
    
    def _randomize_colors(self, text: str) -> str:

        """
        Randomly colorize rows of a table.

        Args:
            text: LaTeX text of a table.

        Returns:
            LaTeX text with randomly colored rows.
        """

        # Get a random number for color selection.
        r: float = np.random.uniform(0, 1)

        latex_text = ""

        for i ,line in enumerate(text.splitlines()):

            line = line.replace("\t", " & ")

            # If between 0.5 and 0.75, Colorize every other row
            # starting from second position.
            if 0.5 <= r < 0.75:
                if i%2 == 0:
                    line = "\\rowcolor{mycolor}\n" + line

            # If between 0.75 and 1.0, Colorize every other row
            # starting from first position
            if 0.76 <= r < 1.0:
                if i%2 == 1:
                    line = "\\rowcolor{mycolor}\n" + line

            line += " \\\\\n"
            latex_text += line

        return latex_text
    
    def _randomly_add_vert_lines(self, 
                                 table_string: str, 
                                 prob: float
                                 ) -> str:

        """
        Randomly add vertical lines between columens in a LaTeX table.

        Args:
            table_string: LaTeX string of a table.
            prob: Probability of adding a vertical line.

        Returns:
            LaTeX string with randomly added vertical lines.

        Raises:
            TypeError: If `table_string` is not a string.
            TypeError: If `prob` is not a float.
        """

        assert isinstance(table_string, str), \
            TypeError(f"Expected str, got {type(table_string)}.")
        assert isinstance(prob, float), \
            TypeError(f"Expected float, got {type(prob)}.")

        # Split the string into lines
        lines: List[str] = table_string.split('\n')

        # Find the line with columns
        for i, line in enumerate(lines):
            if '{' in line and '}' in line:
                column_definitions: str = line
                break

        # Split the column definitions into individual columns
        columns: List[str] = column_definitions.strip().strip('{}').split()

        # Randomly add pipes between the columns
        new_columns: list = []
        for column in columns:
            if column in ['l', 'c', 'r']:
                if random.random() < prob:  # chance of adding a pipe
                    new_columns.append(column + ' |')
                else:
                    new_columns.append(column)
            else:
                new_columns.append(column)

        # Join the new columns back into a string
        new_column_definitions: str = '{ ' + ' '.join(new_columns) + ' }'

        # Replace the old column definitions with the new ones
        lines[i] = new_column_definitions

        # Join the lines back into a single string
        new_table_string: str = '\n'.join(lines)

        return new_table_string

    def convert_tsv_to_latex(self, 
                             column_style: str,
                             text: str,
                             randomize_colors: bool = True,
                             randomize_vertical_lines: bool = True,
                             vertical_line_prob: float = 0.5
                             ) -> str:
        
        """
        Convert a tsv string representation of a table to a LaTeX table.

        Args:
            column_style: LaTeX style for the columns.
            text: tsv string representation of a table.
            randomize_colors: If True, randomize the colors of the cells.
            randomize_vertical_lines: If True, randomize vertical lines between 
                columns.
            vertical_line_prob: Probability of adding a vertical line.

        Returns:
            LaTeX string representation of the table.

        Raises:
            TypeError: If `column_style` or `text` are not strings.
            TypeError: If `randomize_colors` or  
                `randomize_vertical_lines` are not booleans.
            TypeError: If `vertical_line_prob` is not a float.
        """

        assert isinstance(column_style, str), \
            TypeError(f"Expected str, got {type(column_style)}.")
        assert isinstance(text, str), \
            TypeError(f"Expected str, got {type(text)}.")
        assert isinstance(randomize_colors, bool), \
            TypeError(f"Expected bool, got {type(randomize_colors)}.")
        assert isinstance(randomize_vertical_lines, bool), \
            TypeError(f"Expected bool, got {type(randomize_vertical_lines)}.")
        assert isinstance(vertical_line_prob, float), \
            TypeError(f"Expected float, got {type(vertical_line_prob)}.")

        # Replace special characters with their LaTeX equivalents
        text = self.replace_special_to_latex(text)

        # Do color randomization of rows if True, else
        # return the text as is.
        if randomize_colors:
            logging.info("Randomizing row colors.")
            latex_text: str = self._randomize_colors(text)
        else:
            latex_text = self._return_latex_text(text)

        # Set up LaTeX formatting.
        header1: str = "\\usepackage{colortbl}\n\\usepackage{xcolor}\n" 
        colors: List[str] = self._return_colors() #Simple return method to declutter
        header2: str = "\\begin{document}\n\\begin{table}\n"

        header: str = header1 + np.random.choice(colors) + header2

        # If randomize_vertical_lines, add random pipes between column
        # definitions.
        if randomize_vertical_lines:
            logging.info("Randomizing column lines.")
            column_style = self._randomly_add_vert_lines(column_style, vertical_line_prob)

        footer: str = "\\hline\n\\end{tabular}\n\\end{table}\n\\end{document}"

        # Create final LaTeX table.
        formatted_latex_text: str = header + column_style + latex_text + footer
        
        return formatted_latex_text
    
    def convert_latex_to_tsv(self, latex_table: str) -> str:

        """
        Returns reformatted table from LaTeX format to simple tsv.  

        Args:
            latex_table: The LaTeX formatted table.

        Returns:
            The reformatted table in simple tsv format as to be loaded
                as structured data.
        
        Raises:
            TypeError: If latex_table is not a str.
        """

        assert isinstance(latex_table, str), \
            TypeError(f"Expected str, got {type(latex_table)}.")

        formatted_table: str = latex_table.replace("\\", "")
        formatted_table = formatted_table.replace("textbf", "")
        formatted_table = formatted_table.replace("{", "")
        formatted_table = formatted_table.replace("}","")
        formatted_table = formatted_table.replace("hline", "")
        formatted_table = formatted_table.replace("*", "")

        return formatted_table

    def _read_jsonl(self, file_path: str) -> List[Any]:

        """
        Reads a .jsonl file and returns a list of dictionaries.

        Args:
            file_path: The path to the .jsonl file.

        Returns:
            A list of dictionaries read from the .jsonl file.

        Raises:
            TypeError: If file_path is not a string.
            ValueError: If file_path does not have a .jsonl extension.
            FileNotFoundError: If the file does not exist.
        """
        
        assert isinstance(file_path, str), \
            TypeError(f"Expected str, got {type(file_path)}.")
        assert file_path.endswith(".jsonl"), \
            ValueError(f"Expected .jsonl file, got {file_path}.")
        
        data: list = []
        try:
            with open(file_path) as reader:
                for obj in reader:
                    data.append(eval(obj))
        except FileNotFoundError:
            logging.error(f"File {file_path} not found.")
            raise FileNotFoundError(f"File {file_path} not found.")

        return data
    
    # TODO: Provide more control on font types.
    def _write_tex_file(self, tex_filepath: str, 
                        table_latex: str, 
                        prefix: str,
                        font_types: List[str] = [
                            "lmodern",
                            "times",
                            "helvet", 
                            "courier",
                            "mathpazo",
                            "newcent"
                        ],
                        ) -> None:
        
        """
        Writes a tex file as part of generating tables to images pipeline.

        Args:
            tex_filepath: The path to the tex file.
            table_latex: The LaTeX code for the table.
            prefix: The prefix for the tex file.
            font_types: The font types to use.

        Raises:
            TypeError: If `tex_filepath`, `table_latex`, 
                 or `prefix` are not strings.
            TypeError: If `font_types` is not a list.
            ValueError: If `font_types` contains an unsupported font type.
        """
        
        assert isinstance(tex_filepath, str), \
            TypeError(f"Expected str, got {type(tex_filepath)}.")
        assert isinstance(table_latex, str), \
            TypeError(f"Expected str, got {type(table_latex)}.")
        assert isinstance(prefix, str), \
            TypeError(f"Expected str, got {type(prefix)}.")
        assert isinstance(font_types, list), \
            TypeError(f"Expected list, got {type(font_types)}.")

        assert all(font_type for font_type in ["lmodern",
                                                "times",
                                                "helvet", 
                                                "courier",
                                                "mathpazo",
                                                "newcent"]), \
            ValueError(f"Invalid font type in: {font_types}.")
        
        # Set formatting.
        margin: str = str(round(random.uniform(0.2, 0.8), 2))
        font_type: str = random.choice(font_types)
        width: str = random.choice(["0.3", "0.4", "0.5", "0.6"])

        # Replace placeholders in prefix.
        prefix = prefix.replace("FONT_TYPE", font_type)
        prefix = prefix.replace("{MARGIN}", margin)
        prefix = prefix.replace("{RULE_WIDTH}", width)
    
        latex_code: str = prefix + table_latex
        # Write to tex file.
        logging.info(f"Writing LaTeX table to {tex_filepath}.")
        with open(tex_filepath, 'w') as f:
            f.write(latex_code)

    def _crop_synth_table(self, image_path: str) -> None:

        """
        Crops the synthetic table image based on its border.

        Args:
            image_path: Path to the synthetic table image.
        
        Raises:
            TypeError: If image_path is not a string.
        """

        assert isinstance(image_path, str), \
            TypeError(f"Expected str, got {type(image_path)}")

        # Load the image natrually and in grayscale
        image = cv2.imread(image_path)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply a binary threshold to the grayscale image
        _, binary = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest rectangular contour assuming it's the table
        table_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(table_contour)

        # Crop the table from the image
        height, width = gray_image.shape
        cropped_table = image[max(0, y-random.randint(10, 100)):min(y+h+random.randint(10, 100), height), 
                            max(0, x-random.randint(10, 100)):min(x+w+random.randint(10, 100), width)]
        
        keeptrying = True
        counter = 0
        while keeptrying:
            try:
                cv2.imwrite(image_path, cropped_table)
                keeptrying = False
            except Exception as e:
                print(e)
                time.sleep(0.5)
                counter += 1
                if counter >= 10:
                    break

        logging.info(f"Writing image to: {image_path}.")
        cv2.imwrite(image_path, cropped_table)
    
    # TODO: Provide control
    def _image_augmentations(self, 
                             augmentations: List[str] = [
                                 "RandomBrightnessContrast",
                                 "Blur",
                                 "Downscale",
                                 "GaussNoise",
                                 "ISONoise",
                                 "ImageCompression",
                                 "RandomGamma",
                                 "ToGray"
                             ]
                             ) -> Compose:
        
        """
        Args:
            augmentations: List of augmentations to apply.

        Returns:
            Composition of all the augmentations.

        Raises:
            TypeError: If augmentations is not a list.
            ValueError: If augmentations contains an unsupported augmentation.
        """

        assert isinstance(augmentations, list), \
            TypeError(f"Expected list, got {type(augmentations)}.")
        
        # TODO: allow for control over params.
        transformations_map = {
            "RandomBrightnessContrast": A.RandomBrightnessContrast(p=0.5),
            "Blur": A.Blur(p=0.5, blur_limit=(1,3)),
            "Downscale": A.Downscale(p=0.5, scale_min=0.75, scale_max=0.85),
            "GaussNoise": A.GaussNoise(p=0.5, var_limit=(50.0, 250.0)),
            "ISONoise": A.ISONoise(p=0.2, color_shift=(0.1, 0.5), intensity=(0.1, 0.5)),
            "ImageCompression": A.ImageCompression(p=0.5, quality_lower=65, quality_upper=90),
            "RandomGamma": A.RandomGamma(p=0.5, gamma_limit=(50, 150)),
            "ToGray": A.ToGray(p=0.5), 
        }

        if not all([item in transformations_map.keys() for item in augmentations]):
            raise ValueError(f"transformation must be in {transformations_map.keys()}")
        
        # Initialize the pipeline with the selected augmentations.
        pipeline = []

        # Add the selected augmentations to the pipeline.
        for name in augmentations:
            transformation: Any = transformations_map.get(name)
            if transformation:
                pipeline.append(transformation)
        
        # Compose the augmentations into a single transform.
        transform: Compose = A.Compose(pipeline)

        return transform
    
    def generate_images(self, 
                         folder_name: str, 
                         image_name: str,
                         data: str,
                         randomize_dpi: bool = True,
                         use_augmentations: bool = True) -> None:
        
        """
        Generates synthetic tables of images from passed templates, 
            optionally applies dpi randomization and image augmentations,
            and writes the image.

        Args:
            folder_name: The folder name to write images to.  Should end in "/images".
            image_name: Unique image name.  Will be saved with the .jpg extension.
            data: The LaTeX formatted table to convert into an image.
            randomize_dpi: Bool to enable randomized dpi.
            use_augmentations: Bool to enable image augmentations.

        Returns:
            None.
        
        Raises:
            TypeError: If folder_name or data are not str.
            TypeError: If randomize_dpi or use_augmentations are not bool.
            ValueError: if folder_name does not end with "/images".
        """
        
        assert isinstance(folder_name, str), \
            TypeError(f"Expected str, got {type(folder_name)}.")
        assert isinstance(data, str), \
            TypeError(f"Expected str, got {type(data)}.")
        assert isinstance(randomize_dpi, bool), \
            TypeError(f"Expected bool, got {type(randomize_dpi)}.")
        assert isinstance(use_augmentations, bool), \
            TypeError(f"Expected bool, got {type(use_augmentations)}.")
        
        if not folder_name.endswith("/images"):
            raise ValueError(f"Expected a folder ending with /images, got {folder_name}")
        
        os.makedirs(folder_name, exist_ok=True)

        # Write intermediate tex file.
        self._write_tex_file('table.tex', data, DOCUMENT_PREFIX_LANDSCAPE)

        # Create new filepaths for file types.
        tex_filepath: Path = Path(f"/{folder_name}/{image_name}.tex")
        pdf_filepath: Path = tex_filepath.with_suffix(".pdf")
        img_filepath: Path = tex_filepath.with_suffix(".jpg")

        # Compile the LaTeX file to PDF
        subprocess.run(['pdflatex', '-interaction=nonstopmode', 'table.tex'])
        while not os.path.exists("./table.pdf"):
            time.sleep(0.5)
        #Move files to respective paths.
        subprocess.run(['mv', 'table.tex', tex_filepath])
        subprocess.run(['mv', 'table.pdf', pdf_filepath])

        # Randomize DPI if True.
        if randomize_dpi:
            images: List[Image.Image] = convert_from_path(pdf_filepath, dpi=np.random.choice([90, 120, 250]))
        else:
            images = convert_from_path(pdf_filepath, dpi=250)
        image = images[-1] # Sometimes more than one page is generated, so we take the last one

        # Do image augmentations if True.
        if use_augmentations:
            transform: Any = self._image_augmentations()
            image = transform(image=np.array(image))['image']
            # image = image.astype(np.uint8)
            image = Image.fromarray(image)

        logging.info(f"Saving image to: {img_filepath}")
        image.save(img_filepath, 'JPEG')
        
        # Clean up intermediate files.
        self._crop_synth_table(str(img_filepath))
        subprocess.run(['rm', tex_filepath])
        subprocess.run(['rm', pdf_filepath])

class QAItem(BaseModel):
    "Model representing query and answer pairs for a json."

    query: str = Field(..., description="query generated from the table.")
    answer: str = Field(..., description="answer to the query generated from the table.")

class QAList(BaseModel):
    "Model representing the list of query answer pairs."

    qa_list: List[QAItem] = Field(..., description="List of qa pairs extracted from the table.")

class TableAugmentor:

    def __init__(self, config_path: str) -> None:

        assert isinstance(config_path, str), TypeError(f"Expected str, got {type(config_path)}.")
        
        self.llm_info, self.prompt_info = self.load_configs(config_path)
        self.init_llm()
        self.init_table_modifying_chain()
        self.init_table_qa_chain()
        self.init_table_ocr_chain()

    def load_configs(self, config_path: str) -> Any:
        """
        Loads a yaml config file and returns llm info.

        Args:
            config_path: Path to the config yaml file.

        Returns:
            A tuple of dictionaries containing the llm information.
        """

        assert isinstance(config_path, str), \
            TypeError(f"Must be type str, but got {type(config_path)}.")

        try:
            with open(config_path, "r") as yaml_file:
                config: dict = yaml.safe_load(yaml_file)
        except FileNotFoundError:
            logging.error(f"{config_path} not found.")
            raise FileNotFoundError(f"{config_path} does not exist.")

        llm_info, prompt_info = config["llm"], config["prompts"]

        return llm_info, prompt_info

    def init_table_modifying_chain(self) -> None:
        """
        Initializes the table modifying chain by loading the entity prompt and combining
        it with the language model and a Str output parser.

        Args:
            None

        Returns:
            None
        """

        modifying_prompt: Any = load_prompt(os.path.join(repo_dir, self.prompt_info["table_modification"]))
        self.modifying_chain = modifying_prompt | self.llm | StrOutputParser()
    
    def init_table_ocr_chain(self) -> None:
        """
        Initializes the table modifying chain by loading the ocr prompt and combining
        it with the language model and a Str output parser.

        Args:
            None

        Returns:
            None
        """

        table_ocr_prompt: Any = load_prompt(os.path.join(repo_dir, self.prompt_info["ocr_query_answer"]))
        self.table_ocr_chain = table_ocr_prompt | self.llm | StrOutputParser()

        # TODO: Add config file
    def init_table_qa_chain(self) -> None:
        """
        Initializes the table modifying chain by loading the entity prompt and combining
        it with the language model and a Str output parser.

        Args:
            None

        Returns:
            None
        """

        # Define the parser.
        parser = PydanticOutputParser(pydantic_object=QAList) # type: ignore

        table_qa_prompt = PromptTemplate(
            template=tableqa_template,
            input_variables=['table'],
            partial_variables={'format_instructions': parser.get_format_instructions()},
        )

        # Initialize the chain.
        self.table_qa_chain = table_qa_prompt | self.llm | JsonOutputParser()

    def init_llm(self) -> None:
        """
        Initializes the llm to be used for various synethic table generation tasks.  
        Use of Llama 3.1 70B and 405B highly recommeneded.

        Args:
            None.

        Returns:
            None.
        """

        llm = APIGateway.load_llm(
            type=self.llm_info['api'],
            streaming=self.llm_info['streaming'],
            coe=self.llm_info['coe'],
            do_sample=self.llm_info['do_sample'],
            max_tokens_to_generate=self.llm_info['max_tokens_to_generate'],
            temperature=self.llm_info['temperature'],
            select_expert=self.llm_info['select_expert'],
            process_prompt=self.llm_info['process_prompt'],
            )

        self.llm = llm

    def _write_appended_json(self,
                             json_path: str,
                             data: dict) -> None:
        """
        Utility method to apped json payloads to a list in
        a json file as expected by LlaVa 1.5.

        Args:
            json_path: The path to the desired json location 
                (including the .json extension).
            data: The json payload to append.

        Returns:
            None.
        
        Raises:
            TypeError: If json_path is not a str.
            TypeError: If data is not a dict.
        """

        assert isinstance(json_path, str), \
            TypeError(f"Expected str, got {type(json_path)}.")
        assert isinstance(data, dict), \
            TypeError(f"Expected dict, got {type(data)}.")

        with open(json_path, "r+") as f:
                payload = json.load(f)
                payload.append(data)
                f.seek(0)
                json.dump(payload, f, indent=4)
                f.truncate()
    
    # TODO: Generalize and add config
    def create_training_data(self, 
                             num_samples: int,
                             data_directory: str,
                             split: str = "train",
                             synth_tables: List[Dict[str, str]] = synth_tables,
                             ) -> None:
        
        """
        Creates training data based on synthetic tables that come from a dictionary 
        of column types and templates.  

        Args:
            num_samples: The number of synthetic examples to create.
            data_directory: The output directory for all synthetic data.
            split: The split to create examples for.  Must be either train or val.
            synth_tables: A dictionary of column headers for LaTeX table formatting 
                and the modified tsv format from the templates.

        Returns:
            None.

        Raises:
            TypeError: If num_samples is not int.
            TypeError: If data_directory or split are not str.
            TypeError: If synth_tables is not a dict.
            ValueError: If split is not train or val.
        """

        assert isinstance(num_samples, int), \
            TypeError(f"Expected int, got {type(num_samples)}.")
        assert isinstance(data_directory, str), \
            TypeError(f"Expected str, got {type(data_directory)}.")
        assert isinstance(split, str), \
            TypeError(f"Expected str, got {type(data_directory)}.")
        assert isinstance(synth_tables, list), TypeError(f"Expected list, got {type(synth_tables)}.")

        if not (value for value in ["train", "val"]): 
            raise ValueError(f"Must be train or val, got {split}.")


        self.num_samples: int = num_samples
        self.data_directory = data_directory
        self.synthetic_folder: str = os.path.join(data_directory, "images")

        # Set the json path to be standard for LlaVa 1.5 format.
        json_path: str = os.path.join(self.data_directory, f"annotations_{split}.json")

        # Create directory if it does not exist and then
        # create a json file for a list of payloads.
        if not os.path.exists(self.synthetic_folder):
            os.makedirs(os.path.join(self.synthetic_folder))
        with open(json_path, "w") as f:
            json.dump([], f)
        
        # Instantiate table tools object for later use.
        table_tools = TableTools()

        # Generate synthetic data for specified number of
        # samples.
        for i in range(self.num_samples):
            # Get a random table.
            table = random.choice(synth_tables)
            # Provide a unique identifier for the sample.
            unique_uuid: str = str(uuid.uuid4())
            # Get the column definition for the table.
            columns: str = table['columns']
            # Modify the table entities and numbers with LLM.
            try:
                new_table: str = self.modifying_chain.invoke(table["tsv_formatted"])
            except Exception as e:
                logging.error(e)
            
            # Create a natural user prompt asking for table OCR using LLM.
            try:
                table_ocr_prompt: str = self.table_ocr_chain.invoke(new_table)
            except Exception as e:
                logging.error(e)
            # Convert the table to LaTeX format from modified tsv.
            synth_table: str = table_tools.convert_tsv_to_latex(columns, new_table)

            # Generate the image of the table and save to appropriate
            # folder.
            table_tools.generate_images(folder_name=self.synthetic_folder, 
                            image_name=unique_uuid,
                            data=synth_table)
            
            # Generate the payload.
            new_data: dict = {
                "id": unique_uuid,
                "image": f"{unique_uuid}.jpg",
                "conversations": [
                    {"from": "human", "value": table_ocr_prompt},
                    {"from": "gpt", "value": table_tools.convert_latex_to_tsv(new_table)}
                ]
            }

            # Append new data to json file.
            self._write_appended_json(json_path=json_path,
                                      data=new_data)

            qa_list: List[Dict[str, str]] = self.table_qa_chain.invoke(synth_table)["qa_list"]

            # Gather qa table questions from the output list.
            for qa in qa_list:
                # Gather payload for item in qa list.
                new_data = {
                "id": unique_uuid,
                "image": f"{unique_uuid}.png",
                "conversations": [
                    {"from": "human", "value": qa["query"]},
                    {"from": "gpt", "value": qa["answer"]}
                    ]
                }

                # Append new data to json file.
                self._write_appended_json(json_path=json_path,
                                      data=new_data)



            




    




