import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

import cv2
import json
from langchain.schema import Document
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
from paddleocr import PaddleOCR,  PPStructure

ocr = PaddleOCR(use_angle_cls=False, lang='en') # need to run only once to download and load model into memory
layout_engine = PPStructure(recovery=False, layout=True, table=True, ocr=False, show_log=False) # need to run only once to download and load model into memory



class PaddleOCRLoader():
    """
    This class loads a PDF document and extracts its content using PaddleOCR and PaddleStructure.
    """
    def __init__(self, document_path, output_folder=None, save_intermediate=True, header_height = 0, footer_height = 0, font_path=None):
        """
        Initialize the PaddleOCRLoader class.
        Args:
            document_path (str): Path to the PDF document to load.
            output_folder (str, optional): Output folder for saving intermediate files. Defaults to 'data/extraction'.
            save_intermediate (bool, optional): Whether to save intermediate files for debugging purpouses . Defaults to True.
            header_height (int, optional): Height of the header in pixels. Defaults to 0.
            footer_height (int, optional): Height of the footer in pixels. Defaults to 0.
            font_path (str, optional): Path to the font file. Defaults to '../data/fonts/simfang.ttf'.
        """
        if output_folder is None:
            self.output_folder = os.path.join(kit_dir,"data/extraction")
        else: 
            self.output_folder = output_folder
        if font_path is None:
            self.font_path = os.path.join(kit_dir,"data/fonts/simfang.ttf")
        else:
            self.font_path = font_path 
        self.document_path = document_path
        self.save_intermediate = save_intermediate
        self.header_height = header_height
        self.footer_height = footer_height

        
    def load(self):
        """get langchain documens from PDF file
        Returns:
            list: langchain docs
        """
        self.documents = []
        print(self.save_intermediate)
        texts=self.load_pdf(self.document_path, 
                       output_folder=self.output_folder, 
                       save_intermediate=self.save_intermediate, 
                       header_height = self.header_height, 
                       footer_height = self.footer_height
                       )
        for page, content in enumerate(texts):
            metadata={"source": self.document_path, "page":page} 
            doc=Document(page_content=content, metadata=metadata)
            self.documents.append(doc)
        return self.documents
    
    
    #PDF Conversion
    
    def convert_pdf_to_images(self, pdf_file_path, output_folder='data/extraction'):
        """
            this method converts a pdf file to a series of images of each page
        Args:
            pdf_file_path (str): pdf file path
            output_folder (str, optional): output directory. Defaults to 'data/extraction'.

        Returns:
            str: _description_ output directory
        """
        images = convert_from_path(pdf_file_path)
        output_folder=os.path.join(output_folder,os.path.basename(pdf_file_path).split('.')[0])
        os.makedirs(output_folder, exist_ok=True)
        for i, image in enumerate(images):
            img_name=f"{os.path.basename(pdf_file_path).split('.')[0]}_page_{i}.jpg"
            output_path = f"{output_folder}/{img_name}"
            image.save(output_path, 'JPEG')
        return output_folder
    
    
    # OCR and tables-layout engine
      
    def simple_ocr(self, img_file_path, ocr=ocr):
        """
        This method performs simple OCR on a single image
        Args:
            img_file_path (str): image file path
            ocr (PaddleOCR, optional): PaddleOCR engine object. Defaults to ocr.
        Returns:
            str: output ocr object
        """
        result=ocr.ocr(img_file_path, cls=False)
        return result

    def structured_ocr(self, img_file_path, ocr=layout_engine):
        """
        This method performs an structure deetction, table transcription and OCR on a single image
        Args:
            img_file_path (str): image file path
            ocr (PaddleOCR, optional): PaddleStructure engine object. Defaults to layout_engine.
        Returns:
            str: output ocr object
        """
        img = cv2.imread(img_file_path)
        return ocr(img)
    
        
    # structure extraction
    
    def show_paddle_structure_bboxs(self, image_path, result, save=False):
        """
        This method shows the bounding boxes of a structured_ocr execution over a provided image_file
        Args:
            image_path (str): image file path
            result (str): output structured ocr object
            save (bool, optional): save image in original directory. Defaults to False.
        Rturns: cv2 image obbjectt
        """
        image = cv2.imread(image_path)
        type_color = {
            'header': (255,220,0), #yellow
            'table': (255,60,155), #purple
            'table caption': (255,200,200), #light pink
            'figure': (0,255,30), # light green
            'figure_caption': (0,20,180), # dark blue
            'title': (0,180,255), #light blue
            'text': (200,10,255), #brigth pink
            'reference':(150,150,150), #gray
            'footer': (255,50,0) #orange
        }
        bboxs=[]
        for element in result:
            bbox = (element['bbox'],element['type']) 
            bboxs.append(bbox)
        for bbox in bboxs:
            x1, y1, x2, y2 = bbox[0]
            color=type_color.get(bbox[1], (255,0,0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, bbox[1], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if save:
            path=os.path.join(os.path.dirname(image_path), "".join(os.path.basename(image_path).split('.')[:-1]))
            bbox_image_path=f"{path}_bboxs.jpg"
            cv2.imwrite(bbox_image_path,image)
        return image

    def show_simple_bboxes(self, image_path, bboxs, save=False, tag=None):
        """
        This method shows a list of bounding boxes over a provided image file path
        Args:
            image_path (str): image file path
            bboxs (list): list of bounding boxes
            save (bool, optional): save image in original directory. Defaults to False.
            tag (str, optional): tag for the out image. Defaults to None.
        """
        image = cv2.imread(image_path)
        color =  (255,60,155) #purple
        for i, bbox in enumerate(bboxs):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if save: 
            path=os.path.join(os.path.dirname(image_path), "".join(os.path.basename(image_path).split('.')[:-1]))
            bbox_image_path=f"{path}_bboxs_{tag}.jpg"
            cv2.imwrite(bbox_image_path, image)
        return image
        
    def get_tables(self, result):
        """
        This method gets the tables from a structured_ocr execution
        Args:
            result (str): output structured ocr object
        Returns:
            list: list of bounding boxes
            list: list of html tables
        """
        bboxs=[]
        htmls=[]
        for element in result:
            if element['type']=='table':
                bboxs.append(element['bbox'])
                htmls.append(element['res']['html'])
        return bboxs, htmls 

    def save_tables(self, tables, path ,tag="table", save_html=False):
        """
        This method saves the page tables in a json file
        Args:
            tables (list): list of html tables
            path (str): image_path 
            tag (str, optional): tag to use in jsonfile. Defaults to 'table'
            save_html (bool, optional): save html tables outside as files. Defaults to False.
        Returns:
            (str): json output path
        """
        path=os.path.join(os.path.dirname(path), "".join(os.path.basename(path).split('.')[:-1]))
        tables_dict = {}
        for i, table in enumerate(tables):
            tables_dict[f'{tag} {i}']=table
            if save_html:
                with open(f'{path}_table_{i}.html', 'w') as table_file:
                    table_file.write(table.replace('<table>','<table border="1">'))
        out_path=f'{path}_tables.json'
        with open(out_path, 'w') as json_file:
            json.dump(tables_dict, json_file)
        return out_path

    def get_figures(self, result):
        """
        This method gets the figures from a structured_ocr execution
        Args:
            result (str): output structured ocr object
        Returns:
            list: list of bounding boxes
            list: list of figures (image array)
        """
        bboxs=[]
        figures=[]
        for element in result:
            if element['type']=='figure':
                bboxs.append(element['bbox'])
                figures.append(element['img'])
        return bboxs, figures 

    def save_figures(self, figures, path ,tag="figure"):
        """
        This method saves the page figures in a folder and reference them in a json file
        Args:
            tables (list): list of figures
            path (str): image_path 
            tag (str, optional): tag to use in jsonfile. Defaults to "figure".
        Returns:
            (str): json output path
        """
        path=os.path.join(os.path.dirname(path), "".join(os.path.basename(path).split('.')[:-1]))
        figures_dict = {}
        for i, figure in enumerate(figures):
            figure_path=f"{path}_figure_{i}.jpg"
            image = Image.fromarray(figure)
            image.save(figure_path, 'JPEG')
            figures_dict[f'{tag} {i}']=figure_path
        out_path=f'{path}_figures.json'
        with open(out_path, 'w') as json_file:
            json.dump(figures_dict, json_file)
        return out_path 

    def get_equations(self, result):
        """
        This method gets the equations from a structured_ocr execution
        Args:
            result (str): output structured ocr object
        Returns:
            list: list of bounding boxes
            list: list of equations (image array)
        """
        bboxs=[]
        equations=[]
        for element in result:
            if element['type']=='equation':
                bboxs.append(element['bbox'])
                equations.append(element['img'])
        return bboxs, equations 

    def save_equations(self, equations, path ,tag="equation"):
        """
        This method saves the page equations in a folder and reference them in a json file
        Args:
            tables (list): list of equations
            path (str): image_path 
            tag (str, optional): tag to use in jsonfile. Defaults to "equation".
        Returns:
            (str): json output path
        """
        path=os.path.join(os.path.dirname(path), "".join(os.path.basename(path).split('.')[:-1]))
        equations_dict = {}
        for i, equation in enumerate(equations):
            equation_path=f"{path}_equation_{i}.jpg"
            image = Image.fromarray(equation)
            image.save(equation_path, 'JPEG')
            equations_dict[f'{tag} {i}']=equation_path
        out_path=f'{path}_equations.json'
        with open(out_path, 'w') as json_file:
            json.dump(equations_dict, json_file)
        return out_path
    
    def mask_elements_from_image(self, image, bboxs, tag=None, save=False, image_path=None):
        """
        This method creates an image with the original image but whit bboxes masked with a tag numerated text 
        Args:
            image (np.array): image array
            bboxs (list): list of bounding boxes
            tag (str, optional): tag to use in the masked bboxes. Defaults to None.
            save (bool, optional): save new masked image in original directory. Defaults to False.
            image_path (str, optional): original image path needed if save enabled. Defaults to None.
        Returns:
            path: path of saved image
            np.array: masked image
        """
        # Create a mask image with the same size as the original image
        mask = Image.new("L", image.size, 255)  # "L" mode is for grayscale
        # Create a drawing object for the mask
        draw = ImageDraw.Draw(mask)
        # Draw black rectangles on the mask at the positions of the bounding boxes
        for box in bboxs:
            draw.rectangle(box, fill=0)
        # Apply the mask to erase content within the bounding boxes
        erased_image = Image.composite(image, Image.new('RGB', image.size, (255, 255, 255)), mask)
        # Create a drawing object for the erased image
        draw_erased = ImageDraw.Draw(erased_image)
        font = ImageFont.truetype(self.font_path, 30)
        # Draw the text in the middle of each bounding box
        for i,box in enumerate(bboxs):
            text=f'**{tag} {i}**'
            # Calculate the center of the bounding box
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            # Calculate the position to place the text
            text_size = draw_erased.textsize(text, font)
            text_position = (center_x - text_size[0] // 2, center_y - text_size[1] // 2)
            # Draw the text on the erased image
            draw_erased.text(text_position, text, fill=(0, 0, 0), font=font)
        if save:
            path=os.path.join(os.path.dirname(image_path), "".join(os.path.basename(image_path).split('.')[:-1]))
            mask_image_path=f"{path}_mask.jpg"
            erased_image.save(mask_image_path)
        else:
            mask_image_path=None
        return mask_image_path, erased_image
    
    # structure cleaning
    
    def calculate_intersection_percentage(self, bbox1, bbox2):
        """ this function calculates the intersection between two bboxes
        Args:
            bbox1 (list): firs bounding box
            bbox2 (list): second bounding box
        Returns:
            float: percentage of intersection between bbox1 and bbox2
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        # Calculate intersection area 
        intersection_x_min = max(x1_min, x2_min)
        intersection_y_min = max(y1_min, y2_min)
        intersection_x_max = min(x1_max, x2_max)
        intersection_y_max = min(y1_max, y2_max)
        if intersection_x_max < intersection_x_min or intersection_y_max < intersection_y_min:
            # No intersection
            return 0.0
        intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
        # Calculate total area of bounding boxes
        total_area_bbox1 = (x1_max - x1_min) * (y1_max - y1_min)
        total_area_bbox2 = (x2_max - x2_min) * (y2_max - y2_min)
        # Calculate intersection percentage
        intersection_percentage = (intersection_area / min(total_area_bbox1, total_area_bbox2)) * 100.0
        return intersection_percentage

    def bb_intersect(self, bbox1, bbox2):
        """ 
        this function calculates if there is intersection between two bboxes if so calculates intersection percentage
        Args:
            bbox1 (list): first bounding box
            bbox2 (list): second bounding box
        Returns:
            bool: intersection
            float: percentage of intersection between bbox1 and bbox2
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        # Check for intersection along x-axis
        if x1_max < x2_min or x1_min > x2_max:
            return False, 0.0
        # Check for intersection along y-axis
        if y1_max < y2_min or y1_min > y2_max:
            return False, 0.0
        # Bounding boxes overlap
        intersection_percentage = self.calculate_intersection_percentage(bbox1, bbox2)
        return True, intersection_percentage

    def merge_bboxes(self, bbox1, bbox2):
        """ 
        This function generates a new bbox which contains both bbox1 and bbox2
        Arggs: 
            bbox1 (list): first bounding box
            bbox2 (list): second bounding box
        Returns:
            list: merged bounding box
        """    
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        # Calculate the minimum and maximum coordinates to create a new bounding box
        merged_bbox = [ min(x1_min, x2_min),
                        min(y1_min, y2_min),
                        max(x1_max, x2_max),
                        max(y1_max, y2_max)
                        ]
        return merged_bbox

    def get_content_bboxes(self, bboxes, max_persentage_overlap_allowed = 10):
        """ this function returns a clean list of bounding boxes merging bboxes that are overlaped
        Args:
            bboxes (list): list of bounding boxes
            max_persentage_overlap_allowed (int, optional): percentage of overlap allowed. Defaults to 10.
        Returns:
            list: list of bounding boxes
        """
        bbox_indexes_to_drop=True
        while bbox_indexes_to_drop:
            bbox_indexes_to_drop=False
            new_bboxes=[]
            bbox_indexes_to_drop=[]
            for i in range(len(bboxes)): #iterate over all bboxes 
                intersected=False #initialize flag for markig if the analized bbox get an intersection with any other bbox
                for j in range(1,len(bboxes[i:])): #iterate over the following bboxes
                    intersection, persentage = self.bb_intersect(bboxes[i],bboxes[i+j]) #find intersects
                    if persentage > max_persentage_overlap_allowed: #if intersects in more than n% of the total area
                        intersected=True #flag the the i bbox has an intersection
                        new_bboxes.append(self.merge_bboxes(bboxes[i],bboxes[i+j])) #add the merged bbox to the new bbox list (this will replace the i bbox with the merged one)
                        bbox_indexes_to_drop.append(i+j) #add the intersected bbox tho the bboxes to drop
                        if i in bbox_indexes_to_drop: #remove the i ocurrence form indexes to remove in case now is stored another merged bbox
                            bbox_indexes_to_drop.remove(i)
                        break #stop iterating over following elemnts in j if there are intersection
                if not intersected: #if bbox i was not intersected with another bbox
                    new_bboxes.append(bboxes[i]) #add the i bbox to the new_bboxes
            bbox_indexes_to_drop=list(set(bbox_indexes_to_drop))
            new_bboxes= [element for index, element in enumerate(new_bboxes) if index not in bbox_indexes_to_drop] #remove all bboxes droped  
            bboxes=new_bboxes
        return bboxes

    def expand_bounding_boxes(self, bboxes, img_size, n):
        """ this function takes in a list of bounding boxes and expand it n pixels in each drection
        Args:
            bboxes (list): listo of boundig boxes
            img_size (tuple): width and height of the origina image
            n (int): number of pixels to expand
        Returns:
            list: bounding boxes
        """
        w, h = img_size 
        expanded_bboxes = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            # Expand bounding box by 'n' pixels
            xmin = max(0, xmin - n)
            ymin = max(0, ymin - n)
            xmax = min(w, xmax + n)
            ymax = min(h, ymax + n)
            expanded_bboxes.append([xmin, ymin, xmax, ymax])
        return expanded_bboxes
    
    
    # Multicolumn ordering
    
    def sort_by_y(self, bounding_boxes):
        """
        Sort bounding boxes based on the y-coordinate of the top-left point
        Args:
            bounding_boxes (list): list of bounding boxes
        Returns:
            list: sorted bounding boxes
        """
        return sorted(bounding_boxes, key=lambda box: box[1])
    
    def detect_columns(self, bounding_boxes):
        """
        This function detect if  there is 1 or 2 columns 
        then sort each bounding boxes to the corresponding colum 
        based on the x-coordinate of the top-left point
        Args:
            bounding_boxes (list): list of bounding boxes
        Retuns:
            (list): list of first column bounding boxes
            (list): list of second column bounding boxes
        """
        sorted_boxes = sorted(bounding_boxes, key=lambda box: box[0])
        # Calculate the mid x-coordinate of the x center of bounding boxes to find potential columns
        mid_x = sum((box[0]+(box[2]-box[0])/2) for box in sorted_boxes) / len(sorted_boxes)
        # Check if there are boxes which right x coordinate starts after the mid center-x coordinate
        num_columns = 1 if all(box[0] < mid_x for box in sorted_boxes) else 2
        if num_columns == 1:
            return 1, [[bounding_boxes],[]]
        left_column = []
        right_column = []
        for box in sorted_boxes:
            column = left_column if box[0] < mid_x else right_column
            column.append(box)
        return 2, [left_column, right_column]

    def order_one_column(self, column_boxes):
        """
        Order bboxes within a single column based on y-coordinate
        Args:
            column_boxes (list): list of culumns of bounding boxes
        Returns:
            list: ordered bounding boxes
        """
        return [box for box in sorted(column_boxes[0], key=lambda box: box[1])]
    def order_two_columns(self, column_boxes):
        """
        Order bboxes within a single column based on y-coordinate
        Args:
            column_boxes (list): list of culumns of bounding boxes
        Returns:
            list: ordered list of columns of bounding boxes  
        """
        left_column = sorted(column_boxes[0], key=lambda box: box[1])
        right_column = sorted(column_boxes[1], key=lambda box: box[1])
        return left_column, right_column

    def order_horizontal_colums(self, column_boxes):
        """
        Order bboxes within a single column based on x-coordinate
        Args:
            column_boxes (list): list of culumns of bounding boxes
        Returns:
            list: ordered bounding boxes
        """
        return [box for box in sorted(column_boxes[0], key=lambda box: box[0])]

    def order_paragraphs(self, bounding_boxes, img_size, header_height=0, footer_height=0):
        """
        This fuction order paragraphs bounding boxes in a logical readable order taking in acount the number of columns detected, and the header and footer height 
        Args:
            bounding_boxes (list): List of bounding boxes
            img_size (touple): width and height of the origina image
            header_height (int, optional): header height Defaults to 0.
            footer_height (int, optional): footer height Defaults to 0.
        Returns:
            list: ordered bounding boxes
        """
        # Step 1: Sort bounding boxes based on y-coordinate
        sorted_boxes = self.sort_by_y(bounding_boxes)
        # Step 2: Detect columns
        num_columns, column_boxes = self.detect_columns(sorted_boxes)
        # Step 3: Identify header and footer sections
        header_boxes = [box for box in sorted_boxes if box[3] < header_height], []
        footer_boxes = [box for box in sorted_boxes if box[1] > (img_size[1] - footer_height)], []
        # step 4 Remove header and footer boxes from the main list
        main_boxes=[]
        for column in column_boxes:
            main_column = [elem for elem in column if (elem not in header_boxes[0] and elem not in footer_boxes[0])]
            main_boxes.append(main_column)
        # Step 5: Order paragraphs
        ordered_paragraphs = []
        if header_boxes:
            ordered_paragraphs.extend(self.order_horizontal_colums(header_boxes))
        if column_boxes:
            if num_columns == 1:
                ordered_paragraphs.extend(self.order_one_column(main_boxes[0]))
            elif num_columns == 2:
                left_column, right_column = self.order_two_columns(main_boxes)
                ordered_paragraphs.extend(left_column)
                ordered_paragraphs.extend(right_column)
        if footer_boxes:
            ordered_paragraphs.extend(self.order_horizontal_colums(footer_boxes))
        return ordered_paragraphs
    
    
    #  Docs assembly
    def crop_and_concat(self, image_path, bounding_boxes, tag='vertical'):
        """ This function is used to crop and concatenate images from image given a lsit of ordered bounding boxes
        Args:
            image_path (str): path to the image to crop
            bounding_boxes (list): list of ordered bounding boxes
            tag (str, optional): sufix of the output file. Defaults to'vertical'.
        Returns:
            str: path of the concatenated image
        """
        # Open the original image
        original_image = Image.open(image_path)
        # Create an empty list to store cropped images
        cropped_images = []
        # Crop the image according to each bounding box
        for box in bounding_boxes:
            left, top, right, bottom = box
            cropped_image = original_image.crop((left, top, right, bottom))
            cropped_images.append(cropped_image)
        # Calculate the total height for the new image
        total_height = sum(cropped_image.height for cropped_image in cropped_images)
        # Create a new image with the same width as the original image and the total height
        new_image = Image.new('RGB', (original_image.width, total_height), color='white')
        # Paste each cropped image onto the new image
        current_height = 0
        for cropped_image in cropped_images:
            new_image.paste(cropped_image, (0, current_height))
            current_height += cropped_image.height
        # Save or display the resulting image
        # new_image.show()
        # Alternatively, you can save the image using the following line
        path=os.path.join(os.path.dirname(image_path), "".join(os.path.basename(image_path).split('.')[:-1]))
        save_path = f"{path}_{tag}.jpg"
        new_image.save(save_path)
        return save_path 

    def replace_from_extracted(self, text, tables_json_file_path, figures_json_file_path, equations_json_file_path):
        """
        This function replaces the extracted tables, figures and equations in the given text 
        Args:
            text (_str): simple ocr extracted text
            tables_json_file_path (_str): path of tables json
            figures_json_file_path (_str): path of figures json
            equations_json_file_path (_str): path of equations json
        Returns:
            str: text with replacements 
        """
        replacements = {}
        with open(tables_json_file_path, 'r') as file:
            json_data = json.load(file)
            replacements.update(json_data)
        with open(figures_json_file_path, 'r') as file:
            json_data = json.load(file)
            replacements.update(json_data)
        with open(equations_json_file_path, 'r') as file:
            json_data = json.load(file)
            replacements.update(json_data)
        for key, value in replacements.items():
            text=text.replace(f'**{key}**', value)
        return text
    
    def load_pdf(self, pdf_path, output_folder='data/extraction', save_intermediate=False, header_height = 0, footer_height = 0):
        """this metod thakes a pdf file and extracts the text, table and images from it

        Args:
            pdf_path (str): file path
            output_folder (str, optional): pat for storing intermediate files. Defaults to '../data.extraction'.
            save_intermediate (bool, optional): if is required to save some intermediuate results for debug purpouses. Defaults to False.
            header_height (int, optional): header height in pixels. Defaults to 0.
            footer_height (int, optional): footer height in pixels. Defaults to 0.

        Returns:
            list: list of texts, html like tables, and figures referencesof each apge content
        """
        file_ouput_forder=self.convert_pdf_to_images(pdf_path, output_folder=output_folder)
        texts=[]
        #iterate over each page of the pdf
        image_paths=list(os.listdir(file_ouput_forder))
        for image_path in image_paths:
            #open image
            file_path=os.path.join(file_ouput_forder,image_path)
            img =  Image.open(file_path)
            # process layout detaction
            structured_ocr_result=self.structured_ocr(file_path)
            # save tables
            tables_bboxs, tables = self.get_tables(structured_ocr_result)
            tables_json_file_path = self.save_tables(tables, file_path, save_html=save_intermediate)
            _, masked_img = self.mask_elements_from_image(img, tables_bboxs, "table")
            # save figures
            figures_bboxs, figures = self.get_figures(structured_ocr_result)
            figures_json_file_path = self.save_figures(figures, file_path)
            _, masked_img = self.mask_elements_from_image(masked_img, figures_bboxs, "figure")
            # save equations and save masked image
            equations_bboxs, equations = self.get_equations(structured_ocr_result)
            equations_json_file_path = self.save_equations(equations, file_path)
            mask_img_path, masked_img = self.mask_elements_from_image(masked_img, equations_bboxs, "equation", save=True, image_path=file_path)
            #show structure bboxes 
            _structured_bboxes_image = self.show_paddle_structure_bboxs(file_path, structured_ocr_result, save=save_intermediate)
            #clean up and order bounding boxes
            bboxes = [element["bbox"] for element in structured_ocr_result]
            bboxes = self.get_content_bboxes(bboxes)
            img_size = (img.getbbox()[2],img.getbbox()[3])
            bboxes = self.order_paragraphs(bboxes, img_size, header_height=header_height,footer_height=footer_height)
            bboxes = self.expand_bounding_boxes(bboxes, img_size, 3)
            print(save_intermediate)
            _paragraph_bboxes_image = self.show_simple_bboxes(file_path, bboxes, save=save_intermediate, tag="ordered")
            # create new in line image from masked image and ordered bboxes
            final_image_path = self.crop_and_concat(mask_img_path, bboxes, tag="vertical")
            # do simple ocr over the inline masked image, and unmask text 
            simple_ocr_result = self.simple_ocr(final_image_path, ocr=ocr)
            masked_text = "\n".join([line[1][0] for line in simple_ocr_result])
            full_text = self.replace_from_extracted(masked_text, tables_json_file_path, figures_json_file_path, equations_json_file_path)
            texts.append(full_text)
        return texts

if __name__ == "__main__": 
    loader=PaddleOCRLoader(sys.argv[1])
    print(loader.load())