import json
import yaml
import pandas as pd
import os, glob
import yaml
import json
import csv
import xml.etree.ElementTree as ET
from roboflow import Roboflow
import os
import argparse


class AnnotationEval():
    def __init__(
        self,
        local_folder: str,
        roboflow_folder: str
    ):
        
        """
        Loads local images + labels and roboflow images + labels. 
        Checks that each label in each image locally matches the label in roboflow

            Args:
                local_folder (str): local folder
                roboflow_folder (str): data downloaded/exported from roboflow.

            Returns:
                Similarity Score
        """
        self.local_folder = local_folder
        self.roboflow_folder = roboflow_folder
    

    def extract_text_after_last_slash(self,text):
        last_slash_index = text.rfind('/')
        if last_slash_index != -1:
            text_after_last_slash = text[last_slash_index + 1:]
            return text_after_last_slash
        return None

    def parse_xml(self,xml_string):
        tree = ET.parse(xml_string)
        root = tree.getroot()
        bbox_dict = {}
        # Find all <object> elements
        object_elements = root.findall('.//object')
        # Iterate over each <object> element
        for i, object_element in enumerate(object_elements):
            bbox = {}
            # Extract bounding box coordinates
            bndbox_element = object_element.find('bndbox')
            bbox['xmin'] = round(float(bndbox_element.find('xmin').text))
            bbox['xmax'] = round(float(bndbox_element.find('xmax').text))
            bbox['ymin'] = round(float(bndbox_element.find('ymin').text))
            bbox['ymax'] = round(float(bndbox_element.find('ymax').text))
            # Extract name
            name_element = object_element.find('name')
            bbox['name'] = name_element.text
            # Add bbox to the dictionary with counter as the key
            bbox_dict[i] = bbox
        return bbox_dict

    def collect_images_labels(self):
        
        self.local_images = []
        self.local_annotations = []
        self.roboflow_annotations = []
        
        # Loop through each subfolder in image_folder
        for root, dirs, files in os.walk(self.local_folder):
            # Check if 'xml' folder exists in the current subfolder
            if 'xml' in dirs:
                xml_folder = os.path.join(root, 'xml')
                # Get all XML files in xml_folder
                #xml_files = [file for file in os.listdir(xml_folder) if file.endswith('.xml')]
                xml_files = sorted(glob.glob(os.path.join(xml_folder, "*.xml")))
                # Add the XML files to the master_list
                self.local_annotations.extend(xml_files)

        # Loop through each subfolder in image_folder
        for root, dirs, files in os.walk(self.local_folder):
            # Check if 'images' folder exists in the current subfolder
            if 'images' in dirs:
                images_folder = os.path.join(root, 'images')
                # Get all JPEG files in the images_folder
                #jpeg_files = [file for file in os.listdir(images_folder) if file.endswith('.jpg') or file.endswith('.jpg')]
                jpeg_files = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))


                # Add the JPEG files to the local_images list
                self.local_images.extend(jpeg_files)
                
        name_requirements = ["train","test","valid"]
        # Loop through each subfolder in image_folder
        for root, dirs, files in os.walk(self.roboflow_folder):
            # Check if 'images' folder exists in the current subfolder
            for folder in dirs:
                # Check if the subfolder meets the name requirement
                if any(req in folder for req in name_requirements):

                    xml_path = os.path.join(root, folder)
                    # Get all XML files in the subfolder
                    #xml_files = [file for file in os.listdir(xml_path) if file.endswith('.xml')]
                    xml_files = sorted(glob.glob(os.path.join(xml_path, "*.xml")))
                    # Add the XML files to the local_images list
                    self.roboflow_annotations.extend(xml_files)
                    
        # Print the local_images list
        print('local image count',len(self.local_images))
        print('local annotation count',len(self.local_annotations))
        print('robfolow annotation count',len(self.roboflow_annotations))
        return self.local_images,self.local_annotations,self.roboflow_annotations


    def run_eval_loop(self):
    
        count = 0
        loop_count = 0
        roboflow_count = 0
        match1 = 0
        overall_accuracy = []
        no_difference_count = 0
        roboflow_key_count = 0
        local_key_count = 0
        key_match = 0

        for image in self.local_images:
            if count < len(self.local_images):
                               
                f = os.path.join(image)
                image_hash = self.extract_text_after_last_slash(image.split(".")[0].replace("#","-"))

                # split the image path to the hash
                current_annotation = self.local_annotations[count]
                annotation_hash = self.extract_text_after_last_slash(current_annotation.split(".")[0]).replace("#","-")
                
                if image_hash == annotation_hash:
                
                    match1 +=1 
                
                    for roboflow_annotation in self.roboflow_annotations:
                        #Roboflow labels and hash
                        roboflow_hash = ((roboflow_annotation.split("/"))[-1].split('.')[0][:-4])
                        
                        if roboflow_hash == image_hash:
                        
                            roboflow_count +=1
                                
                            local_parsed = self.parse_xml(current_annotation)
                            roboflow_parsed = self.parse_xml(roboflow_annotation)
                            
                            label_count_local = len(local_parsed)
                            roboflow_count_local = len(roboflow_parsed)
                            local_key_count += label_count_local
            
                            for key in local_parsed: 
                                
                                if key in roboflow_parsed:
                                    roboflow_key_count+=1
                                    difference = 0
                                    for sub_key in local_parsed[key]:
                                        if sub_key in roboflow_parsed[key] and type(local_parsed[key][sub_key]) == int and (local_parsed[key][sub_key]-local_parsed[key][sub_key]) >1:
                                            difference += (local_parsed[key][sub_key] - roboflow_parsed[key][sub_key])
                                        if type(local_parsed[key][sub_key]) == str:
                                            if local_parsed[key][sub_key] != roboflow_parsed[key][sub_key]:
                                                difference += 1
                                    if difference <=1:
                                        no_difference_count +=1
                                
                                    elif difference >1:
                                        print('PIXEL MISMATCH')
                                        print(image_hash,annotation_hash,roboflow_hash)
                                        print(difference)
                    
             
                    count+=1

                    if loop_count > len(self.local_images)*len(self.local_annotations):
                        break
                    
        print('\n')
        print('KEY_MATCH %',str((roboflow_key_count/local_key_count)*100)+'%')
        print('\n')
        print('LABEL SIMILARITY %',str((no_difference_count/roboflow_key_count)*100)+'%')
        print('\n')
        print('TOTAL LABELS',local_key_count)
        print('\n')
        print('TOTAL IMAGE MATCH',match1)
