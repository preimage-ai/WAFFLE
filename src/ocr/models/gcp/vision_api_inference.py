import json
import os

from google.cloud import vision
from google.cloud.vision_v1.types.text_annotation import TextAnnotation


class GcpVisionInference:
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def get_document_text_detection(self, img_path):
        """
        Returns Vision AI full JSON reponse from an image

        Args:
            uri: string, path to image

        Returns:
            string, a full JSON response from Vision AI
        """
        image = vision.Image()
        image.content = open(img_path, "rb").read()
        response = self.client.document_text_detection(image=image)
        return TextAnnotation.to_json(response.full_text_annotation)
    
    def store_document_text_detection(self, img_path, output_path):
        """
        Stores Vision AI full JSON reponse from an image

        Args:
            uri: string, path to image
            output_path: string, path to store the JSON response
        """
        if os.path.exists(output_path):
            # print(f"File {output_path} already exists, skipping")
            return output_path
        document = self.get_document_text_detection(img_path)
        with open(output_path, "w") as f:
            json.dump(document, f)
            return output_path