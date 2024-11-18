from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# taken from https://huggingface.co/docs/transformers/model_doc/trocr#inference
class TrOCRInference:
    def __init__(self, model_path="microsoft/trocr-base-handwritten"):
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)

    def extract_text(self, image):
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text[0]