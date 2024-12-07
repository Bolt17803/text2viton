from diffusers import StableDiffusionPipeline
import torch
import os
import torch
import cv2
# from inference import get_model
# import supervision as sv
# import numpy as np
# import supervision as sv
# from diffusers import StableDiffusionPipeline
# from segment_anything import sam_model_registry, SamPredictor
# from groundingdino.util.inference import Model
# from transformers import AutoProcessor, AutoModel
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from segment_anything import SamPredictor, sam_model_registry
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import numpy as np
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

model_dir = "TokenCompose/TokenCompose_SD21_A"

pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)

pipe = pipe.to("cuda")

prompt = "full image of a half sleeves yellow tshirt with stripes having adidas slogo on it"
# prompt = "Full image of a full sleeves green color t-shirt with guess logo"


image = pipe(prompt).images[0]

save_path = "./Modified_VirtualTryON/DATA/test/cloth/test_generated_image2.jpg"
image.save(save_path)

print("Image saved as generated_image.png")

#=================== padding =======================
# # Load image
image_path = save_path#"./Modified_VirtualTryON/DATA/test/cloth/test_generated_image1.jpg"  # Replace with your image file path
image = cv2.imread(image_path)
height, width = image.shape[:2]
top_padding = (1024 - height) // 2
bottom_padding = 1024 - height - top_padding  # Ensure total padding adds up to 256 pixels

# Pad the image
padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

new = "./Modified_VirtualTryON/DATA/test/cloth/test_generated_image_processed2.jpg"
cv2.imwrite(new, padded_image)
#---------------------mask generator
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

image = Image.open(new)
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits.cpu()

predicted_class_ids = torch.argmax(logits, dim=1)

# Convert the predictions to numpy
predicted_class_ids = predicted_class_ids.squeeze().cpu().numpy()

upper_clothing_class_id = 4

upper_clothing_mask = (predicted_class_ids == upper_clothing_class_id).astype(np.uint8) * 255

dress_class_id = 7

dress_mask = (predicted_class_ids == upper_clothing_class_id).astype(np.uint8) * 255


result = cv2.bitwise_or(upper_clothing_mask, dress_mask)

resized_mask = cv2.resize(result, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./Modified_VirtualTryON/DATA/test/cloth-mask/test_generated_image_processed2.jpg", resized_mask)

