import base64
from enum import Enum
from io import BytesIO

import torch
import cv2
import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO
import clip

# Constants
YOLO_MODEL_WEIGHTS = './yoloWeights.pt'
ENCODING = 'utf-8'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# minimum detection confidence for the yolo model - if a piece of clothing is predicted with less than 50% confidence we
# exclude it
MINIMUM_MASK_DETECTION_CONFIDENCE = .5


class InputImageType(Enum):
    """
    Used to specify whether input request contains a url of an image to download or a base 64 encoded image
    """
    URL = "URL"
    BASE_64 = "BASE64"


class Item(BaseModel):
    """
    Used to determine the input request shape
    image - url string or base64 encoded image string
    inputImageType - URL or base64
    embedding_only - whether the segmentation model should run or not
    """
    image: str
    inputImageType: InputImageType
    embedding_only = False


# Load the yolo model weights from the local directory and download the CLIP model weights
YOLO_MODEL = YOLO(YOLO_MODEL_WEIGHTS)
CLASS_NAMES = YOLO_MODEL.names
EMBEDDING_MODEL, EMBEDDING_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)


def get_image_from_bytes(image):
    """
    Converts a base64 image string to a usable image format using numpy and opencv
    :param image:
    :return:
    """
    base64_bytes = base64.b64decode(image)
    nparr = np.frombuffer(base64_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np

def download_image(image_url):
    """
    If an url is specified than download it. Otherwise, just use default image
    :param image_url: url to download
    :return: openCV of image
    """
    if image_url:
        r = requests.get(image_url)
    else:
        r = requests.get(
            'https://i.pinimg.com/236x/fc/2e/29/fc2e29043ae177bbf2436e13f26fd5eb--the-sartorialist-mans.jpg')

    PIL_image = Image.open(BytesIO(r.content))
    open_cv_image = np.array(PIL_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def get_image_embedding(image):
    """
    Converts an openCV format image to a PIL image. Pre-processses it for CLIP. Runs its embedding and then
    normalizes  it
    :param image: openCV
    :return: normalized CLIP image embedding
    """
    PILimage = Image.fromarray(image)
    pre_processed_image = (
        EMBEDDING_PREPROCESS(PILimage)
        .unsqueeze(0)
        .to(DEVICE)
    )
    with torch.no_grad():
        image_features = EMBEDDING_MODEL.encode_image(pre_processed_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return np.array(image_features.cpu().numpy()).tolist()


def predict(item, run_id, logger):
    item = Item(**item)

    # If doing embedding only then don't do yolo inference for segmentation
    embedding_only = item.embedding_only

    if item.inputImageType == InputImageType.URL:
        image = download_image(item.image)
    elif item.inputImageType == InputImageType.BASE_64:
        image = get_image_from_bytes(item.image)
    else:
        image = download_image('')

    class_names = []
    masks = []
    embeddings = []
    confidences = []

    if not embedding_only:
        yolo_results = YOLO_MODEL.predict(image, imgsz=640, verbose=False, conf=MINIMUM_MASK_DETECTION_CONFIDENCE)[0]
        for i in range(len(yolo_results.masks)):
            class_id = int(yolo_results.boxes.cls[i].item())
            class_name = CLASS_NAMES[class_id]
            mask = yolo_results.masks.xy[i].tolist()
            embedding = get_image_embedding(image)
            class_names.append(class_name)
            masks.append(mask)
            embeddings.append(embedding)
            confidences.append(yolo_results.boxes.conf[i].item())
    else:
        embedding = get_image_embedding(image)
        embeddings.append(embedding)

    return {
        "class_names": class_names,
        "masks": masks,
        "embeddings": embeddings,
        "confidences": confidences
    }
