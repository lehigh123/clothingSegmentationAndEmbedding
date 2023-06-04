import base64
from io import BytesIO
from random import randrange

import cv2
import numpy as np
import requests
from PIL import Image
from matplotlib import pyplot as plt

ENCODING = 'utf-8'


def display_mask_on_image(img_path,
                          mask_points,
                          class_name,
                          mask_color=[255, 0, 0],
                          mask_outline=[255, 0, 0]):
    image = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    image = cv2.fillPoly(image, mask_points, mask_color, 1)
    image = cv2.polylines(image, mask_points,
                          False, mask_outline, 2)
    plt.title(class_name)
    plt.imshow(image)
    plt.show()


def display_response(response_json, image, class_filter={}, consolidate_same_classes=True):
    if consolidate_same_classes:
        image_annotations = {}
        for i in range(len(response_json['result']['class_names'])):
            class_name = response_json['result']['class_names'][i]
            if (len(class_filter) > 0):
                if (class_name not in class_filter):
                    continue
            mask = response_json['result']['masks'][i]
            points = np.array(mask, dtype=np.int32)
            current_points = image_annotations.get(class_name, [])
            current_points.append(points)
            image_annotations[class_name] = current_points
        for class_name in image_annotations:
            mask_points = image_annotations[class_name]
            mask_color = (randrange(0, 255), randrange(0, 255), randrange(0, 255))
            display_mask_on_image(image, mask_points, class_name, mask_color=mask_color)
    else:
        for i in range(len(response_json['result']['class_names'])):
            class_name = response_json['result']['class_names'][i]
            if (len(class_filter) > 0):
                if (class_name not in class_filter):
                    continue
            mask = response_json['result']['masks'][i]
            points = np.array(mask, dtype=np.int32)
            if type(points) != list:
                points = [points]
            display_mask_on_image(
                image,
                points,
                class_name
            )


def get_image_from_web(online_image_url):
    try:
        r = requests.get(online_image_url)
        pil_image = Image.open(BytesIO(r.content))
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image
    except Exception as e:
        print("Error getting: ", e)
        return None


def get_base_64_string(input_image_path):
    with open(input_image_path, "rb") as image_file:
        base64Bytes = base64.b64encode(image_file.read())
        base64_string = base64Bytes.decode(ENCODING)
    return base64_string


def get_image_from_bytes(image):
    base64_bytes = base64.b64decode(image)
    nparr = np.frombuffer(base64_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return img_np
