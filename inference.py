# Source:
# https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html
# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html

import json, logging, sys, os, io, requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def net():
    logger.info("Creating model ...")
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 5)
    )
    return model


def model_fn(model_dir):
    logger.info(f"Loading model ...")
    logger.info(f"Model directory: {model_dir}")
    model = net().to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location = device))
    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info(f"Parsing input ...")
    logger.info(f"Content type: {content_type}")
    logger.info(f"Request body: {request_body}")
    if content_type == JPEG_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body))
    if content_type == JSON_CONTENT_TYPE:
        request = json.loads(request_body)
        logger.info(f"Loaded JSON object: {request}")
        img_content = requests.get(request["url"]).content
        return Image.open(io.BytesIO(img_content))
    raise Exception("Unsupported content type: {}".format(content_type))
   
        
def predict_fn(input_object, model):
    logger.info("Predicting ...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_object=transform(input_object)
    if torch.cuda.is_available():
        input_object = input_object.cuda()
    model.eval()
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction