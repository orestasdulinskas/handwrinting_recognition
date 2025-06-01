import json
import os
import base64
import pickle
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models.detection as detection
from textblob import TextBlob
import boto3
import io

s3_client = boto3.client('s3')

# initialize models
def initialize_models():
    # set PyTorch hub directory to /tmp in Lambda
    torch.hub.set_dir('/tmp/torch_hub')
    
    # device selection
    device = torch.device("cpu")
    
    bucket_name = os.environ.get('MODEL_BUCKET', 'journal-transcription')
    annotation_model_key = os.environ.get('ANNOTATION_MODEL_KEY', 'annotation_model.pth')
    ocr_model_key = os.environ.get('OCR_MODEL_KEY', 'model_state.pth')
    num_to_char_key = os.environ.get('NUM_TO_CHAR_KEY', 'num_to_char.pkl')
    
    def load_model_from_s3(bucket, key):
        try:
            print(f"Loading {key} from S3 bucket {bucket}")
            response = s3_client.get_object(Bucket=bucket, Key=key)
            model_bytes = response['Body'].read()
            buffer = io.BytesIO(model_bytes)
            buffer.seek(0)
            return buffer
        except Exception as e:
            print(f"Error loading {key} from S3: {str(e)}")
            raise
    
    # load auto-annotator
    print("Initializing auto-annotator")
    annotation_model = detection.fasterrcnn_resnet50_fpn(pretrained=False, weights=None)
    in_features = annotation_model.roi_heads.box_predictor.cls_score.in_features
    annotation_model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    
    # load weights from S3
    print("Loading annotation model weights from S3")
    annotation_buffer = load_model_from_s3(bucket_name, annotation_model_key)
    state_dict = torch.load(annotation_buffer, map_location=device)
    annotation_model.load_state_dict(state_dict)
    annotation_model.to(device)
    annotation_model.eval()
    
    # load decoder from S3
    print("Loading decoder from S3")
    num_to_char_buffer = load_model_from_s3(bucket_name, num_to_char_key)
    num_to_char = pickle.load(num_to_char_buffer)
    num_characters = len(num_to_char)
    
    # define & load OCR model
    print("Initializing OCR model")
    ocr_model = HandwritingRecognitionModel(num_characters)
    ocr_buffer = load_model_from_s3(bucket_name, ocr_model_key)
    ocr_model.load_state_dict(torch.load(ocr_buffer, map_location=device))
    ocr_model.to(device)
    ocr_model.eval()
    
    print("All models initialized successfully")
    return device, annotation_model, ocr_model, num_to_char

# define OCR model architecture
class HandwritingRecognitionModel(nn.Module):
    def __init__(self, num_characters):
        super(HandwritingRecognitionModel, self).__init__()
        
        self.cnn = nn.Sequential(
            # first CNN layer
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Dropout2d(0.5),
            
            # second CNN layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Dropout2d(0.5),
            
            # third CNN layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            
            # fourth CNN layer
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.5),

            # fifth CNN layer
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.lstm_input_size = 512 * 12
        
        # bi-LSTM network
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=256,
            dropout=0.7,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.output = nn.Linear(512, num_characters)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.permute(0, 3, 1, 2)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.output(x)
        
        return nn.functional.log_softmax(x, dim=2)

# define fixed height
FIXED_HEIGHT = 45

# initialize models at module level
device, annotation_model, ocr_model, num_to_char = initialize_models()

def predict_and_visualize_from_image(model, image, device, threshold=0.8):
    # transform image to tensor
    orig_image = image.copy()
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # prediction
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # extract boxes and scores
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # keep boxes above threshold
    true_boxes = []
    for i, box in enumerate(boxes):
        if scores[i] > threshold:
            true_boxes.append(box)
    
    # draw bounding boxes on the image
    for i, box in enumerate(true_boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(orig_image, f"{scores[i]:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    true_boxes = sorted(true_boxes, key=lambda x: x[1])
    return image, true_boxes, orig_image

def preprocess_crop(image, box):
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]

    h = y2 - y1
    w = x2 - x1

    aspect_ratio = w / h if h != 0 else 1
    new_width = max(1, int(FIXED_HEIGHT * aspect_ratio))

    resized = cv2.resize(crop, (new_width, FIXED_HEIGHT))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype(np.float32) / 255.0
    normalized = np.expand_dims(normalized, axis=-1)

    return normalized

def collate_inference_images(image_tensors):
    widths = [img.shape[1] for img in image_tensors]
    max_width = max(widths)

    # pad tensors to max width
    padded_images = []
    for img in image_tensors:
        pad_width = max_width - img.shape[1]
        padded = np.pad(img, ((0, 0), (0, pad_width), (0, 0)), mode="constant")
        padded_images.append(padded)

    batch = np.stack(padded_images, axis=0)
    batch = np.transpose(batch, (0, 1, 2, 3))
    return torch.tensor(batch, dtype=torch.float32)

def decode_predictions(preds, num_to_char):
    pred_indices = torch.argmax(preds, dim=2).detach().cpu().numpy()
    
    # decode predictions
    decoded = []
    for indices in pred_indices:
        prev_idx = None
        chars = []
        for idx in indices:
            if idx != prev_idx and idx != 0: # 0 is <PAD>
                chars.append(num_to_char.get(idx, ""))
            prev_idx = idx
        decoded.append("".join(chars))

    return decoded

def run_inference(model, image, boxes, num_to_char, device):
    image_tensors = [preprocess_crop(image, box) for box in boxes]
    batch = collate_inference_images(image_tensors).to(device)
    batch = batch.contiguous()

    with torch.no_grad():
        preds = model(batch)

    return decode_predictions(preds, num_to_char)

def transcribe_handwriting(image):
    # run auto-annotator
    _, boxes, annotated_image = predict_and_visualize_from_image(annotation_model, image, device)

    # run OCR prediction
    results = run_inference(ocr_model, image, boxes, num_to_char, device)
    raw_text = " ".join(results)
    
    # replacement dictionary
    replace_dict = {"  ": " ", "   ": " ", " t ":" ", " f ":" ", " b ":" ", " d ":" "}

    # loop to replace
    for old, new in replace_dict.items():
        raw_text = raw_text.replace(old, new)

    # correct the spelling
    corrected_text = str(TextBlob(raw_text).correct())

    return raw_text, corrected_text, boxes, annotated_image

def decode_image(image_data):
    try:
        # check if the data is already bytes
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
        # check if it's base64 encoded
        elif isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
            nparr = np.frombuffer(image_data, np.uint8)
        else:
            raise ValueError("Unsupported image data format")
            
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    except Exception as e:
        raise Exception(f"Error decoding image: {str(e)}")

def lambda_handler(event, context):

    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({})
        }

    try:
        start_time = time.time()
        
        # determine input source
        image = None
        
        # direct image data in the event body
        if 'image' in event:
            image = decode_image(event['image'])
        
        # API Gateway request with base64-encoded body
        elif 'body' in event:
            body = event['body']
            if event.get('isBase64Encoded', False):
                body = base64.b64decode(body)
            
            if isinstance(body, str):
                try:
                    body_json = json.loads(body)
                    if 'image' in body_json:
                        image = decode_image(body_json['image'])
                except json.JSONDecodeError:
                    pass
        
        # S3 event
        elif 'Records' in event and len(event['Records']) > 0:

            import boto3
            s3_client = boto3.client('s3')
            
            record = event['Records'][0]
            if record.get('eventSource') == 'aws:s3':
                bucket = record['s3']['bucket']['name']
                key = record['s3']['object']['key']
                
                response = s3_client.get_object(Bucket=bucket, Key=key)
                image_data = response['Body'].read()
                image = decode_image(image_data)
        
        if image is None:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No valid image provided in the request'})
            }
        
        # process the image
        raw_text, corrected_text, boxes, annotated_image = transcribe_handwriting(image)
        inference_time = round(time.time() - start_time, 2)

        if event.get('complete', False) or event.get('dev', False):
            # encode the annotated image as base64
            success, buffer = cv2.imencode('.jpg', annotated_image)
            if not success:
                raise Exception("Failed to encode annotated image")
            annotated_image_base64 = base64.b64encode(buffer).decode("utf-8")
            
            response_body = {
                'raw_transcription': raw_text,
                'corrected_transcription': corrected_text,
                'inference_time': inference_time,
                'annotated_image': annotated_image_base64
            }
        else:
            response_body = {
                'corrected_transcription': corrected_text
            }
            
        return {
            'statusCode': 200, 
            'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        # log for CloudWatch
        print(f"Error processing request: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f"Failed to process image: {str(e)}"
            })
        }

if __name__ == "__main__":
    test_event = {
        'dev': True,
        'image': None
    }
    print(lambda_handler(test_event, None))