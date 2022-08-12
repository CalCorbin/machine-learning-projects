"""Module for managing Views."""
from io import BytesIO
import os
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from joblib import load
from PIL import Image
from dotenv import load_dotenv
import boto3
import numpy as np

load_dotenv()

# Load the model
DIGITS_MODEL = ''
s3 = boto3.resource(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)
if os.getenv('CURRENT_ENV') == 'local':
    DIGITS_MODEL = load(os.getenv('DIGITS_MODEL'))
else:
    with BytesIO() as data:
        s3.Bucket(os.getenv('AWS_S3_BUCKET')).download_fileobj(os.getenv('DIGITS_MODEL'), data)
        data.seek(0)
        DIGITS_MODEL = load(data)


def index():
    """Index endpoint."""
    return HttpResponse("This is Cal's machine learning service.")


def health():
    """Health endpoint."""
    return HttpResponse("OK")


@csrf_exempt
def predict_digit(request):
    """This endpoint is used to predict the digit in an image."""

    if request.method != 'POST':
        return JsonResponse(
            {'error': 'This endpoint only accepts POST requests.'},
            status=400
        )

    if 'imageFile' not in request.FILES:
        return JsonResponse(
            {'error': 'Did you forget to provide a png image of a digit?'},
            status=400
        )

    # Load the image submitted by the user and convert to gray scale.
    image = request.FILES['imageFile']
    image = Image.open(image).convert('L')

    # Resize the image.
    image = image.resize((28, 28))
    image = np.array(image).flatten()
    image = image.reshape(1, -1)

    # Predict the digit in the image.
    predicted = DIGITS_MODEL.predict(image)

    # return HttpResponse("OK")
    return JsonResponse({'predicted': predicted.tolist()})
