from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from joblib import load
from PIL import Image
import numpy as np

digits_model = load('./machine_learning_projects/trained_models/digits_model.joblib')


def index(request):
    return HttpResponse("This is Cal's machine learning service.")


def health(request):
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
    image = image.resize((28, 28), Image.ANTIALIAS)
    image = np.array(image).flatten()
    image = image.reshape(1, -1)

    # Predict the digit in the image.
    predicted = digits_model.predict(image)

    # return HttpResponse("OK")
    return JsonResponse({'predicted': predicted.tolist()})
