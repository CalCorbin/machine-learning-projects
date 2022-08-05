from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import json

from joblib import load

trained_model_path = 'machine_learning_projects/trained_models/digits_model.joblib'
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

    # data = json.loads(request.body)
    # print(data)
    # digits_model.score(X_test, y_test)

    return HttpResponse("OK")
