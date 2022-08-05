from django.http import HttpResponse, JsonResponse
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from joblib import load


def index(request):
    return HttpResponse("This is Cal's machine learning service.")


def health(request):
    return HttpResponse("OK")


def predict_digit(request):
    """This endpoint is used to predict the digit in an image."""

    # Load the model from the file
    digits_model = load('./machine_learning_projects/trained_models/digits_model.joblib')

    return HttpResponse("OK")
