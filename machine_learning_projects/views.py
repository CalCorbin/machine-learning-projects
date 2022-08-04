from django.http import HttpResponse, JsonResponse
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import json


def index(request):
    return HttpResponse("This is Cal's machine learning service.")


def health(request):
    return HttpResponse("OK")


def train_digits(request):
    """
    This function takes the digit dataset from sklearn and trains
    a model around the data.
    """

    # Load digits and flatten the images
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.15, shuffle=False
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    # Generate a classification report
    digits_classification_report = metrics.classification_report(
        y_test, predicted, target_names=digits.target_names, output_dict=True
    )
    digits_pd_dataframe = pd.DataFrame(digits_classification_report).transpose()
    digits_pd_dataframe = digits_pd_dataframe.to_json(orient='records', indent=2)
    parsed = json.loads(digits_pd_dataframe)

    return JsonResponse({"success": True, "classification_report": parsed})
