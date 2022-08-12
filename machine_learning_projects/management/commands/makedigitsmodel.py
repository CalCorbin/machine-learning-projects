# pylint: disable=no-member
"""Module for Django management commands makedigitsmodel."""
from django.core.management.base import BaseCommand
from joblib import dump
from sklearn import metrics
from sklearn import ensemble

# Tensorflow is really big, so only install it if you need to generate a new model.
# pylint: disable=import-error
import tensorflow as tf

TRAINED_MODEL_PATH = 'machine_learning_projects/trained_models/digits_model.joblib'


class Command(BaseCommand):
    """Command class for Django management command makedigitsmodel.
    This command trains a model around the digits dataset."""
    help = 'Trains a model around the digits dataset'

    def handle(self, *args, **options):
        self.stdout.write('\nTraining a model around the digits dataset...')

        # Create a classifier: a support vector classifier
        forest_classifier = ensemble.RandomForestClassifier()

        # Split data into train and test subsets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # Learn the digits on the train subset
        forest_classifier.fit(x_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = forest_classifier.predict(x_test)

        # Generate a classification report
        digits_classification_report = metrics.classification_report(y_test, predicted)
        print('\nClassification Report:\n', digits_classification_report)

        # Save the model to a file
        dump(forest_classifier, TRAINED_MODEL_PATH)

        self.stdout.write(self.style.SUCCESS(f'Done! Model saved to {TRAINED_MODEL_PATH}'))
        return 0
