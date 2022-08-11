from django.core.management.base import BaseCommand, CommandError
from joblib import dump
from sklearn import datasets, svm, metrics
from sklearn import ensemble
import tensorflow as tf

trained_model_path = 'machine_learning_projects/trained_models/digits_model.joblib'


class Command(BaseCommand):
    help = 'Trains a model around the digits dataset'

    def handle(self, *args, **options):
        self.stdout.write('\nTraining a model around the digits dataset...')

        # Create a classifier: a support vector classifier
        forest_classifier = ensemble.RandomForestClassifier()

        # Split data into train and test subsets
        (X_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # Learn the digits on the train subset
        forest_classifier.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = forest_classifier.predict(x_test)

        # Generate a classification report
        digits_classification_report = metrics.classification_report(y_test, predicted)
        print('\nClassification Report:\n', digits_classification_report)

        # Save the model to a file
        dump(forest_classifier, trained_model_path)

        self.stdout.write(self.style.SUCCESS('Done! Model saved to %s' % trained_model_path))
        return 0
