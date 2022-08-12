# pylint: disable=no-member
"""Module for Django management commands makedigitsmodel."""
import numpy as np
from django.core.management.base import BaseCommand
from joblib import dump
from sklearn import metrics
from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV

# Tensorflow is really big, so only install it if you need to generate a new model.
# pylint: disable=import-error
import tensorflow as tf

TRAINED_MODEL_PATH = 'digits_model.joblib'

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


class Command(BaseCommand):
    """Command class for Django management command makedigitsmodel.
    This command trains a model around the digits dataset."""
    help = 'Trains a model around the digits dataset'

    def add_arguments(self, parser):
        parser.add_argument(
            '--best_params',
            action='store_true',
            help='Print the best Random Forest parameters found'
        )

    def handle(self, *args, **options):
        self.stdout.write('\nTraining a model around the digits dataset...')

        # Create random forest classifier.
        forest_classifier = ensemble.RandomForestClassifier(
            random_state=42,
            bootstrap=False,
            max_depth=40,
            max_features='auto',
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=1400
        )

        # Split data into train and test subsets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        if options['best_params']:
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores.
            forest_classifier_random = RandomizedSearchCV(
                forest_classifier,
                param_distributions=random_grid,
                n_iter=100,
                cv=3,
                verbose=2,
                random_state=42,
                n_jobs=-1
            )
            forest_classifier_random.fit(x_train, y_train)
            self.stdout.write('\nBest parameters:')
            print(forest_classifier_random.best_params_)
            self.stdout.write('\nBest score:')
            print(forest_classifier_random.best_score_)
            self.stdout.write('\n')

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
