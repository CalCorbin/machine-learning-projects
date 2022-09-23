"""Module for testing machine_learning_project views."""
import os
from django.test import TestCase
from django.urls import reverse


class IndexTestCase(TestCase):
    """Test the index endpoint."""

    def test_index(self):
        """Test the index endpoint."""
        url = reverse("index")
        response = self.client.get(url)

        assert response.status_code == 200
        assert response.content == \
               b"Hello, stranger. You're at the machine_learning_projects index."


class HealthTestCase(TestCase):
    """Test the health endpoint."""

    def test_health(self):
        """Test the health endpoint."""
        url = reverse("health")
        response = self.client.get(url)

        assert response.status_code == 200
        assert response.content == b"ML Projects is healthy."


class PredictDigitTestCase(TestCase):
    """Test the predict-digit endpoint."""

    def test_should_return_400_when_no_image_is_provided(self):
        """This test is to ensure that the endpoint
        returns a 400 status code when no image is provided."""
        url = reverse("predict-digit")
        response = self.client.post(url)

        assert response.status_code == 400
        assert response.json() == {"error": "Did you forget to provide a png image of a digit?"}

    def test_predict_digit(self):
        """This test is to ensure that the endpoint
        returns a predicted number value based on the image provided."""
        url = reverse("predict-digit")

        # Load the image submitted by the user and POST.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, "test_image.png")
        with open(path, "rb") as image:
            response = self.client.post(url, {"imageFile": image})

        assert response.status_code == 200
        assert response.json() == {"predicted": [1]}
