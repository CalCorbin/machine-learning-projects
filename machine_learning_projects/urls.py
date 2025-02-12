"""Module for managing URLs."""
from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("health/", views.health, name="health"),
    path("predict-digit/", views.predict_digit, name="predict-digit"),
]
