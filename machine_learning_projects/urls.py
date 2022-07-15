from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("health/", views.health, name="health"),
    path("which-bear/", views.which_bear, name="which-bear"),
    path("get-bear-images/", views.get_bear_images, name="get-bear-images"),
]
