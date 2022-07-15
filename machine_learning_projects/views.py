from django.http import HttpResponse
from dotenv import load_dotenv
from fastai.vision.widgets import *
from fastbook import *

load_dotenv()


def index(request):
    return HttpResponse("This is Cal's machine learning API. It will host a bear classification app.")


def health(request):
    return HttpResponse("OK")


def which_bear(request):
    return HttpResponse("BEARS")


def get_bear_images(request):
    key = os.getenv("AZURE_SEARCH_KEY")

    # Create a new Azure Search client
    results = search_images_bing(key, "grizzly bear")
    ims = results.attrgot("contentUrl")
    image_count = len(ims)
    return HttpResponse(f"Images Indexed {image_count}")
