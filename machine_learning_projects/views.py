from django.http import HttpResponse, JsonResponse
from dotenv import load_dotenv
from fastai.vision.widgets import *
from fastbook import *

load_dotenv()

bear_types = 'grizzly', 'black', 'teddy'
path = Path('bears')


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

    if not path.exists():
        path.mkdir()
        for bear_type in bear_types:
            dest = (path / bear_type)
            dest.mkdir(exist_ok=True)
            results = search_images_bing(key, f'{bear_type} bear')
            download_images(dest, urls=results.attrgot('contentUrl'))

    image_files = get_image_files(path)
    failed = verify_images(image_files)

    # Remove failed images
    if failed:
        failed.map(Path.unlink)

    response = {
        "image_count": image_count,
        "success": True,
    }
    return JsonResponse(response)
