from django.http import HttpResponse


def index(request):
    return HttpResponse("This is Cal's machine learning API. It will host a bear classification app.")
