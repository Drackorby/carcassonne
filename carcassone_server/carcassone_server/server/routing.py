from django.urls import re_path

from .consumers import ModelConsumer

websocket_urlpatterns = [
    re_path(r"^action/(?P<id>\w+)/$", ModelConsumer.as_asgi()),
]