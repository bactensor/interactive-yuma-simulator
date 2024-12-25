from django.conf import settings
from django.contrib.admin.sites import site
from django.urls import include, path

from .core.consumers import DefaultConsumer
from .core.views import simulation_view

urlpatterns = [
    path("admin/", site.urls),
    path("", include("django.contrib.auth.urls")),
    path("simulator/", simulation_view, name="simulation_view"),
]
ws_urlpatterns = [
    path("ws/v0/", DefaultConsumer.as_asgi()),
]

if settings.DEBUG_TOOLBAR:
    urlpatterns += [
        path("__debug__/", include("debug_toolbar.urls")),
    ]
