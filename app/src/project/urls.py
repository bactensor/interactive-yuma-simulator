from django.conf import settings
from django.contrib.admin.sites import site
from django.urls import include, path
from django.views.generic import RedirectView

from .core.consumers import DefaultConsumer
from .core.views import simulate_single_case_view, simulation_view

urlpatterns = [
    path("admin/", site.urls),
    path("", include("django.contrib.auth.urls")),
    path("", RedirectView.as_view(url="/simulator/", permanent=False)),
    path("simulator/", simulation_view, name="simulation_view"),
    path("simulate_single_case/", simulate_single_case_view, name="simulate_single_case"),
]
ws_urlpatterns = [
    path("ws/v0/", DefaultConsumer.as_asgi()),
]

if settings.DEBUG_TOOLBAR:
    urlpatterns += [
        path("__debug__/", include("debug_toolbar.urls")),
    ]
