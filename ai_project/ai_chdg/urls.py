from django.urls import path
from . import views

app_name = "ai_chdg";

urlpatterns = [
    path('divorce/ai_chdg/learning', views.learning, name='learning'),
]