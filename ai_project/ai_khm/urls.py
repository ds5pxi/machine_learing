from django.urls import path
from . import views

app_name = "ai_khm";

urlpatterns = [
    path('divorce/ai_khm/learning', views.learning, name='learning'),
]