from django.urls import path
from . import views

app_name = "ai_khm";

urlpatterns = [
    path('foods/ai_khm/survey', views.foods_survey, name='foods_survey'),
    path('foods/ai_khm/result', views.foods_result, name='foods_result'),
]