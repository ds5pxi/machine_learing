from django.urls import path
from . import views

app_name = "ai_chdg";

urlpatterns = [
    path('divorce/ai_chdg/learning', views.learning, name='learning'),
    path('survey/', views.survey, name='survey'),
    path('result/', views.survey, name='result'),
    path('result/', views.result, name='result'),
    path('foods/ai_ljh/learning', views.foods_learning, name='foods_learning'),
]