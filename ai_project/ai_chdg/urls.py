from django.urls import path
from . import views

app_name = "ai_chdg"

urlpatterns = [
    path('survey/', views.survey, name='survey'),
    path('result/', views.survey, name='result'),
    path('result/', views.result, name='result'),
]