from django.urls import path
from . import views

urlpatterns = [
    path('', views.main),
    path('foods/main/survey', views.foods_survey, name='foods_survey'),
    path('foods_ml', views.foods_ml, name='foods_ml'),
    path('foods_result', views.foods_result, name='foods_result'),
]
