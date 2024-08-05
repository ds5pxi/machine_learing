from django.urls import path
from . import views

app_name = "ai_ljh";

urlpatterns = [
    path('divorce/ai_ljh/learning', views.learning, name='learning'),
    path('foods/ai_ljh/learning', views.foods_learning, name='foods_learning'),
]