from django.urls import path
from . import views

app_name = "ai_chdg";

urlpatterns = [
    path('divorce/ai_chdg/learning', views.learning, name='learning'),
    path('foods/ai_chdg/learning', views.foods_learning, name='foods_learning'),
]