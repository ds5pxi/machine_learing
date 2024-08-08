from django.urls import path
from . import views

app_name = "ai_jjw";

# 서브 앱 url 등록(include)
urlpatterns = [
    path('divorce/ai_jjw/learning', views.learning, name='learning'),
    path('foods/ai_jjw/learning', views.foods_learning, name='foods_learning'),
]
