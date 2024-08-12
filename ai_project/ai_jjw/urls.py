from django.urls import path
from . import views

app_name = "ai_jjw";

# 서브 앱 url 등록(include)
urlpatterns = [
    path('divorce/ai_jjw/learning', views.learning, name='learning'), 
    path('divorce/ai_jjw/divorce_data_preprocessing', views.divorce_data_preprocessing, name='divorce_data_preprocessing'),
    path('foods/ai_jjw/foods_learning', views.foods_learning, name='foods_learning'),
    path('foods/ai_jjw/foods_result', views.foods_result, name='foods_result'),
]
