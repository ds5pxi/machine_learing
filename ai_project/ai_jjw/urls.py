from django.urls import path
from . import views

app_name = "ai_jjw";

# 서브 앱 url 등록(include)
urlpatterns = [
    path('divorce/ai_jjw/learning', views.learning, name='learning'),
    path('divorce/ai_jjw/analyze_divorce', views.analyze_divorce, name='analyze_divorce'),
    path('divorce/ai_jjw/decision_tree', views.decision_tree_view, name='decision_tree'),
    path('divorce/ai_jjw/svm', views.svm_view, name='svm'),
]
