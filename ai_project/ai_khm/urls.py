from django.urls import path
from . import views

app_name = "ai_khm";

urlpatterns = [
    path('divorce/ai_khm/learning', views.learning, name='learning'),
    path('divorce/ai_khm/decision_tree', views.decision_tree_view, name='decision_tree'),
    path('divorce/ai_khm/svm', views.svm_view, name='svm'),
]