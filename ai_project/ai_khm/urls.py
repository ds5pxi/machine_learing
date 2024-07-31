from django.urls import path
from . import views

app_name = "ai_khm";

urlpatterns = [
    path('divorce/ai_khm/learning', views.learning, name='learning'),
    path('divorce/ai_khm/decision_tree', views.decision_tree_view, name='decision_tree'),
    path('divorce/ai_khm/knn', views.knn_view, name='knn'),
    path('divorce/ai_khm/knn_test_params', views.knn_test_params, name='knn_test_params'),
    path('divorce/ai_khm/knn_opt_prms_result', views.knn_opt_prms_result, name='knn_opt_prms_result'),
    path('divorce/ai_khm/svm', views.svm_view, name='svm'),
]