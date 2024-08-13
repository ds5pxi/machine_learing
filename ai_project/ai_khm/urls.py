from django.urls import path
from . import views

app_name = "ai_khm";

urlpatterns = [
    path('divorce/ai_khm/learning', views.learning, name='learning'),
    path('divorce/ai_khm/decision_tree', views.decision_tree_view, name='decision_tree'),
    path('divorce/ai_khm/knn', views.knn_view, name='knn'),
    path('divorce/ai_khm/knn_test_params', views.knn_test_params, name='knn_test_params'),
    path('divorce/ai_khm/knn_opt_prms_result', views.knn_opt_prms_result, name='knn_opt_prms_result'),
    path('divorce/ai_khm/svm_scatter', views.svm_scatter, name='svm_scatter'),
    path('divorce/ai_khm/svm', views.svm_view, name='svm'),
    path('divorce/ai_khm/svm_cv', views.svm_cv, name='svm_cv'),
    path('divorce/ai_khm/svm_test_params', views.svm_test_params, name='svm_test_params'),
    path('divorce/ai_khm/svm_opt_prms_result', views.svm_opt_prms_result, name='svm_opt_prms_result'),
    path('divorce/ai_khm/logistic_regression', views.logistic_regression_view, name='logistic_regression'),
    path('divorce/ai_khm/voting', views.voting_view, name='voting'),
    path('divorce/ai_khm/voting_test_params', views.voting_test_params, name='voting_test_params'),
    path('foods/ai_khm/survey', views.foods_survey, name='foods_survey'),
    path('foods/ai_khm/result', views.foods_result, name='foods_result'),
]