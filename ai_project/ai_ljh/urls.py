from django.urls import path
from . import views

app_name = "ai_ljh";

urlpatterns = [
    path('divorce/ai_ljh/learning', views.learning, name='learning'),
    path('divorce/ai_ljh/decision_tree', views.decision_tree_view, name='decision_tree'),
    path('divorce/ai_ljh/knn', views.knn_view, name='knn'),
    path('divorce/ai_ljh/knn_test_params', views.knn_test_params, name='knn_test_params'),
    path('divorce/ai_ljh/knn_opt_prms_result', views.knn_opt_prms_result, name='knn_opt_prms_result'),
    path('divorce/ai_ljh/svm_scatter', views.svm_scatter, name='svm_scatter'),
    path('divorce/ai_ljh/svm', views.svm_view, name='svm'),
    path('divorce/ai_ljh/svm_cv', views.svm_cv, name='svm_cv'),
    path('divorce/ai_ljh/svm_test_params', views.svm_test_params, name='svm_test_params'),
    path('divorce/ai_ljh/svm_opt_prms_result', views.svm_opt_prms_result, name='svm_opt_prms_result'),
    path('divorce/ai_ljh/logistic_regression', views.logistic_regression_view, name='logistic_regression'),
    path('divorce/ai_ljh/voting', views.voting_view, name='voting'),
    path('divorce/ai_ljh/voting_test_params', views.voting_test_params, name='voting_test_params'),
    path('foods/ai_ljh/learning', views.foods_learning, name='foods_learning'),
    path('foods/ai_ljh/result', views.foods_result, name='foods_result'),
    path('foods/ai_ljh/select_best_model', views.select_best_model, name='select_best_model'),
    path('foods/ai_ljh/analyze', views.analyze_result, name='analyze_result'),
    path('developer-1/', views.developer_page_1, name='developer_page_1'),
    path('developer-2/', views.developer_page_2, name='developer_page_2'),
    path('developer-3/', views.developer_page_3, name='developer_page_3'),
    path('developer-4/', views.developer_page_4, name='developer_page_4'),
] 