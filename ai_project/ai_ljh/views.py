from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from main.views import korean
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import sklearn.svm as svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import io
from django.http import HttpResponse
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Create your views here.
def learning(request):
    df_divorce = pd.read_csv(r'C:\Users\User\OneDrive\바탕 화면\2차project\jihan_mr\machine_learing\ai_project\static\file\ai_data\ai_khm\divorce_data.csv')

    df_dvc_html = df_divorce.to_html(index=False, classes='table table-bordered')

    col_list = [
        "논의를 하다가 상태가 악화되었을 때, 우리 중 한 명이 사과하면 논의가 더 이상 악화되지 않는다",
        "가끔 상황이 어려워져도 우리의 차이점을 무시할 수 있다",
        "필요하면 처음부터 배우자의 논의를 받아들여 수정할 수 있다",
        "배우자와 함께하는 시간은 우리에게 특별하다",
        "우리는 파트너로서 집에서 보내는 시간이 없다",
        "우리는 같은 가정에서 사는 가족보다는 남에 가깝다",
        "나는 배우자와 휴가를 즐긴다",
        "나는 배우자와 여행을 즐긴다",
        "배우자와 나의 목표는 대부분 공통적이다",
        "미래가 되었을 때, 과거를 돌이켜보면 배우자와 나는 서로 화목하게 잘 지냈다는 것을 알 수 있다",
        "배우자와 나는 개인의 자유라는 측면에서 비슷한 가치관을 가지고 있다",
        "배우자와 나는 비슷한 연예 감각을 가진 편이다",
        "사람들(아이, 친구 등)에 대한 목표는 대부분 같다",
        "배우자와 나의 꿈은 비슷하고 화목하다",
        "우리는 사랑이 무엇이어야 하는지에 대해 배우자와 뜻이 같다",
        "우리는 내 배우자와의 인생에서 행복해지는 것에 대해 같은 견해를 가지고 있다",
        "배우자와 나는 결혼이 어떻게 되어야 하는지에 대해 비슷한 생각을 가지고 있다",
        "배우자와 나는 결혼에서 역할이 어떻게 되어야 하는지에 대해 비슷한 생각을 가지고 있다",
        "배우자와 나는 신뢰에 대한 가치관이 비슷하다",
        "배우자가 좋아하는 것을 정확히 알고 있다",
        "배우자가 아플 때 어떤 돌봄을 받고 싶은지 잘 안다",
        "배우자가 매우 좋아하는 음식을 알고 있다",
        "배우자가 살면서 어떤 스트레스를 받는지 말할 수 있다",
        "나는 배우자의 내면에 대해 알고 있다",
        "배우자의 기본적인 근심에 대해 알고 있다",
        "배우자의 현재 스트레스의 원인이 무엇인지 알고 있다",
        "배우자의 희망과 소원을 알고 있다",
        "나는 배우자에 대해 잘 알고 있다",
        "나는 배우자의 친구와 그들과의 사회적 관계에 대해 알고 있다",
        "나는 배우자와 말다툼을 하면 공격적으로 느껴진다",
        "배우자와 상의할 때 배우자의 성격에 대해 부정적인 진술을 할 수 있다",
        "나는 상의하는 동안 공격적인 표현을 할 수 있다",
        "배우자와의 논의는 차분하지 않다",
        "나는 배우자가 이야기를 꺼내는 방식이 싫다",
        "우리의 논의는 흔히 갑자기 발생한다",
        "내가 무슨 일이 일어났는지 알기 전에 논의가 시작된다",
        "배우자와 이야기를 나누다보면 갑작스럽게 침착함이 깨진다",
        "가끔은 내가 집을 떠나는 편이 좋다고 생각한다",
        "배우자와 상의하는 것보다 침묵을 지키는게 낫다",
        "배우자와 상의할 때 화를 다스리지 못 할 것 같아서 침묵한다",
        "나는 논의 하는 것이 옳다고 생각한다",
        "나는 내가 비난받는 것과는 아무 관련이 없다",
        "나는 내가 비난받는 것에 대해 죄책감을 느끼지 않는다",
        "집에 문제가 발생했을 때 내 잘못은 없다",
        "나는 배우자의 부족한 부분에 대해 주저하지 않고 이야기 할 수 있다",
        "나는 논의할 때 배우자의 부적절함을 상기시킨다",
        "나는 배우자에게 배우자의 무능함을 말하는 것이 두렵지 않다",
        "이혼 결과"
    ]

    df_data, df_target = divorce_data_preprocessing()
    
    content = {
        "df_dvc_html": df_dvc_html,
        "dvc_dict": dict(zip(df_divorce.columns.tolist(), col_list)),
        "data_info": df_data.info(),
        "data_head": df_data.head().to_html(index=False, classes='table table-bordered'),
        "data_desc": df_data.describe().to_html(classes='table table-bordered'),
        "data_isnull_sum": df_data.isnull().sum().to_list()
    }
   
    return render(request, 'divorce/ai_khm/learning.html', content)

# 이혼 확률 데이터 전처리
def divorce_data_preprocessing():
    df_divorce = pd.read_csv(r'C:\Users\User\OneDrive\바탕 화면\2차project\jihan_mr\machine_learing\ai_project\static\file\ai_data\ai_khm\divorce_data.csv')

    df_data = df_divorce.drop(["Divorce_Y_N"], axis=1)
    df_target = df_divorce["Divorce_Y_N"]

    return df_data, df_target

# 최소값, 최대값, 증가값에서 증가값의 형에 따라 목록 값을 반환
def return_list_type_of_add_val(prm_min_val, prm_max_val, prm_add_val, prm_point):
    if prm_add_val.find(".") != -1:
        min_val = float(prm_min_val)
        max_val = float(prm_max_val)
        add_val = float(prm_add_val)

        point = int(prm_point)       # 소수점 아래 반올림 수

        var = min_val

        prm_list = []

        while var <= max_val:
            var = round(var, point)
            print("변수 값: ", var)

            prm_list.append(var)

            var = var + add_val
    else:
        min_val = int(prm_min_val)
        max_val = int(prm_max_val)
        add_val = int(prm_add_val)

        prm_list = range(min_val, max_val, add_val)

    return prm_list

# 의사결정트리 모델 
def decision_tree_view(request):
    df_data, df_target = divorce_data_preprocessing()

    X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.3, random_state=62)
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)

    pred = dt_clf.predict(X_test)

    content = {
        "accuracy_score": accuracy_score(y_test, pred),
        "hyper_param": dt_clf.get_params()
    }

    return render(request, 'divorce/ai_khm/decision_tree.html', content)

# knn 최적의 파라미터 찾는 메서드
def get_knn_best_params(n_neighbor_list, X_train, y_train):
    params = {
        'n_neighbors': n_neighbor_list,
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    grid_cv = GridSearchCV(KNeighborsClassifier(), param_grid=params, scoring='accuracy', cv=5)
    grid_cv.fit(X_train, y_train)

    return grid_cv.best_score_, grid_cv.best_params_

# svm 최적의 파라미터 찾는 메서드(선형 모델 분류)
def get_svm_linear_best_params(cost_list, clf, X_train, y_train):
    params = {
        'C': cost_list
    }

    grid_cv = GridSearchCV(clf, param_grid=params, scoring='accuracy', cv=5, verbose=1)
    grid_cv.fit(X_train, y_train)

    return grid_cv.best_score_, grid_cv.best_params_

# voting 최적의 파라미터 찾는 메서드(의사결정트리, knn 모델, 로지스틱 회귀)
def get_voting_best_params(cost_list, n_neighbors_list, n_depth_list, n_split_list, vo, X_train, y_train):
    params = {
        'lr__C': cost_list,
        'knn__n_neighbors': n_neighbors_list,
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
        'dt__max_depth': n_depth_list,
        'dt__min_samples_split': n_split_list
    }

    grid_cv = GridSearchCV(vo, param_grid=params, scoring='accuracy', cv=5)
    grid_cv.fit(X_train, y_train)

    return grid_cv.best_score_, grid_cv.best_params_

# knn 모델
def knn_view(request):
    df_data, df_target = divorce_data_preprocessing()

    X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2, random_state=10)

    global knn_X_train, knn_X_test, knn_y_train, knn_y_test
    # 다른 메서드에서 사용을 위해 전역변수에 대입
    knn_X_train = X_train
    knn_X_test = X_test
    knn_y_train = y_train
    knn_y_test = y_test

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    knn_best_score, knn_best_params = get_knn_best_params(range(1, 100, 10), X_train, y_train)

    content = {
        "accuracy_score": accuracy_score(y_test, pred),
        "knn_best_score": knn_best_score,
        "knn_best_params": knn_best_params
    }

    return render(request, 'divorce/ai_khm/knn.html', content)

# 공통 그래프 셋팅
def comm_graph_setting():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return graph

# knn 정확도 그래프
def knn_accuracy_graph(knn_min_val, knn_max_val):
    korean()        # 한글 셋팅

    training_accuracy = []
    test_accuracy = []
    n_neighbors_settings = range(knn_min_val, knn_max_val)

    for n_neighbor in n_neighbors_settings:
        knn = KNeighborsClassifier(n_neighbors=n_neighbor)
        knn.fit(knn_X_train, knn_y_train)
        training_accuracy.append(knn.score(knn_X_train, knn_y_train))
        test_accuracy.append(knn.score(knn_X_test, knn_y_test))

    plt.switch_backend('AGG')
    plt.plot(n_neighbors_settings, training_accuracy, label='훈련 정확도')
    plt.plot(n_neighbors_settings, test_accuracy, label='검증 정확도')
    plt.ylabel('정확도')
    plt.xlabel('n_neighbors')
    plt.legend()

    graph = comm_graph_setting()

    return graph

# knn 최고의 파라미터 테스트
def knn_test_params(request):
    if request.method == "GET":
        return render(request, 'divorce/ai_khm/knn_test_params.html')
    elif request.method == "POST":
        knn_min_val = int(request.POST['knnMinVal'])
        knn_max_val = int(request.POST['knnMaxVal'])
        knn_add_val = int(request.POST['knnAddVal'])

        knn_best_score, knn_best_params = get_knn_best_params(range(knn_min_val, knn_max_val, knn_add_val), knn_X_train, knn_y_train)

        content = {
            "knn_best_score": knn_best_score,
            "knn_best_params": knn_best_params,
            "graph": knn_accuracy_graph(knn_min_val, knn_max_val)
        }

        return render(request, 'divorce/ai_khm/knn_test_params.html', content)

# knn 최적화된 파라미터 입력 후 결과
def knn_opt_prms_result(request):
    if request.method == "GET":
        return render(request, 'divorce/ai_khm/knn_opt_prms_result.html')
    elif request.method == "POST":
        knn_metric = request.POST['knnMetric']
        knn_neightbors = int(request.POST['knnNeighbors'])
        knn_weights = request.POST['knnWeights']

        knn = KNeighborsClassifier(metric=knn_metric, n_neighbors=knn_neightbors, weights=knn_weights)
        knn.fit(knn_X_train, knn_y_train)

        knn_opt_score = round(knn.score(knn_X_test, knn_y_test))
        
        content = {
            "knn_opt_score": knn_opt_score
        }

        return render(request, 'divorce/ai_khm/knn_opt_prms_result.html', content)

# svm 산포도
def svm_scatter(request):
    df_data, df_target = divorce_data_preprocessing()

    plt.switch_backend('AGG')
    plt.scatter(df_data["Sorry_end"], df_data["Ignore_diff"], c=df_target)

    graph = comm_graph_setting()

    content = {
        "svm_scatter": graph
    }

    return render(request, 'divorce/ai_khm/svm_scatter.html', content)

# svm 모델
def svm_view(request):
    df_data, df_target = divorce_data_preprocessing()

    X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.3, random_state=62)

    global svm_linear_clf, svm_rbf_clf, svm_data, svm_labels, svm_X_train, svm_X_test, svm_y_train, svm_y_test
    # 다른 메서드에서 사용을 위해 전역변수에 대입
    svm_data = df_data
    svm_labels = df_target
    svm_X_train = X_train
    svm_X_test = X_test
    svm_y_train = y_train
    svm_y_test = y_test
    svm_linear_clf = svm.SVC(kernel='linear')
    svm_linear_clf.fit(X_train, y_train)

    svm_rbf_clf = svm.SVC(kernel='rbf')
    svm_rbf_clf.fit(X_train, y_train)

    content = {
        "svm_linear_score": svm_linear_clf.score(X_test, y_test),
        "svm_rbf_score": svm_rbf_clf.score(X_test, y_test)
    }

    return render(request, 'divorce/ai_khm/svm.html', content)

# svm 교차 검증
def svm_cv(request):
    scores = cross_val_score(svm_linear_clf, svm_data, svm_labels, cv=5)
    df_svm_cv = pd.DataFrame(cross_validate(svm_linear_clf, svm_data, svm_labels, cv=5))
    df_svm_cv_html = df_svm_cv.to_html(classes='table table-bordered')
    
    svm_best_score, svm_best_params = get_svm_linear_best_params([0.001, 0.01, 0.1, 1, 10, 25, 50, 100], svm_linear_clf, svm_X_train, svm_y_train)

    content = {
        "svm_cv_mean": scores.mean(),
        "df_svm_cv_html": df_svm_cv_html,
        "svm_best_score": svm_best_score,
        "svm_best_params": svm_best_params
    }

    return render(request, 'divorce/ai_khm/svm_cv.html', content)

def svm_test_params(request):
    if request.method == "GET":
        return render(request, 'divorce/ai_khm/svm_test_params.html')
    elif request.method == "POST":
        # 증가 값에 소수점 포함 시 최소 값, 최대 값, 증가 값 모두 실수화, 아니면 정수화
        cost_list = return_list_type_of_add_val(request.POST['svmMinVal'], request.POST['svmMaxVal'], request.POST['svmAddVal'], request.POST['svmPoint'])

        svm_best_score, svm_best_params = get_svm_linear_best_params(cost_list, svm_linear_clf, svm_X_train, svm_y_train)

        content = {
            "svm_best_score": svm_best_score,
            "svm_best_params": svm_best_params
        }

        return render(request, 'divorce/ai_khm/svm_test_params.html', content)
    
# SVM 최적화된 파라미터 입력 후 결과
def svm_opt_prms_result(request):
    if request.method == "GET":
        return render(request, 'divorce/ai_khm/svm_opt_prms_result.html')
    elif request.method == "POST":
        # C 값에 소수점 포함 시 실수화, 아니면 정수화
        str_svm_c_val = request.POST['svmC']
        svm_kernel_val = request.POST['svmKernel']

        if str_svm_c_val.find(".") != -1:
            svm_c_val = float(str_svm_c_val)
        else:
            svm_c_val = int(str_svm_c_val)

        svm_clf = svm.SVC(kernel=svm_kernel_val, C=svm_c_val)

        svm_clf.fit(svm_X_train, svm_y_train)
        
        content = {
            "svm_opt_score": svm_clf.score(svm_X_test, svm_y_test)
        }

        return render(request, 'divorce/ai_khm/svm_opt_prms_result.html', content)
    
# 로지스틱 회귀 모델
def logistic_regression_view(request):
    df_data, df_target = divorce_data_preprocessing()

    standard_scaled = StandardScaler().fit_transform(df_data)

    X_train, X_test, y_train, y_test = train_test_split(standard_scaled, df_target, test_size=0.3, random_state=10)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)

    df_lr_coef = pd.DataFrame(lr.coef_, columns=df_data.columns)

    content = {
        "lr_score": lr.score(X_test, y_test),
        "lr_acc_score": accuracy_score(y_test, pred),
        "lr_roc_auc_score": roc_auc_score(y_test, pred),
        "df_lr_coef": df_lr_coef.to_html(index=False, classes="table table-bordered")
    }

    return render(request, 'divorce/ai_khm/logistic_regression.html', content)

# 보팅 분류기
def voting_view(request):
    df_data, df_target = divorce_data_preprocessing()

    X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2, random_state=10)

    lr = LogisticRegression(max_iter=4000)
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier()

    soft_vo = VotingClassifier([('lr', lr), ('knn', knn), ('dt', dt)], voting='soft')
    soft_vo.fit(X_train, y_train)
    soft_pred = soft_vo.predict(X_test)

    hard_vo = VotingClassifier([('lr', lr), ('knn', knn), ('dt', dt)], voting='hard')
    hard_vo.fit(X_train, y_train)
    hard_pred = hard_vo.predict(X_test)

    model_acc_list = []

    for model in [lr, knn, dt]:
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)
        model_pred = model.predict(X_test)

        acc = accuracy_score(y_test, model_pred)

        model_acc_list.append(f'{model_name} 의 정확도: {acc}')

    standard_scaler = StandardScaler()
    standard_scaler.fit(X_train)
    X_train_std = standard_scaler.transform(X_train)
    X_test_std = standard_scaler.transform(X_test)

    global voting_soft_vo, voting_X_train_std, voting_X_test_std, voting_y_train, voting_y_test
    # 다른 메서드에서 사용을 위해 전역변수에 대입
    voting_soft_vo = soft_vo
    voting_X_train_std = X_train_std
    voting_X_test_std = X_test_std
    voting_y_train = y_train
    voting_y_test = y_test

    soft_vo.fit(X_train_std, y_train)
    standard_pred = soft_vo.predict(X_test_std)

    voting_cv_list = []

    for model in [soft_vo, lr, knn, dt]:
        scores = cross_val_score(model, np.concatenate((X_train_std, X_test_std)), np.concatenate((y_train, y_test)), scoring='accuracy', cv=5)

        voting_cv_list.append(f'모델: {model.__class__.__name__}')
        voting_cv_list.append('전체 정확도')
        voting_cv_list.append(scores)
        voting_cv_list.append(f'평균 정확도: {np.mean(scores)}')
        voting_cv_list.append('=' * 30)

    cost_list = [0.001, 0.01, 0.1, 1, 10]
    n_neighbors_list = range(1, 100, 10)
    n_depth_list = range(1, 21, 10)
    n_split_list = range(2, 50, 10)

    voting_best_score, voting_best_params = get_voting_best_params(cost_list, n_neighbors_list, n_depth_list, n_split_list, soft_vo, X_train_std, y_train)

    content = {
        "soft_voting_acc_score": accuracy_score(y_test, soft_pred),
        "hard_voting_acc_score": accuracy_score(y_test, hard_pred),
        "standard_soft_voting_acc_score": accuracy_score(y_test, standard_pred),
        "model_acc_list": model_acc_list,
        "voting_cv_list": voting_cv_list,
        "voting_best_score": voting_best_score,
        "voting_best_params": voting_best_params
    }

    return render(request, 'divorce/ai_khm/voting.html', content)

# 보팅 최적의 파라미터
def voting_test_params(request):
    if request.method == "GET":
        return render(request, 'divorce/ai_khm/voting_test_params.html')
    elif request.method == "POST":
        # 증가 값에 소수점 포함 시 최소 값, 최대 값, 증가 값 모두 실수화, 아니면 정수화
        cost_list = return_list_type_of_add_val(request.POST['votingCMinVal'], request.POST['votingCMaxVal'], request.POST['votingCAddVal'], request.POST['votingCPoint'])
        n_neighbors_list = return_list_type_of_add_val(request.POST['votingNeighborsMinVal'], request.POST['votingNeighborsMaxVal'], request.POST['votingNeighborsAddVal'], request.POST['votingNeighborsPoint'])
        n_depth_list = return_list_type_of_add_val(request.POST['votingMaxDepthMinVal'], request.POST['votingMaxDepthMaxVal'], request.POST['votingMaxDepthAddVal'], request.POST['votingMaxDepthPoint'])
        n_split_list = return_list_type_of_add_val(request.POST['votingMinSamplesSplMinVal'], request.POST['votingMinSamplesSplMaxVal'], request.POST['votingMinSamplesSplAddVal'], request.POST['votingMinSamplesSplPoint'])

        voting_best_score, voting_best_params = get_voting_best_params(cost_list, n_neighbors_list, n_depth_list, n_split_list, voting_soft_vo, voting_X_train_std, voting_y_train)

        content = {
            "voting_best_score": voting_best_score,
            "voting_best_params": voting_best_params
        }

        return render(request, 'divorce/ai_khm/voting_test_params.html', content)
    
    
    

# *******************************************************************************************************************************************************************************

# 음식 데이터 전처리
def food_data_preprocessing():
    df_raw_foods = pd.read_excel(r'C:\Users\User\OneDrive\바탕 화면\2차project\jihan_mr\machine_learing\ai_project\static\file\fd_ljh\food_final.xlsx')

    df_foods = df_raw_foods.drop(columns=['menu'])
    df_labels = df_raw_foods['menu']

    return df_foods, df_labels
    
# 음식 추천 시스템
def foods_learning(request):
    return render(request, 'foods/ai_ljh/learning.html')

# 최적 모델 선택 함수
def select_best_model(df_foods, df_labels):
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(df_foods, df_labels, test_size=0.2, random_state=42)

    # KNN 파라미터 설정 및 GridSearch
    knn_params = {'n_neighbors': range(1, 10)}
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring='accuracy', n_jobs=-1)
    knn_grid.fit(X_train, y_train)

    # 랜덤 포레스트 파라미터 설정 및 GridSearch
    rf_params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    # 각 모델의 최적 파라미터와 성능 비교
    knn_best = knn_grid.best_estimator_
    rf_best = rf_grid.best_estimator_

    knn_pred = knn_best.predict(X_test)
    rf_pred = rf_best.predict(X_test)

    knn_accuracy = accuracy_score(y_test, knn_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    # if knn_accuracy > rf_accuracy:
    #     return knn_best, knn_accuracy
    # else:
    #     return rf_best, rf_accuracy
    return knn_best, knn_accuracy, rf_best, rf_accuracy

# 그래프 생성 함수
def analyze_graph(knn_accuracy, rf_accuracy):
    plt.figure(figsize=(8, 6))
    models = ['KNN', 'Random Forest']
    accuracies = [knn_accuracy, rf_accuracy]

    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')

    # 그래프를 이미지로 변환
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # 이미지를 base64로 인코딩하여 전달
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic

# 분석 결과를 보여줄 함수
def analyze_result(request, knn_accuracy=0.8, rf_accuracy=0.85):
    graphic = analyze_graph(knn_accuracy, rf_accuracy)

    # 결과 설명 텍스트 생성
    result_description = f"KNN 모델의 정확도는 {round(knn_accuracy * 100, 2)}%이고, "\
                         f"랜덤 포레스트 모델의 정확도는 {round(rf_accuracy * 100, 2)}%입니다."

    content = {
        "graphic": graphic,
        "result_description": result_description
    }

    return render(request, 'foods/ai_ljh/analyze.html', content)

# 음식 결과 시스템
def foods_result(request):
    food_dict = {
        1: "냉면(물, 비빔, 평양, 함흥 등)",
        2: "국수(잔치, 비빔, 쌀)",
        3: "국밥(설렁탕, 육개장, 수육 등)",
        4: "덮밥(제육덮밥, 불고기덮밥 등)",
        5: "볶음류(오징어볶음, 김치볶음밥, 철판볶음 등)",
        6: "찌개(김치찌개, 된장찌개, 순두부찌개 등)",
        7: "삼계탕",
        8: "비빔밥(돌솥비빔밥, 산채비빔밥 등)",
        9: "감자탕 또는 뼈해장국",
        10: "마라탕",
        11: "짜장 또는 짬뽕",
        12: "초밥",
        13: "우동",
        14: "떡볶이 또는 순대 또는 튀김",
        15: "라면 또는 김밥",
        16: "돈까스",
        17: "샌드위치 또는 샐러드",
        18: "햄버거",
        19: "피자",
        20: "스테이크 또는 파스타(스파게티)",
        21: "컵밥 또는 도시락",
        22: "삼겹살 또는 소고기 또는 양꼬치",
        23: "치킨",
        24: "족발",
        25: "회(물회) 또는 생선구이(고등어구이, 임연수 구이 등)",
        26: "뷔페(점심뷔페 등) 또는 백반"
    }

    food_img_list = [
        "naengmyeon.jpg",
        "noodles.jpg",
        "gukbap.jpg",
        "rice.jpg",
        "stir-fry.jpg",
        "stew.png",
        "samgyetang.jpg",
        "bibimbap.jpg",
        "back-bone-stew.jpg",
        "malatang.jpg",
        "black-bean-sauce-noodles.jpg",
        "sushi.jpg",
        "udon-noodles.jpg",
        "tteokbokki.jpg",
        "ramen.jpg",
        "pork-cutlet.jpg",
        "sandwich.jpg",
        "burger.jpg",
        "pizza.jpg",
        "steak.jpg",
        "korean-lunch-box.jpg",
        "pork-belly.jpg",
        "chicken.jpg",
        "pork-feet.jpg",
        "raw-fish.jpg",
        "buffet.jpg"
    ]
    
    df_foods, df_labels = food_data_preprocessing()

    # 최적 모델과 정확도 선택
    # best_model, best_accuracy = select_best_model(df_foods, df_labels)
    knn_best, knn_accuracy, rf_best, rf_accuracy = select_best_model(df_foods, df_labels)

    # knn = KNeighborsClassifier(n_neighbors=3)
    # knn.fit(df_foods, df_labels)

    # X_test: 사용자 입력값
    X_test = pd.DataFrame({
        'emotion': [request.POST['emotion']],
        'season': [request.POST['season']],
        'weather': [request.POST['weather']],
        'people': [request.POST['people']],
        'price': [request.POST['price']],
        'time': [request.POST['time']],
        'sex': [request.POST['sex']]
    })
    # 최적 모델로 예측 수행
    best_model = rf_best if rf_accuracy > knn_accuracy else knn_best
    best_accuracy = max(rf_accuracy, knn_accuracy)
    pred = best_model.predict(X_test)
    
    # pred = best_model.predict(X_test)
    # pred = knn.predict(X_test)
    # 정확도 비교 그래프 생성
    graphic = analyze_graph(knn_accuracy, rf_accuracy)

    # 결과 설명 텍스트 생성
    result_description = f"당신의 선택에 따라 추천된 음식은 '{food_dict.get(pred[0])}'입니다. "\
                         f"이 추천은 {round(best_accuracy * 100, 2)}%의 정확도를 가진 최적화된 모델에 의해 만들어졌습니다."
    content = {
        "food_result": food_dict.get(pred[0]),
        "food_picture": food_img_list[pred[0] - 1],
        "accuracy": best_accuracy,
        "graphic": graphic,
        "result_description": result_description
    }

    return render(request, 'foods/ai_ljh/result.html', content)



def developer_page_1(request):
    return render(request, 'dev/developer_1.html')

def developer_page_2(request):
    return render(request, 'dev/developer_2.html')

def developer_page_3(request):
    return render(request, 'dev/developer_3.html')

def developer_page_4(request):
    return render(request, 'dev/developer_4.html')