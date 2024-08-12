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

# Create your views here.
def learning(request):
    df_divorce = pd.read_csv('D:/machine_learing/ai_project/static/file/ai_data/ai_khm/divorce_merge.csv')

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
    df_divorce = pd.read_csv('D:/machine_learing/ai_project/static/file/ai_data/ai_khm/divorce_merge.csv')

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
    
# 음식 데이터 전처리
def food_data_preprocessing():
    df_raw_foods = pd.read_excel('D:/machine_learing/ai_project/static/file/ai_data/foods/main/food_data.xlsx')

    df_foods = df_raw_foods.drop(columns=['menu'])
    df_labels = df_raw_foods['menu']

    return df_foods, df_labels
    
# 음식 추천 시스템
def foods_learning(request):
    return render(request, 'foods/ai_khm/learning.html')

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

    eatery_dict = {
        1: {
            "name1": "천수냉면",
            "address1": "서울 동작구 만양로14길 22 1층",
            "lat1": 37.5117410302136,
            "lng1": 126.945546512266,
            "name2": "부자왕만두냉면",
            "address2": "서울 동작구 노량진로 110 1층",
            "lat2": 37.5131853341731,
            "lng2": 126.938038724901,
            "name3": "상도동함흥면옥",
            "address3": "서울 동작구 상도로 219 1층",
            "lat3": 37.5054269376064,
            "lng3": 126.943183283281
        },
        2: {
            "name1": "노량진국수",
            "address1": "서울 동작구 노량진로8길 16 1층",
            "lat1": 37.5127128551461,
            "lng1": 126.936191144207,
            "name2": "백가네 해물칼국수",
            "address2": "서울 동작구 노량진로8길 28 1층",
            "lat2": 37.5125801620593,
            "lng2": 126.936908252474,
            "name3": "산목",
            "address3": "서울 동작구 노량진로14길 25 1층",
            "lat3": 37.5125600836707,
            "lng3": 126.942588015381
        },
        3: {
            "name1": "명백집",
            "address1": "서울 동작구 노량진로8길 3 경성빌딩 1층 102호",
            "lat1": 37.5129296184414,
            "lng1": 126.93649160368,
            "name2": "한국밥",
            "address2": "서울 동작구 만양로 105 1층",
            "lat2": 37.5128549427149,
            "lng2": 126.944007081257,
            "name3": "명인설렁탕",
            "address3": "서울 동작구 노량진로8길 43 1층",
            "lat3": 37.5124842425616,
            "lng3": 126.937828470355
        },
        4: {
            "name1": "규동집",
            "address1": "서울 동작구 만양로14가길 4 1층",
            "lat1": 37.511783275723,
            "lng1": 126.945326374606,
            "name2": "행복은간장밥",
            "address2": "서울 동작구 노량진로16길 25 1층",
            "lat2": 37.5125442726531,
            "lng2": 126.943998943313,
            "name3": "텐카이치",
            "address3": "서울 동작구 만양로14길 24 1층",
            "lat3": 37.511870091749,
            "lng3": 126.945628421452
        },
        5: {
            "name1": "싸다김밥",
            "address1": "서울 동작구 노량진로 144 1층",
            "lat1": 37.5134256845674,
            "lng1": 126.941864127879,
            "name2": "양철북",
            "address2": "서울 동작구 만양로 84 삼익주상복합아파트 제지하1층 24호",
            "lat2": 37.5113671453073,
            "lng2": 126.945225673807,
            "name3": "본포",
            "address3": "서울 동작구 노량진로 110 1층",
            "lat3": 37.5131853341731,
            "lng3": 126.938038724901
        },
        6: {
            "name1": "부대통령뚝배기",
            "address1": "서울 동작구 만양로 100",
            "lat1": 37.5123857060582,
            "lng1": 126.944401840416,
            "name2": "양푼집",
            "address2": "서울 동작구 노량진로 140 상가 1층 114호",
            "lat2": 37.512756878817,
            "lng2": 126.941700638806,
            "name3": "신촌찌개집",
            "address3": "서울 동작구 만양로18길 18 1층",
            "lat3": 37.5129961696558,
            "lng3": 126.94524472988
        },
        7: {
            "name1": "토속골",
            "address1": "서울 동작구 노량진로8길 48",
            "lat1": 37.5122513350095,
            "lng1": 126.937832283006,
            "name2": "정동진",
            "address2": "서울 동작구 만양로 85",
            "lat2": 37.5110305766853,
            "lng2": 126.944361448704,
            "name3": "약초마을",
            "address3": "서울 동작구 노량진로 26 1층 102호",
            "lat3": 37.5129529701191,
            "lng3": 126.928927523118
        },
        8: {
            "name1": "델리 2호점",
            "address1": "서울 동작구 장승배기로28길 39",
            "lat1": 37.5124606379534,
            "lng1": 126.942454510529,
            "name2": "요기어때",
            "address2": "서울 동작구 노량진로16길 29 1층",
            "lat2": 37.5123170043047,
            "lng2": 126.944115049011,
            "name3": "마당분식",
            "address3": "서울 동작구 만양로14가길 19",
            "lat3": 37.5124439018104,
            "lng3": 126.944959534673
        },
        9: {
            "name1": "우정식당",
            "address1": "서울 동작구 노량진로8길 39 1층",
            "lat1": 37.5125948085533,
            "lng1": 126.937511222115,
            "name2": "참한감자탕",
            "address2": "서울 동작구 만양로14다길 4 1층",
            "lat2": 37.5115107543748,
            "lng2": 126.945201929505,
            "name3": "청석골감자탕",
            "address3": "서울 동작구 노량진로16길 24",
            "lat3": 37.5125411192118,
            "lng3": 126.943809375778
        },
        10: {
            "name1": "마라홀릭마라탕",
            "address1": "서울 동작구 노량진로14가길 6 1층",
            "lat1": 37.5124337852074,
            "lng1": 126.942819758054,
            "name2": "탕화쿵푸",
            "address2": "서울 동작구 만양로14가길 16 1층",
            "lat2": 37.5122326434719,
            "lng2": 126.945211694569,
            "name3": "딘딘향",
            "address3": "서울 동작구 노량진로14가길 10 1층",
            "lat3": 37.5125069840064,
            "lng3": 126.943083471175
        },
        11: {
            "name1": "국풍",
            "address1": "서울 동작구 등용로14길 82 2층",
            "lat1": 37.5126493491307,
            "lng1": 126.936051848371,
            "name2": "취복루",
            "address2": "서울 동작구 노량진로 96 2층",
            "lat2": 37.5130997295998,
            "lng2": 126.936327224739,
            "name3": "샹하이",
            "address3": "서울 동작구 만양로 84 지하1층 11,12호",
            "lat3": 37.5113671453073,
            "lng3": 126.945225673807
        },
        12: {
            "name1": "스시준",
            "address1": "서울 동작구 만양로14길 20 1층",
            "lat1": 37.5116671951986,
            "lng1": 126.945453817995,
            "name2": "미스터초밥",
            "address2": "서울 동작구 상도로31길 19",
            "lat2": 37.5062358749024,
            "lng2": 126.944541781061,
            "name3": "호랑이초밥",
            "address3": "서울 동작구 상도로 248",
            "lat3": 37.5047242918459,
            "lng3": 126.945974703857
        },
        13: {
            "name1": "역전우동0410",
            "address1": "서울 동작구 노량진로16길 35 1층",
            "lat1": 37.5120033326881,
            "lng1": 126.944047305147,
            "name2": "길동우동",
            "address2": "서울 동작구 장승배기로 100 1층",
            "lat2": 37.5065534914095,
            "lng2": 126.939959320202,
            "name3": "153구포국수",
            "address3": "서울 동작구 노량진로 154 1층",
            "lat3": 37.5134246896846,
            "lng3": 126.942965253134
        },
        14: {
            "name1": "와우신내떡",
            "address1": "서울 동작구 만양로14가길 3 1층",
            "lat1": 37.5118104872992,
            "lng1": 126.945134750779,
            "name2": "두끼떡볶이",
            "address2": "서울 동작구 만양로 98 2층",
            "lat2": 37.5120709143191,
            "lng2": 126.944442453412,
            "name3": "떡슐랭",
            "address3": "서울 동작구 만양로14가길 27 1층",
            "lat3": 37.5127151824378,
            "lng3": 126.944741487929
        },
        15: {
            "name1": "김밥사랑",
            "address1": "서울 동작구 노량진로 110 1층",
            "lat1": 37.5131853341731,
            "lng1": 126.938038724901,
            "name2": "엄마손김밥",
            "address2": "서울 동작구 등용로14길 81",
            "lat2": 37.5127499054503,
            "lng2": 126.935727705757,
            "name3": "대박분식",
            "address3": "서울 동작구 만양로 89 1층",
            "lat3": 37.511399036386,
            "lng3": 126.944253158202
        },
        16: {
            "name1": "무공돈까스",
            "address1": "서울 동작구 노량진로 110 1층",
            "lat1": 37.5131853341731,
            "lng1": 126.938038724901,
            "name2": "삼삼가마솥돈까스",
            "address2": "서울 동작구 만양로 90-1 1층",
            "lat2": 37.5115178153565,
            "lng2": 126.944501000368,
            "name3": "이든돈카츠",
            "address3": "서울 동작구 만양로14다길 3 1층",
            "lat3": 37.511501287393,
            "lng3": 126.945382568239
        },
        17: {
            "name1": "파머스포케",
            "address1": "서울 동작구 만양로8길 63 1층",
            "lat1": 37.5103107126825,
            "lng1": 126.9458007938,
            "name2": "샐러디",
            "address2": "서울 동작구 노량진로16길 25 1층",
            "lat2": 37.5125442726531,
            "lng2": 126.943998943313,
            "name3": "써브웨이",
            "address3": "서울 동작구 노량진로 152-1 1층",
            "lat3": 37.5134559023022,
            "lng3": 126.942857209421
        },
        18: {
            "name1": "버거락",
            "address1": "서울 동작구 노량진로 157 1층",
            "lat1": 37.5139791754713,
            "lng1": 126.9433563073,
            "name2": "맥도날드",
            "address2": "서울 동작구 노량진로 158",
            "lat2": 37.5134328037672,
            "lng2": 126.943538374682,
            "name3": "노브랜드버거",
            "address3": "서울 동작구 만양로 106 1층",
            "lat3": 37.5129495852342,
            "lng3": 126.944467364767
        },
        19: {
            "name1": "피자보이시나",
            "address1": "서울 동작구 만양로14길 21",
            "lat1": 37.5118693539822,
            "lng1": 126.945396325369,
            "name2": "몽때박피자",
            "address2": "서울 동작구 노량진로14가길 16 2층",
            "lat2": 37.5126013305635,
            "lng2": 126.943482559841,
            "name3": "고피자",
            "address3": "서울 동작구 노량진로 161 1층",
            "lat3": 37.5140337882468,
            "lng3": 126.943947948512
        },
        20: {
            "name1": "스파게티스토리",
            "address1": "서울 동작구 노량진로 140 메가스터디타워 지하1층 B106호",
            "lat1": 37.512756878817,
            "lng1": 126.941700638806,
            "name2": "뚝스토리",
            "address2": "서울 동작구 노량진로12길 12-6 1층",
            "lat2": 37.5129335259683,
            "lng2": 126.938081462097,
            "name3": "이쉐프",
            "address3": "서울 동작구 만양로 95 2층",
            "lat3": 37.5119216912344,
            "lng3": 126.944025197004
        },
        21: {
            "name1": "컵속애",
            "address1": "서울 동작구 노량진로16길 28-1 1층",
            "lat1": 37.512360325336,
            "lng1": 126.943888573986,
            "name2": "한솥도시락",
            "address2": "서울 동작구 노량진로 110 1층 106호",
            "lat2": 37.5131853341731,
            "lng2": 126.938038724901,
            "name3": "본도시락",
            "address3": "서울 동작구 장승배기로 143-1 1층",
            "lat3": 37.510448166056,
            "lng3": 126.939943080504
        },
        22: {
            "name1": "다독이네숯불구이",
            "address1": "서울 동작구 노량진로 106-5 1층",
            "lat1": 37.512959323801,
            "lng1": 126.937793917313,
            "name2": "짠돈",
            "address2": "서울 동작구 만양로18길 18 1층",
            "lat2": 37.5129961696558,
            "lng2": 126.94524472988,
            "name3": "참숯칭따오양꼬치",
            "address3": "서울 동작구 노량진로8길 70 1층",
            "lat3": 37.5119468945341,
            "lng3": 126.938885115094
        },
        23: {
            "name1": "계림원",
            "address1": "서울 동작구 노량진로8길 8 1층",
            "lat1": 37.5130173137316,
            "lng1": 126.936208416941,
            "name2": "레커훈스",
            "address2": "서울 동작구 만양로 83",
            "lat2": 37.5109554496203,
            "lng2": 126.944396906689,
            "name3": "영계소문옛날통닭",
            "address3": "서울 동작구 만양로 112",
            "lat3": 37.5133362743662,
            "lng3": 126.944421042599
        },
        24: {
            "name1": "신나족발",
            "address1": "서울 동작구 등용로 124 1층",
            "lat1": 37.5121830815913,
            "lng1": 126.932178750257,
            "name2": "고려왕족발",
            "address2": "서울 동작구 만양로14길 9",
            "lat2": 37.5115282158397,
            "lng2": 126.94496936875,
            "name3": "족발야시장",
            "address3": "서울 동작구 만양로 110 1층",
            "lat3": 37.5131913106977,
            "lng3": 126.944437777194
        },
        25: {
            "name1": "노량해전",
            "address1": "서울 동작구 만양로16길 4 1층",
            "lat1": 37.5115693412681,
            "lng1": 126.944668473774,
            "name2": "순천집",
            "address2": "서울 동작구 노량진로 80 큐브스테이트 2층",
            "lat2": 37.5128873135611,
            "lng2": 126.934587901193,
            "name3": "오징어야",
            "address3": "서울 동작구 만양로14길 9",
            "lat3": 37.5115282158397,
            "lng3": 126.94496936875
        },
        26: {
            "name1": "건강한밥상",
            "address1": "서울 동작구 장승배기로27길 9-1",
            "lat1": 37.5116991130465,
            "lng1": 126.939399389594,
            "name2": "서일식당",
            "address2": "서울 동작구 노량진로6길 2 1층",
            "lat2": 37.5129359439746,
            "lng2": 126.935858865911,
            "name3": "정가네밥상",
            "address3": "서울 동작구 노량진로10길 28 1층",
            "lat3": 37.5121629184451,
            "lng3": 126.937437835801
        }
    }
    
    df_foods, df_labels = food_data_preprocessing()

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(df_foods, df_labels)

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

    pred = knn.predict(X_test)

    content = {
        "food_result": food_dict.get(pred[0]),
        "food_picture": food_img_list[pred[0] - 1],
        "food_info": eatery_dict.get(pred[0])
    }

    return render(request, 'foods/ai_khm/result.html', content)