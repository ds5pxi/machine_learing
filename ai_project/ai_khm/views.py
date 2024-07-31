from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from main.views import korean
import matplotlib.pyplot as plt
import base64
from io import BytesIO

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


def svm_view(request):
    return render(request, 'divorce/ai_khm/svm.html')