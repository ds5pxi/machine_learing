from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
from django.http import JsonResponse
from sklearn.utils import shuffle

# Create your views here.
def learning(request):
    df_divorce = pd.read_csv(r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\ai_data\ai_khm\divorce_data.csv', sep=';')

    df_dvc_html = df_divorce.to_html(index=False, classes='table table-bordered')

    col_list = [
        "싸웠을 때 우리 중 한 명이 사과하면 싸움은 종료된다",
        "때때로 상황이 어려워져도 우리의 차이점을 무시할 수 있다",
        "필요하면 배우자의 의견을 반영하여 고친다",
        "배우자의 의견이 필요하면 연락한다",
        "배우자와 함께하는 시간은 특별하다",
        "집에서 시간을 잘 안 보낸다",
        "단지 같은 가정에서 사는 가족보다는 남에 가깝다",
        "배우자와 휴가를 즐긴다",
        "배우자와 여행하는 것을 즐긴다",
        "우리의 대부분의 목표는 배우자와 같은 경우가 많다",
        "미래가 되었을 때, 과거를 돌이켜보면 배우자와 화목하게 지낸 경우가 많다",
        "배우자와 나는 개인의 자유를 존중하는 편이다",
        "배우자와 나는 비슷한 연예 감각을 가진 편이다",
        "사람들(아이, 친구 등)에 대한 목표는 대부분 같은 편이다",
        "배우자와 나의 꿈은 비슷하다",
        "우리는 사랑에 대한 생각이 같은 편이다",
        "우리는 인생의 행복에 대해 같은 견해를 가지고 있다",
        "배우자와 나는 결혼에 대한 생각이 비슷하다",
        "배우자와 나는 결혼에서의 역할에 대한 생각이 비슷하다",
        "배우자와 나는 신뢰에 대한 가치관이 비슷하다",
        "배우자가 좋아하는 것을 확실히 안다",
        "배우자가 아플 때 어떻게 돌봐야 할지 잘 안다",
        "배우자가 매우 좋아하는 음식을 잘 안다",
        "배우자가 주로 어떤 스트레스를 받고 있는지 잘 말할 수 있다",
        "배우자의 내면에 대해 잘 안다",
        "배우자의 근본적 불안감에 대해 잘 안다",
        "배우자의 현재 스트레스의 원인을 잘 안다",
        "배우자의 희망과 바람을 잘 안다",
        "배우자에 대해 매우 잘 안다",
        "배우자의 친구와 그들과의 사회적 관계를 잘 안다",
        "배우자와 말다툼을 하면 공격적으로 느낀다",
        "배우자와 상의 할 때 주로 '당신은 항상' 또는 '당신은 절대'와 같은 표현을 사용한다",
        "배우자와 논의할 때 배우자의 성격에 대해 부정적인 말을 할 수 있다",
        "논의하는 동안 공격적인 표현을 할 수 있다",
        "논의하는 동안 배우자를 모욕할 수 있다",
        "논의할 때 나는 면목이 없다",
        "배우자와 논의할 때 차분하지 않다",
        "나는 배우자가 주제를 여는 방식이 싫다",
        "우리의 논의는 흔히 갑자기 발생한다",
        "내가 무슨 일이 일어났는지 알기 전에 논의가 시작된다",
        "배우자와 이야기를 나누다보면 갑작스럽게 침착함이 깨진다",
        "배우자와 다투면 한 마디도 하지 않고 밖으로 나간다",
        "나는 환경을 조금 진정시키기 위해 대부분 침묵한다",
        "가끔은 내가 집을 떠나는 편이 좋다고 생각한다",
        "배우자와 상의하는 것보다 침묵하는 편이 더 낫다",
        "나는 토론할 때 배우자에게 상처를 주기 위해 조용히 하고 있다",
        "배우자와 상의할 때 화를 다스리지 못 할 것 같아서 조용히 있는다",
        "논의 하는 것을 옳다고 생각한다",
        "나는 내가 비난받는 것에 대해 아무 관련이 없다",
        "나는 내가 비난받는 것에 대해 죄책감을 느끼지 않는다",
        "나는 집에 문제가 있다고 잘못 생각하지 않는다",
        "배우자의 부족한 부분에 대해 주저하지 않고 말한다",
        "나는 논의할 때 배우자의 부족한 부분을 상기시킨다",
        "배우자에게 배우자의 무능한 부분에 대해 말할 때 두려워 하지 않는다",
    ]
    
    content = {
        "df_dvc_html": df_dvc_html,
        "col_list": col_list
    }
   
    return render(request, 'divorce/ai_jjw/learning.html', content)

def analyze_divorce(request) :
    if request.method == 'POST' :
        # 사용자가 입력한 점수 데이터 가져오기
        scores = []
        for i in range(1,55) :
            score = request.POST.get(f'score_{i}')
            if score is None or score == '':
                return JsonResponse({'error': f'Score {i} is missing or invalid'})
            scores.append(float(score))
        data = np.array(scores).reshape(1,-1)
        
        # 원본 데이터 불러오기
        file_path = r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\ai_data\ai_khm\divorce_data.csv'
        df = pd.read_csv(file_path, sep=';')

        # 데이터 섞기
        df = shuffle(df, random_state=62)
        # 데이터 저장 (원본 파일을 덮어쓰지 않도록 새로운 파일명 사용)
        file_path_shuffled = r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\ai_data\ai_khm\divorce_data_shuffled.csv'
        df.to_csv(file_path_shuffled, index=False, sep=';')

        save = scores + [None] # 이혼 결과를 저장할 자리 비우기, None은 추후 Divorce가 들어갈 자리다
        df.loc[len(df)] = save
        df.to_csv(file_path, index=False, sep=';')

        # 데이터 전처리
        df_data = df.drop(['Divorce'], axis=1)
        df_target = df['Divorce']

        # NaN 값을 제거하거나 대체
        df_data = df_data.fillna(0)  # NaN 값을 0으로 대체하거나 다른 값으로 대체할 수 있습니다.
        df_target = df_target.fillna(0)  # NaN 값을 0으로 대체

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_data)

        X_train, X_test, y_train, y_test = train_test_split(data_scaled, df_target, test_size=0.3, random_state=62)

        model_choice = request.POST.get('model_choice')

        if model_choice == 'logistic' :
            model = LogisticRegression()
            model.fit(X_train, y_train)
        elif model_choice == 'xgboost' :
            model = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=3)
            model.fit(X_train, y_train)
        else :
            return JsonResponse({'error' : 'Invalid model choice'})
        
        # 예측 수행
        data_scaled_input = scaler.transform(data)
        
        pred = model.predict(data_scaled_input)
        result = int(pred[0])

        # 예측 결과를 csv 파일에 추가로 저장
        df.loc[len(df) - 1, 'Divorce'] = result
        df.to_csv(file_path, index=False, sep=';')

        # 예측 결과와 모델 정확도를 출력 페이지로 전달
        accuracy = accuracy_score(y_test, model.predict(X_test))

        if model_choice == 'logistic':
            return render(request, 'divorce/ai_jjw/decision_tree.html', {'accuracy_score': accuracy, 'result': result})
        elif model_choice == 'xgboost':
            return render(request, 'divorce/ai_jjw/svm.html', {'accuracy_score': accuracy, 'result': result})

    return render(request, 'divorce/ai_jjw/learning.html')
        
# 이혼 확률 데이터 전처리
def divorce_data_preprocessing():
    df_divorce = pd.read_csv(r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\ai_data\ai_khm\divorce_data.csv', sep=';')

    df_data = df_divorce.drop(['Divorce'], axis=1)
    df_target = df_divorce['Divorce']
     
    # print('데이터: ', df_data)
    # print('타겟: ', df_target.tail())

    return df_data, df_target

def decision_tree_view(request):
    df_data, df_target = divorce_data_preprocessing()

    data_scaled = StandardScaler().fit_transform(df_data)
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, df_target, test_size=0.3, random_state=62)

    lr_clf = LogisticRegression()
    lr_clf.fit(X_train, y_train)
    pred = lr_clf.predict(X_test)

    params = {
        'penalty':['l2'],
        'C' : [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
    }

    grid_clf = GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=5)
    grid_clf.fit(data_scaled,df_target)

    print('Best Score : ', grid_clf.best_score_)
    print('Best Hyper Parameters : ', grid_clf.best_params_)

    content = {
        "accuracy_score": accuracy_score(y_test, pred),
        "hyper_param": lr_clf.get_params(),
        # data, target 값을 여기에 추가
        "data" : df_data,
        "target" : df_target
    }

    return render(request, 'divorce/ai_jjw/decision_tree.html', content)

def svm_view(request):
    df_data, df_target = divorce_data_preprocessing()

    data_scaled = StandardScaler().fit_transform(df_data)
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, df_target, test_size=0.3, random_state=62)
    
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=3)
    xgb.set_params(early_stopping_rounds=100, eval_metric='logloss')

    # eval_set을 통해 조기 종료를 설정
    xgb.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])

    pred = xgb.predict(X_test)

    param = {
        'max_depth':range(1,10,1),
        'subsample' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    }

    grid_cv = GridSearchCV(XGBClassifier(), param_grid=param, scoring='accuracy', cv=2, verbose=1, n_jobs=-1)
    grid_cv.fit(X_train, y_train)
    print('최적 파라미터 : ', grid_cv.best_params_)
    print('최고 예측 정확도 : ', grid_cv.best_score_)

    content = {
        "accuracy_score": accuracy_score(y_test, pred),
        "hyper_param": grid_cv.get_params()
    }
    return render(request, 'divorce/ai_jjw/svm.html')
