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
    if request.method == 'POST' :
        # 사용자가 입력한 점수 데이터 가져오기
        scores = []
        for i in range(1,8) :
            score = request.POST.get(f'score_{i}')
            if score is None or score == '':
                return JsonResponse({'error': f'Score {i} is missing or invalid'})
            scores.append(score)
        data = np.array(scores).reshape(1,-1)
        
        # 원본 데이터 불러오기
        file_path = r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\food_jjw\food_1.csv'
        df = pd.read_csv(file_path, sep=';')

        # 데이터 섞기
        df = shuffle(df, random_state=62)
        # 데이터 저장 (원본 파일을 덮어쓰지 않도록 새로운 파일명 사용)
        file_path_shuffled = r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\food_jjw\food_1_shuffled.csv'
        df.to_csv(file_path_shuffled, index=False, sep=';')

        df.loc[len(df)] = data
        df.to_csv(file_path, index=False, sep=';')

        # 데이터 전처리
        df_data = df.drop(['menu(class)'], axis=1)
        df_target = df['menu(class)']

        # NaN 값을 제거하거나 대체
        df_data = df_data.fillna(0)  # NaN 값을 0으로 대체하거나 다른 값으로 대체할 수 있습니다.
        df_target = df_target.fillna(0)  # NaN 값을 0으로 대체

        scaler = StandardScaler()
        # data_scaled = scaler.fit_transform(df_data)

        # X_train, X_test, y_train, y_test = train_test_split(data_scaled, df_target, test_size=0.3, random_state=62)

        # 예측 수행
        data_scaled_input = scaler.transform(data)
        
        pred = LogisticRegression.predict(data_scaled_input)
        result = int(pred[0])

        # 예측 결과를 csv 파일에 추가로 저장
        df.loc[len(df) - 1, 'Divorce'] = result
        df.to_csv(file_path, index=False, sep=';')

        # # 예측 결과와 모델 정확도를 출력 페이지로 전달
        # accuracy = accuracy_score(y_test, pred.predict(X_test))

    return render(request, 'foods/ai_jjw/learning.html');

# 데이터 전처리
def food_data_preprocessing():
    df_divorce = pd.read_csv(r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\food_jjw\food_1.csv', sep=';')

    df_data = df_divorce.drop(['menu(class)'], axis=1)
    df_target = df_divorce['menu(class)']

    return df_data, df_target

def LogisticRegression_view(request):
    df_data, df_target = food_data_preprocessing()

    data_scaled = StandardScaler().fit_transform(df_data)
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, df_target, test_size=0.3, random_state=62)

    lr_clf = LogisticRegression()
    lr_clf.fit(X_train, y_train)
    pred = lr_clf.predict(X_test)

    params = {
        'penalty':['l2'],
        'C' : [0.001, 0.01, 0.1, 1, 10, 100]
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

    return render(request, 'foods/ai_jjw/decision_tree.html', content)

def foods_learning(request):
    return render(request, 'foods/ai_jjw/learning.html');