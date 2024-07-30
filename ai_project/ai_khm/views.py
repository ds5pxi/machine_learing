from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
        "이혼 결과"
    ]
    
    content = {
        "df_dvc_html": df_dvc_html,
        "col_list": col_list
    }
   
    return render(request, 'divorce/ai_khm/learning.html', content)

# 이혼 확률 데이터 전처리
def divorce_data_preprocessing():
    df_divorce = pd.read_csv(r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\ai_data\ai_khm\divorce_data.csv', sep=';')

    df_data = df_divorce.drop(['Divorce'], axis=1)
    df_target = df_divorce['Divorce']
     
    print('데이터: ', df_data)
    print('타겟: ', df_target.tail())

    return df_data, df_target

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

def svm_view(request):
    return render(request, 'divorce/ai_khm/svm.html')
