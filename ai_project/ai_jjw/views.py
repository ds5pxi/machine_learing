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
        "논의가 악화될 때 한 사람이 사과하면 논의가 끝난다.",
        "가끔 상황이 어려워도 우리는 차이점을 무시할 수 있다는 것을 안다.",
        "필요할 때, 배우자와의 논의를 처음부터 다시 시작하고 수정할 수 있다.",
        "배우자와 논의할 때, 그와 연락을 취하는 것이 결국 효과가 있을 것이다.",
        "아내와 함께 보낸 시간은 우리에게 특별하다.",
        "우리는 파트너로서 집에서 시간을 보내지 않는다.",
        "우리는 가족이라기보다는 집에서 같은 환경을 공유하는 두 낯선 사람 같다.",
        "아내와 함께하는 휴가를 즐긴다.",
        "아내와 여행하는 것을 즐긴다.",
        "우리의 목표 대부분이 배우자와 공통된다.",
        "미래에 돌아봤을 때, 배우자와 내가 조화를 이루어왔다는 것을 볼 것이라고 생각한다.",
        "배우자와 나는 개인적 자유에 대한 가치가 비슷하다.",
        "배우자와 나는 유희에 대한 감각이 비슷하다.",
        "사람들(자녀, 친구 등)에 대한 우리의 목표가 대부분 같다.",
        "배우자와 우리의 꿈이 유사하고 조화롭다.",
        "사랑이 어떤 것이어야 하는지에 대해 배우자와 호환된다.",
        "인생에서 행복해지는 것에 대해 배우자와 같은 견해를 공유한다.",
        "배우자와 나는 결혼이 어떻게 되어야 하는지에 대한 생각이 비슷하다.",
        "배우자와 나는 결혼에서 역할이 어떻게 되어야 하는지에 대한 생각이 비슷하다.",
        "배우자와 나는 신뢰에 대한 가치가 비슷하다.",
        "아내가 무엇을 좋아하는지 정확히 안다.",
        "배우자가 아플 때 어떻게 돌봐주길 원하는지 안다.",
        "배우자가 좋아하는 음식을 안다.",
        "배우자가 삶에서 어떤 종류의 스트레스를 겪고 있는지 말할 수 있다.",
        "배우자의 내면 세계에 대한 지식이 있다.",
        "배우자의 기본적인 불안을 안다.",
        "배우자의 현재 스트레스 원인이 무엇인지 안다.",
        "배우자의 희망과 소망을 안다.",
        "배우자를 매우 잘 안다.",
        "배우자의 친구들과 그들의 사회적 관계를 안다.",
        "배우자와 논의할 때 공격적인 감정을 느낀다.",
        "배우자와 논의할 때 보통 ‘너는 항상’ 또는 ‘너는 절대’ 같은 표현을 사용한다.",
        "논의 중에 배우자의 성격에 대해 부정적인 발언을 할 수 있다.",
        "논의 중에 공격적인 표현을 사용할 수 있다.",
        "논의 중에 배우자를 모욕할 수 있다.",
        "논의할 때 굴욕적일 수 있다.",
        "배우자와의 논의가 차분하지 않다.",
        "배우자가 주제를 여는 방식을 싫어한다.",
        "논의가 자주 갑자기 발생한다.",
        "상황이 어떻게 되는지 알기 전에 논의를 시작한다.",
        "배우자와 무엇인가를 이야기할 때 갑자기 차분함이 깨진다.",
        "배우자와 논의할 때 그냥 나가고 아무 말도 하지 않는다.",
        "환경을 조금 진정시키기 위해 대부분 침묵을 유지한다.",
        "가끔 집을 잠시 떠나는 것이 좋다고 생각한다.",
        "배우자와 논의하기보다는 침묵하는 것이 낫다.",
        "논의 중 내가 맞더라도 배우자를 다치게 하기 위해 침묵한다.",
        "배우자와 논의할 때, 내 감정을 조절하지 못할까 봐 침묵한다.",
        "논의 중 내가 옳다고 느낀다.",
        "내가 비난받은 것과는 아무런 관련이 없다.",
        "내가 비난받은 것에 대해 실제로 죄가 있는 사람이 아니다.",
        "집에서 문제에 대해 잘못된 사람이 아니다.",
        "배우자의 부족함에 대해 주저하지 않고 말할 것이다.",
        "논의할 때 배우자의 부족함을 상기시킨다.",
        "배우자의 무능함에 대해 말하는 것이 두렵지 않다.",
        "이혼 여부 (예/아니오)"
    ]
    
    content = {
        "df_dvc_html": df_dvc_html,
        "col_list": col_list
    }
   
    return render(request, 'divorce/ai_khm/learning.html', content)

def divorce_data_preprocessing():
    df_divorce = pd.read_csv(r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\ai_jjw\divorce.csv')

    df_data = df_divorce.drop(['Divorce_Y_N'], axis=1)
    df_target = df_divorce['Divorce_Y_N']
     
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

    return render(request, 'divorce/ai_jjw/decision_tree.html', content)