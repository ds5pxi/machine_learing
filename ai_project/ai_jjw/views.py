from django.shortcuts import render
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from django.http import JsonResponse
import requests
from bs4 import BeautifulSoup
import re
from urllib.request import urlopen
import urllib.request
import os
import io
import sys
import json
import urllib.parse
import urllib.request

# 음식 추천 시스템
def foods_learning(request):
    return render(request, 'foods/ai_jjw/foods_learning.html')

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

    # 저장된 파일 경로
    file_path = r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\food_jjw\foodsearch.csv'
    
    # 파일 읽기
    df = pd.read_csv(file_path)
    
    # 데이터 전처리
    df_data = df.drop(columns=['menu'], axis=1)
    df_target = df['menu']

    # NaN 값을 각 열의 평균 값으로 대체 (평균 값은 정수로 변환)
    df_data = df_data.apply(lambda x: x.fillna(int(x.mean())))

    # df_target의 NaN 값 처리, 있다면 모드(최빈값)으로 대체하고, 없을경우 기본값 0으로 대체
    if df_target.isnull().any():
        print("NaN values found in df_target:", df_target[df_target.isnull()])
        mode_value = df_target.mode()
        if not mode_value.empty:
            df_target = df_target.fillna(mode_value[0])
        else:
            df_target = df_target.fillna(0)  # 기본값으로 대체
        
    if request.method == 'POST':
        # NaN 값을 다시 한 번 확인하고 처리
        if df_target.isnull().any():
            print("NaN values after mapping in df_target:", df_target[df_target.isnull()])
            mode_value = df_target.mode()
            if not mode_value.empty:
                df_target = df_target.fillna(mode_value[0])
            else:
                df_target = df_target.fillna(0)
        
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(df_data, df_target)

        # 입력된 데이터로 테스트 데이터 프레임을 생성
        X_test = pd.DataFrame({
            'emotion': [request.POST['emotion']],
            'season': [request.POST['season']],
            'weather': [request.POST['weather']],
            'people': [request.POST['people']],
            'price': [request.POST['price']],
            'time': [request.POST['time']],
            'sex': [request.POST['sex']]
        })
        pred = knn.predict(X_test) # 예측 진행

        # 예측 결과를 조회
        query = food_dict.get(pred[0])

        if query : # 예측된 음식이 존재하면 이미지를 검색하여 url을 가져옴
            image_urls = []
            print("Predicted food:", query)  # 예측된 음식 출력
            # () 안의 내용과 "또는"을 제거하여 검색어 구성
            query_for_search = re.sub(r'\([^)]*\)', '', query) # 검색어에서 특수문자 제거
            query_for_search = query_for_search.replace('또는', '').strip()
            query_for_search = query_for_search.replace(" ", "")

            client_id = "gQtHPxKh10uWGP4XLEQa"
            client_secret = "mrRY6xHHrS"
            url = "https://openapi.naver.com/v1/search/image?query=" + urllib.parse.quote(query_for_search) # JSON 결과
            headers = { # 네이버 api 인증 헤더를 설정
                "X-Naver-Client-Id" : client_id,
                "X-Naver-Client-Secret" : client_secret
            }
            req = urllib.request.Request(url, headers=headers) # 요청 객체 설정
            response = urllib.request.urlopen(req) # 요청을 보내고 응답을 받음
            rescode = response.getcode()
            if(rescode==200): # getcode로 받은 응답이 성공적이면 데이터를 파싱하여 이미지 url출력
                response_body = response.read().decode('utf-8') # 읽어낸 응답내용(read())dmf utf-8 로 디코딩하여 이를 통해 응답 데이터를 문자열로 변환
                data = json.loads(response_body) # JSON 형식의 문자열인 'response_body'를 딕셔너리로 변환
                image_urls = [item['link'] for item in data['items']] # 상위 5개 이미지 링크 추출
                # data[items] : 파싱된 데이터 딕셔너리에서 items 키에 해당하는 값(이미지 검색 결과 리스트)을 가져옴
                # [item['link'] for item in data['items']
                # 리스트 컴프리헨션을 사용하여 각 검색 결과 항목의 'link'값을 추출
                # 검색 결과 리스트에서 각 이미지의 url을 추출하여 리스트에 저장
                print("Image URLs : ", image_urls) # 디버그 출력
            else:
                print("Error Code:" + rescode)
                image_urls= []
        else :
            image_urls = []

        print("Image URLs: ", image_urls)  # 디버그 출력

        content = {
            "food_result": query,
            "image_urls": image_urls[:5]
        } 

        return render(request, 'foods/ai_jjw/result.html', content)

    return render(request, 'foods/ai_jjw/result.html')

def learning(request):
    df_divorce = pd.read_csv(r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\ai_data\ai_khm\divorce_merge.csv')

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
   
    return render(request, 'divorce/ai_jjw/learning.html', content)

def divorce_data_preprocessing():
    df_divorce = pd.read_csv(r'C:\Users\user\Desktop\machine_learing\ai_project\static\file\ai_data\ai_khm\divorce_merge.csv')

    df_data = df_divorce.drop(["Divorce_Y_N"], axis=1)
    df_target = df_divorce["Divorce_Y_N"]

    return df_data, df_target