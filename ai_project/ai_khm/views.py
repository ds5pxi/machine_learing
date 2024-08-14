from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from main.views import korean
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Create your views here.

# 공통 그래프 셋팅
def comm_graph_setting():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return graph

# 음식 데이터 전처리
def food_data_preprocessing():
    df_raw_foods = pd.read_excel('D:/machine_learing/ai_project/static/file/ai_data/ai_khm/food_data_extend.xlsx')

    df_foods = df_raw_foods.drop(columns=['menu'])
    df_labels = df_raw_foods['menu']

    return df_foods, df_labels

# 음식 knn 정확도 그래프
def food_knn_accuracy_graph(data, labels):
    korean()        # 한글 셋팅
    
    if plt.get_backend().upper() == "AGG":
        plt.clf()       # 메모리에 파이플롯 데이터 존재 시 화면 초기화

    training_accuracy = []
    test_accuracy = []
    n_neighbors_settings = range(1, 16)

    X_food_train, X_food_test, y_food_train, y_food_test = train_test_split(data, labels, test_size=0.2, random_state=10)

    for n_neighbor in n_neighbors_settings:
        knn = KNeighborsClassifier(n_neighbors=n_neighbor)
        knn.fit(X_food_train, y_food_train)
        training_accuracy.append(knn.score(X_food_train, y_food_train))
        test_accuracy.append(knn.score(X_food_test, y_food_test))
    
    plt.switch_backend('AGG')
    plt.plot(n_neighbors_settings, training_accuracy, label='훈련 정확도')
    plt.plot(n_neighbors_settings, test_accuracy, label='검증 정확도')
    plt.ylabel('정확도')
    plt.xlabel('n_neighbors')
    plt.legend()

    graph = comm_graph_setting()

    return graph
    
# 음식 추천 시스템
def foods_survey(request):
    return render(request, 'foods/ai_khm/survey.html')

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

    knn = KNeighborsClassifier(n_neighbors=1)
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
        "food_info": eatery_dict.get(pred[0]),
        "food_graph": food_knn_accuracy_graph(df_foods, df_labels)
    }

    return render(request, 'foods/ai_khm/result.html', content)