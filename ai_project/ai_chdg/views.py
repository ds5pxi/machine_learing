from django.shortcuts import render, redirect
from django.conf import settings
import csv
import joblib
import os
import numpy as np

# 모델 불러오기
model_path = os.path.join(os.path.dirname(__file__), 'knn_model.pkl')
model = joblib.load(model_path)

def survey(request):
    if request.method == 'POST':
        # 설문 응답을 리스트로 받기
        responses = [
            int(request.POST.get('question1')),
            int(request.POST.get('question2')),
            int(request.POST.get('question3')),
            int(request.POST.get('question4')),
            int(request.POST.get('question5')),
            int(request.POST.get('question6')),
            int(request.POST.get('question7')),
            int(request.POST.get('question8')),
            int(request.POST.get('question9')),
            int(request.POST.get('question10')),
            int(request.POST.get('question11')),
            int(request.POST.get('question12')),
            int(request.POST.get('question13')),
            int(request.POST.get('question14')),
            int(request.POST.get('question15')),
            int(request.POST.get('question16')),
            int(request.POST.get('question17')),
            int(request.POST.get('question18')),
            int(request.POST.get('question19')),
            int(request.POST.get('question20'))
        ]
        
        # 모델 예측
        data = np.array(responses).reshape(1, -1)
        prediction = model.predict(data)[0]
        probability = round(model.predict_proba(data)[0][prediction] * 100, 2)      # 백분율 계산 및 소수점 2자리 반올림 처리
        
        # 결과 페이지 렌더링
        return render(request, 'divorce/ai_chdg/result.html', {
            'prediction': prediction,
            'probability': probability
        })
    
    return render(request, 'divorce/ai_chdg/survey.html')

def save_survey_data(request):
    if request.method == 'POST':
        # 설문지 데이터 가져오기
        question1 = request.POST.get('question1')
        question2 = request.POST.get('question2')
        question3 = request.POST.get('question3')
        question4 = request.POST.get('question4')
        question5 = request.POST.get('question5')
        question6 = request.POST.get('question6')
        question7 = request.POST.get('question7')
        question8 = request.POST.get('question8')
        question9 = request.POST.get('question9')
        question10 = request.POST.get('question10')
        question11 = request.POST.get('question11')
        question12 = request.POST.get('question12')
        question13 = request.POST.get('question13')
        question14 = request.POST.get('question14')
        question15 = request.POST.get('question15')
        question16 = request.POST.get('question16')
        question17 = request.POST.get('question17')
        question18 = request.POST.get('question18')
        question19 = request.POST.get('question19')
        question20 = request.POST.get('question20')

        # CSV 파일 경로 설정
        csv_file_path = os.path.join(settings.BASE_DIR, 'survey_data.csv') 

        # CSV 파일 존재 여부 확인 밑 헤더 추가
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # 파일이 존재하지 않으면 헤더 추가
            if not file_exists:
                writer.writerow(['harmony','marriage','roles','trust','enjoy_travel','happy','love','friends_social','freeom_value','anxieties','sudden_discussion','calm_breaks'
                                 ,'idk_whats_going_on','inner_world','current_stress','dreams','likes','people_goals','not_calm','stresses'])

            # 데이터 추가
            writer.writerow([question1, question2, question3, question4, question5, question6, question7, question8, question9, question10, question11, question12, question13, question14, question15, question16, question17, question18, question19, question20])    
            
        # 결고 페이지로 리다이렉트 또는 결과 보여주기
        return redirect('result_page')
    
    return render(request, 'survey.html')

def result(request):
    if request.method == 'POST':
        # 모든 POST 데이터 출력 (디버깅 용도)
        print(request.POST)

        # 설문조사 결과를 받아와서 모델을 사용하여 예측
        try:
            features = [
                int(request.POST.get('harmony')),
                int(request.POST.get('marriage')),
                int(request.POST.get('roles')),
                int(request.POST.get('trust')),
                int(request.POST.get('enjoy_travel')),
                int(request.POST.get('happy')),
                int(request.POST.get('love')),
                int(request.POST.get('friends_social')),
                int(request.POST.get('freeom_value')),
                int(request.POST.get('anxieties')),
                int(request.POST.get('sudden_discussion')),
                int(request.POST.get('calm_breaks')),
                int(request.POST.get('idk_whats_going_on')),
                int(request.POST.get('inner_world')),
                int(request.POST.get('current_stress')),
                int(request.POST.get('dreams')),
                int(request.POST.get('likes')),
                int(request.POST.get('people_goals')),
                int(request.POST.get('not_calm')),
                int(request.POST.get('stresses'))
            ]
        except TypeError as e:
            print(f"TypeError: {e}")
            return render(request, 'divorce/ai_chdg/survey.html', {'error': 'All fields are required.'})

        prediction = model.predict([features])[0]
        probability = np.max(model.predict_proba([features])) * 100
        
        return render(request, 'divorce/ai_chdg/result.html', {
            'prediction': prediction,
            'probability': probability,
        })

    return render(request, 'divorce/ai_chdg/survey.html')