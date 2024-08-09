from django.shortcuts import render
from django.core.paginator import Paginator

import joblib
import os
import numpy as np
import pickle

# Create your views here.(기존에 있던 함수)
def learning(request):
    return render(request, 'divorce/ai_chdg/learning.html');

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
        probability = model.predict_proba(data)[0][prediction]
        
        # 결과 페이지 렌더링
        return render(request, 'divorce/ai_chdg/result.html', {
            'prediction': prediction,
            'probability': probability
        })
    
    return render(request, 'divorce/ai_chdg/survey.html')



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
                int(request.POST.get("idk_whats_going_on")),
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




