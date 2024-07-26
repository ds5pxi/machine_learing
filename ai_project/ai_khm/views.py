from django.shortcuts import render
import pandas as pd
import numpy as np

# Create your views here.
def learning(request):
    df_divorce = pd.read_csv('D:/machine_learing/ai_project/static/file/ai_data/ai_khm/divorce_data.csv', sep=';')

    df_dvc_html = df_divorce.to_html(index=False, classes='table table-bordered')
    
    content = {
        "df_dvc_html": df_dvc_html
    }
   
    return render(request, 'divorce/ai_khm/learning.html', content)

# 이혼 확률 데이터 전처리
def divorce_data_preprocessing(df_divorce):
    df_data = df_divorce.drop(['Divorce'], axis=1)
    df_target = df_divorce['Divorce']
     
    print('데이터: ', df_data)
    print('타겟: ', df_target.tail())

    return df_data, df_target