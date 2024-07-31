from django.shortcuts import render
import matplotlib
from matplotlib import font_manager, rc
import platform

# Create your views here.
def main(request):
    return render(request, 'main/main.html');

# 맷플롯립 한글 적용 메서드
def korean():
    try:
        if platform.system() == 'Windows':
            font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
            rc('font', family=font_name)
        else:
            # Mac
            rc('font', family='AppleGothic')
    except:
        pass

    matplotlib.rcParams['axes.unicode_minus'] = False