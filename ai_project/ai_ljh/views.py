from django.shortcuts import render

# Create your views here.
def learning(request):
    return render(request, 'divorce/ai_ljh/learning.html');

def foods_learning(request):
    return render(request, 'foods/ai_ljh/learning.html');