"""
URL configuration for ai_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path
from django.views.generic import RedirectView

# 서브 앱 url 등록(include)
urlpatterns = [
    path('admin/', admin.site.urls),
    path('main/', include('main.urls')),
    re_path(r'^$', RedirectView.as_view(url='/main/', permanent=True)),
    path('ai_khm/', include('ai_khm.urls')),
    path('ai_chdg/', include('ai_chdg.urls')),
    path('ai_jjw/', include('ai_jjw.urls')),
    path('ai_ljh/', include('ai_ljh.urls')),
]
