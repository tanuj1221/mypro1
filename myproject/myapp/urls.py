# myapp/urls.py
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
     path('', views.login_view, name='login'),
    path('index/', views.index, name='index'),
     path('file_upload/', views.file_upload, name='file_upload'),
   

]

