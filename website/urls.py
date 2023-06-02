from django.urls import include, path
from . import views

# these are url configurations

# only deals with shit that's related 
urlpatterns = [
    path('', views.home, name='home'),
    path('define_word/', views.define_word, name='define_word'),
    path('generate_response/', views.generate_response, name='generate_response'),
]

