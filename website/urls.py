from django.urls import path
from . import views

# these are url configurations

# only deals with shit that's related 
urlpatterns = [
    path('generate_response/', views.generate_response, name='generate_response')
]

