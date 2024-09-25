from django.urls import path
from django.contrib.auth.views import LogoutView
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('login/', views.login_view, name='login'), 
    path('logout/', views.logout_view, name='logout'),
    path('doctors/', views.doctors_view, name='doctors'),
    path('patients/', views.patients_view, name='patients'),
    path('analysis/', views.analysis_view, name='analysis'),
    path('training/', views.training_view, name='training'),
    path('settings/', views.settings_view, name='settings'),
    path('profile/', views.profile_view, name='profile'),
     path('doctors/add/', views.add_doctor_view, name='add_doctor'),
]
