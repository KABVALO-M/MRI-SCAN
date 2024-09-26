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
    path('hospitals/', views.hospitals_view, name='hospitals'),
    path('hospitals/add/', views.add_hospital_view, name='add_hospital'),
    path('training/', views.training_view, name='training'),
    path('settings/', views.settings_view, name='settings'),
    path('profile/', views.profile_view, name='profile'),
    path('doctors/add/', views.add_doctor_view, name='add_doctor'),
    path('patients/add/', views.add_patient_view, name='add_patient'),
    path('training-results/', views.training_results_view, name='training_results'),
    path('analysis/<int:id>/', views.view_analysis_view, name='view_analysis'),
    path('doctors/edit/<int:doctor_id>/', views.edit_doctor_view, name='edit_doctor'),
    path('doctors/delete/<int:doctor_id>/', views.delete_doctor_view, name='delete_doctor'),
    path('patients/edit/<int:patient_id>/', views.edit_patient_view, name='edit_patient'),
    path('patients/delete/<int:patient_id>/', views.delete_patient_view, name='delete_patient'),
    path('hospitals/edit/<int:hospital_id>/', views.edit_hospital_view, name='edit_hospital'),
    path('hospitals/delete/<int:hospital_id>/', views.delete_hospital_view, name='delete_hospital'),
]
