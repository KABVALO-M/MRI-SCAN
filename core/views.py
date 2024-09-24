from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

# Login view
def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        # Authenticate the user
        user = authenticate(request, username=username, password=password)

        if user is not None:
            # Log the user in
            login(request, user)
            # Redirect to the home page or dashboard after successful login
            return redirect('dashboard')
        else:
            # Invalid credentials, show an error
            error = "Invalid username or password."
            return render(request, 'core/login.html', {'error': error})

    return render(request, 'core/login.html')

@login_required
def logout_view(request):
    logout(request)
    return redirect('login')

# Dashboard view
@login_required(login_url='/login/')
def dashboard_view(request):
    return render(request, 'core/dashboard.html', {'current_page': 'dashboard'})

@login_required(login_url='/login/')
def doctors_view(request):
    return render(request, 'core/doctors.html', {'current_page': 'doctors'})

@login_required(login_url='/login/')
def patients_view(request):
    return render(request, 'core/patients.html', {'current_page': 'patients'})

@login_required(login_url='/login/')
def analysis_view(request):
    return render(request, 'core/analysis.html', {'current_page': 'analysis'})

@login_required(login_url='/login/')
def training_view(request):
    return render(request, 'core/training.html', {'current_page': 'training'})

@login_required(login_url='/login/')
def settings_view(request):
    return render(request, 'core/settings.html', {'current_page': 'settings'})

@login_required(login_url='/login/')
def profile_view(request):
    return render(request, 'core/profile.html', {'current_page': 'profile'})
