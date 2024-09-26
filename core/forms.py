from django import forms
from .models import TrainingSession, CustomUser, Hospital, Patient, Analysis
from django.contrib.auth.forms import UserCreationForm, UserChangeForm



class DoctorForm(UserChangeForm):
    hospital = forms.ModelChoiceField(
        queryset=Hospital.objects.all(),
        empty_label="Select a Hospital", 
        widget=forms.Select(attrs={
            'id': 'hospital',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
    )

    class Meta:
        model = CustomUser
        fields = ['email', 'first_name', 'last_name', 'ssn', 'phone_number', 'hospital', 'password']

        # Exclude password fields from the fields for this form, if you don't want to use UserChangeForm's default behavior.
        exclude = ['password']  # You can use 'exclude' or specify 'fields' directly

    def __init__(self, *args, **kwargs):
        super(DoctorForm, self).__init__(*args, **kwargs)

        # Add custom CSS classes and IDs to each field
        for field in self.fields:
            self.fields[field].widget.attrs.update({
                'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
            })

        # Password fields are optional for updates
        self.fields['password1'] = forms.CharField(
            required=False,
            widget=forms.PasswordInput(attrs={
                'id': 'password1',
                'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
            }),
            label="New Password"
        )

        self.fields['password2'] = forms.CharField(
            required=False,
            widget=forms.PasswordInput(attrs={
                'id': 'password2',
                'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
            }),
            label="Confirm New Password"
        )

    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get("password1")
        password2 = cleaned_data.get("password2")

        # Validate passwords only if they are provided
        if password1 or password2:  # Check if at least one is provided
            if password1 and password2 and password1 != password2:
                self.add_error('password2', "Passwords do not match.")
        return cleaned_data

class PatientForm(forms.ModelForm):

    class Meta:
        model = Patient
        fields = ['first_name', 'last_name', 'age', 'gender', 'physical_address', 'phone_number', 'email']

    def __init__(self, *args, **kwargs):
        super(PatientForm, self).__init__(*args, **kwargs)
        
        # Apply Tailwind CSS classes to each field
        self.fields['first_name'].widget.attrs.update({
            'id': 'first_name',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['last_name'].widget.attrs.update({
            'id': 'last_name',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['age'].widget.attrs.update({
            'id': 'age',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['gender'].widget.attrs.update({
            'id': 'gender',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['physical_address'].widget.attrs.update({
            'id': 'physical_address',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['phone_number'].widget.attrs.update({
            'id': 'phone_number',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['email'].widget.attrs.update({
            'id': 'email',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })


class TrainingSessionForm(forms.ModelForm):
    class Meta:
        model = TrainingSession
        fields = ['training_file', 'epochs', 'batch_size', 'learning_rate', 'label_info']
        widgets = {
            'training_file': forms.ClearableFileInput(attrs={'class': 'form-control-file'}),
            'epochs': forms.NumberInput(attrs={'class': 'form-control', 'min': 1}),
            'batch_size': forms.NumberInput(attrs={'class': 'form-control', 'min': 1}),
            'learning_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': 0.0001}),
            'label_info': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }


class AnalysisForm(forms.ModelForm):

    class Meta:
        model = Analysis
        fields = ['patient', 'image', 'result']
    
    def __init__(self, *args, **kwargs):
        super(AnalysisForm, self).__init__(*args, **kwargs)
        
        # Apply Tailwind CSS classes to each field
        self.fields['patient'].widget.attrs.update({
            'id': 'patient_code',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
            'placeholder': 'Enter patient code',
        })
        self.fields['image'].widget.attrs.update({
            'id': 'image',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
            'accept': 'image/*',  # Limit the input to image files
        })
        self.fields['result'].widget.attrs.update({
            'id': 'result',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
            'placeholder': 'Enter analysis result',
        })


class HospitalForm(forms.ModelForm):

    class Meta:
        model = Hospital
        fields = ['name', 'physical_address', 'city', 'state', 'country']

    def __init__(self, *args, **kwargs):
        super(HospitalForm, self).__init__(*args, **kwargs)
        
        # Apply Tailwind CSS classes to each field
        self.fields['name'].widget.attrs.update({
            'id': 'name',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['physical_address'].widget.attrs.update({
            'id': 'physical_address',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['city'].widget.attrs.update({
            'id': 'city',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['state'].widget.attrs.update({
            'id': 'state',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['country'].widget.attrs.update({
            'id': 'country',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })