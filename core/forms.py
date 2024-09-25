from django import forms
from .models import TrainingSession, CustomUser, Hospital
from django.contrib.auth.forms import UserCreationForm



class DoctorForm(UserCreationForm):
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
        fields = ['email', 'first_name', 'last_name', 'ssn', 'phone_number', 'hospital', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super(DoctorForm, self).__init__(*args, **kwargs)
        
        # Add custom CSS classes and IDs to each field in __init__ method
        self.fields['email'].widget.attrs.update({
            'id': 'email',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['first_name'].widget.attrs.update({
            'id': 'first_name',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['last_name'].widget.attrs.update({
            'id': 'last_name',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['ssn'].widget.attrs.update({
            'id': 'ssn',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['phone_number'].widget.attrs.update({
            'id': 'phone_number',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['hospital'].widget.attrs.update({
            'id': 'hospital',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['password1'].widget.attrs.update({
            'id': 'password1',
            'class': 'mt-1 block w-full border border-gray-300 rounded-lg py-2 px-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
        })
        self.fields['password2'].widget.attrs.update({
            'id': 'password2',
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


class AnalysisForm(forms.Form):
    # Field for uploading image files (JPG, PNG, BMP)
    image_file = forms.ImageField(
        label='Upload Image',
        widget=forms.ClearableFileInput(attrs={'accept': 'image/jpeg,image/png,image/bmp', 'required': 'required'}),
        help_text='Supported formats: JPG, PNG, BMP. Max size: 5MB.'
    )

    # Input for setting the confidence threshold
    confidence_threshold = forms.FloatField(
        label='Confidence Threshold',
        initial=0.5,  # Default threshold
        widget=forms.NumberInput(attrs={'min': 0.0, 'max': 1.0, 'step': 0.01}),
        help_text='Set the minimum confidence level for predictions (0.0 to 1.0).'
    )
    
    # Custom clean method to handle file size validation (Optional)
    def clean_image_file(self):
        image = self.cleaned_data.get('image_file')
        if image:
            if image.size > 5 * 1024 * 1024:  # 5MB limit
                raise forms.ValidationError('Image file is too large (max size is 5MB).')
            return image