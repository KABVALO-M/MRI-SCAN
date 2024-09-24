from django import forms
from .models import TrainingSession

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
