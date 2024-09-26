# core/models.py

from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator

def validate_image(image):
    max_size = 5 * 1024 * 1024  # 5 MB limit
    if image.size > max_size:
        raise ValidationError("Image file too large ( > 5MB )")
    if not image.content_type.startswith('image'):
        raise ValidationError("File type is not image.")

# Custom User Manager for handling user creation
class CustomUserManager(BaseUserManager):
    """
    Custom manager for CustomUser that overrides the create_user and create_superuser methods.
    """
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError(_('The Email field must be set'))
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError(_('Superuser must have is_staff=True.'))
        if extra_fields.get('is_superuser') is not True:
            raise ValueError(_('Superuser must have is_superuser=True.'))

        return self.create_user(email, password, **extra_fields)


# Custom User model with SSN, Hospital foreign key, and phone number
class CustomUser(AbstractBaseUser, PermissionsMixin):
    """
    Custom User model that uses email instead of username and adds SSN, phone number, and Hospital reference.
    """
    email = models.EmailField(_('email address'), unique=True)
    username = models.CharField(max_length=150, blank=True, null=True)
    first_name = models.CharField(max_length=30, blank=True)
    last_name = models.CharField(max_length=30, blank=True)
    ssn = models.CharField(max_length=11, blank=True, null=True)  # SSN field for user
    phone_number = models.CharField(max_length=15, blank=True, validators=[RegexValidator(regex=r'^\+?1?\d{9,15}$', message="Phone number must be in the format: '+999999999'. Up to 15 digits allowed.")])  # Doctor's phone number
    hospital = models.ForeignKey('Hospital', on_delete=models.SET_NULL, null=True, blank=True)  # Reference to Hospital
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=True)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']

    def __str__(self):
        return self.email


# Hospital model
class Hospital(models.Model):
    name = models.CharField(max_length=255)
    physical_address = models.TextField()
    city = models.CharField(max_length=100, blank=True)
    state = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return self.name



# Patient model with phone number, email, and unique patient_code
class Patient(models.Model):
    """
    Model to store patient details.
    """
    patient_code = models.CharField(max_length=50, unique=True, db_index=True)  # Unique code for patient identification
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    age = models.IntegerField()
    gender_choices = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    gender = models.CharField(max_length=1, choices=gender_choices)
    physical_address = models.TextField()
    phone_number = models.CharField(max_length=15, blank=True, validators=[RegexValidator(regex=r'^\+?1?\d{9,15}$', message="Phone number must be in the format: '+999999999'. Up to 15 digits allowed.")])  # Patient's phone number
    email = models.EmailField(_('email address'), blank=True)  # Patient's email

    def save(self, *args, **kwargs):
        """
        Override the save method to generate a unique patient_code if not provided.
        """
        if not self.patient_code:
            # Generate the base code using the first two letters of first and last name
            base_code = f"{self.first_name[:2].upper()}{self.last_name[:2].upper()}"

            # Use a default patient_code first
            patient_code = base_code
            number = 1

            # Check for existing patient codes with the same base, and append a number if needed
            while Patient.objects.filter(patient_code=patient_code).exists():
                # Increment the number and append to the base code until a unique one is found
                patient_code = f"{base_code}{number}"
                number += 1

            self.patient_code = patient_code

        super().save(*args, **kwargs)  # Call the real save method

    def __str__(self):
        return f"{self.first_name} {self.last_name}"


# TrainingSession remains unchanged
class TrainingSession(models.Model):
    """
    Model to store training session details.
    """
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    training_file = models.FileField(upload_to='uploads/train_data/')  # Path to the uploaded training data
    epochs = models.IntegerField(default=10)  # Number of epochs for training
    batch_size = models.IntegerField(default=32)  # Batch size for training
    learning_rate = models.DecimalField(max_digits=5, decimal_places=4, default=0.001)  # Learning rate
    label_info = models.TextField(blank=True, null=True)  # Additional label information if needed
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp for when the session was created
    status = models.CharField(max_length=20, default='pending')  # Status of the training session (e.g., pending, in-progress, completed)

    def __str__(self):
        return f"Training Session {self.id} by {self.user.email} - Status: {self.status}"


# Analysis model
class Analysis(models.Model):
    """
    Model to store analysis details for a specific patient performed by a doctor.
    """
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='analyses')  # Link to Patient
    doctor = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='doctor')  # Link to Doctor (CustomUser)
    analysis_date = models.DateTimeField(auto_now_add=True)  # Date of the analysis (automatically set when created)
    image = models.ImageField(upload_to='uploads/analysis_images/', blank=True, null=True, help_text="Upload image (optional)", validators=[validate_image])  # Optional image upload (e.g., scan, X-ray)
    result = models.TextField(help_text="Result of the analysis")  # Detailed result or findings of the analysis

    def __str__(self):
        return f"Analysis for {self.patient} by {self.doctor} on {self.analysis_date}"


