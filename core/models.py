# core/models.py

from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from django.utils.translation import gettext_lazy as _

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

class CustomUser(AbstractBaseUser, PermissionsMixin):
    """
    Custom User model that uses email instead of username.
    """
    email = models.EmailField(_('email address'), unique=True)
    username = models.CharField(max_length=150, blank=True, null=True)
    first_name = models.CharField(max_length=30, blank=True)
    last_name = models.CharField(max_length=30, blank=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    
    objects = CustomUserManager()
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']

    def __str__(self):
        return self.email


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