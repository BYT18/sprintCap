from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Pose
# Register your models here.

admin.site.register(Pose)
