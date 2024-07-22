from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Pose
# Register your models here.
from .models import UserProfile

admin.site.register(Pose)
admin.site.register(UserProfile)
