from django.db import models
from django.contrib.auth.models import User

class Pose(models.Model):
    pic = models.ImageField(upload_to='pics/', blank=True, null=True)
    vid = models.FileField(upload_to='pics/', blank=True, null=True)

    def __str__(self):
        return "Pose"
