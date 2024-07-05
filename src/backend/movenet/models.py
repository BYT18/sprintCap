from django.db import models
from django.contrib.auth.models import User

class Pose(models.Model):
    pic = models.FileField(upload_to='pics/', blank=True, null=True)
    vid = models.FileField(upload_to='pics/', blank=True, null=True)
    kin1 = models.ImageField(upload_to='pics/', blank=True, null=True)
    kin2 = models.ImageField(upload_to='pics/', blank=True, null=True)
    kin3 = models.ImageField(upload_to='pics/', blank=True, null=True)
    kin4 = models.ImageField(upload_to='pics/', blank=True, null=True)
    kin5 = models.ImageField(upload_to='pics/', blank=True, null=True)

    x_vals = models.JSONField(null=True, blank=True)  # Requires Django 3.1+
    height = models.FloatField(null = True, blank=True)


    def __str__(self):
        return "Pose"
