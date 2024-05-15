from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers
from .models import Pose

class PoseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pose
        fields = ('pic','vid')
        #exclude = ('pic',)