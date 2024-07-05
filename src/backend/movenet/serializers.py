from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers
from .models import Pose

class PoseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pose

        x_vals = serializers.ListField(
            child=serializers.IntegerField(min_value=0, max_value=100)
        )

        height = serializers.FloatField(min_value=0, max_value=300)

        fields = ('pic','vid','kin1','kin2','kin3','kin4','kin5','x_vals', 'height')
        #exclude = ('pic',)