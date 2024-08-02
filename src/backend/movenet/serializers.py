from django.contrib.auth import authenticate
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers
from .models import Pose,UserProfile

class PoseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pose

        x_vals = serializers.ListField(
            child=serializers.IntegerField(min_value=0, max_value=100)
        )

        height = serializers.FloatField(min_value=0, max_value=300)

        slowmo = serializers.IntegerField()

        image_urls = serializers.ListField(
            child=serializers.URLField(max_length=200)
        )

        fields = ('pic','vid','kin1','kin2','kin3','kin4','kin5','x_vals','height', 'slowmo')
        #exclude = ('pic',)

"""from django.contrib.auth.models import User
from .models import UserProfile

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer()

    class Meta:
        model = UserProfile
        fields = ['user', 'dob', 'height', 'profile_pic']"""


from rest_framework import serializers
from django.contrib.auth.models import User
from rest_framework_simplejwt.tokens import RefreshToken

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['user', 'dob', 'height', 'profile_pic','name',"femur_len"]

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('username', 'password', 'email')
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User.objects.create_user(validated_data['username'], validated_data['email'], validated_data['password'])
        return user

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()

    def validate(self, data):
        user = authenticate(**data)
        if user and user.is_active:
            return user
        raise serializers.ValidationError("Incorrect Credentials")

