from rest_framework import generics,views
from .serializers import PoseSerializer
from rest_framework.response import Response
from rest_framework.decorators import action
from django.http import FileResponse
import os

#from rest_framework.viewsets import ViewSet
from rest_framework import status
"""from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from rest_framework import permissions"""

from .utils import get_gif
from .scanner import get_dim
from .models import Pose

"""import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import imageio
from IPython.display import HTML, display
from tqdm import tqdm
import math"""

# view for registering users
class AnalysisView(generics.ListCreateAPIView):
#class AnalysisView(views.APIView):
#class AnalysisView(ViewSet):
    serializer_class = PoseSerializer
    queryset = Pose.objects.all()

    def perform_create(self, serializer):
    #    serializer.save(vid=self.request.kwar)

    #def get_queryset(self):
    #def get(self, request, format=None):

        #return PetSeeker.objects.create(**serializer.validated_data, user=self.request.user)
        print(serializer.validated_data['vid'].file)
        print(serializer.validated_data['vid'])
        g = get_gif(serializer.validated_data['vid'],serializer.validated_data['pic'], serializer.validated_data['height'], serializer.validated_data['slowmo'])
        #s = PoseSerializer
        #save(vid=g)#

        #serializer.save(pic='./pics/output_video.mp4')
        #serializer.save(pic='./pics/output_video.mov')
        serializer.save(pic='./pics/output_video.mp4',kin1='./pics/key_frame_1.png',kin2='./pics/key_frame_2.png',kin3='./pics/key_frame_3.png',kin4='./pics/key_frame_4.png',kin5='./pics/key_frame_5.png', x_vals = g)

        # Construct the response
        response_data = {
            "output_video": './pics/output_video.mp4',
            "key_frames": [
                './pics/key_frame_1.png',
                './pics/key_frame_2.png',
                './pics/key_frame_3.png',
                './pics/key_frame_4.png',
                './pics/key_frame_5.png'
            ],
            "other": g
        }

        return Response(response_data, status=status.HTTP_201_CREATED)
#serializer.save(pic=g)
        #return Response(r'C:\Users\hocke\Desktop\UofT\Third Year\CSC309\moveNet\src\backend\movenet\test.mp4')
        #return Response(r'C:\Users\hocke\Desktop\UofT\Third Year\CSC309\moveNet\src\backend\media\pics\output_video.mp4')


    @action(detail=True, methods=['get'])
    def download(self, request, pk=None):
        instance = self.get_object()
        file_path = instance.pic
        if not os.path.exists(file_path):
            return Response({"message": "File not found"}, status=status.HTTP_404_NOT_FOUND)
        return FileResponse(open(file_path, 'rb'), content_type='video/mov')

"""from rest_framework import generics
from .models import UserProfile
from .serializers import UserProfileSerializer

class UserProfileList(generics.ListCreateAPIView):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer

class UserProfileDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
"""

from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializer, LoginSerializer, UserSerializer,UserProfileSerializer
from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticated
from .models import UserProfile

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer

class LoginView(generics.GenericAPIView):
    permission_classes = (AllowAny,)
    serializer_class = LoginSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data
        refresh = RefreshToken.for_user(user)
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        })

class UserProfileView(generics.RetrieveAPIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = UserSerializer

    def get_object(self):
        #user = self.request.user
        #UserProfile.objects.get_or_create(user=user)
        return self.request.user


class UserDetailView(generics.RetrieveUpdateAPIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = UserProfileSerializer

    def get_object(self):
        user = self.request.user
        profile, created = UserProfile.objects.get_or_create(user=user)
        return profile

    def put(self, request, *args, **kwargs):
        profile = self.get_object()
        serializer = self.get_serializer(profile, data=request.data, partial=True)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def post(self, request, *args, **kwargs):
        profile = self.get_object()
        leg_len = get_dim(request.data['img'],request.data['leg'])
        #serializer = self.get_serializer(profile, data=request.data)
        return Response(leg_len, status=status.HTTP_200_OK)

        #if serializer.is_valid():
        #    serializer.save()
        #    return Response(serializer.data, status=status.HTTP_200_OK)
        #return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)