from rest_framework import generics,views
from .serializers import PoseSerializer
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet
from rest_framework import status
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from rest_framework import permissions

from .utils import get_gif
import tensorflow as tf
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
import math

# view for registering users
class AnalysisView(generics.ListCreateAPIView):
#class AnalysisView(views.APIView):
#class AnalysisView(ViewSet):
    serializer_class = PoseSerializer



    def perform_create(self, serializer):
    #    serializer.save(vid=self.request.kwar)

    #def get_queryset(self):
    #def get(self, request, format=None):

        #return PetSeeker.objects.create(**serializer.validated_data, user=self.request.user)
        print(serializer.validated_data['vid'].file)
        print(serializer.validated_data['vid'].temporary_file_path())
        g = get_gif(serializer.validated_data['vid'].temporary_file_path())
        #s = PoseSerializer
        #save(vid=g)


        serializer.save(pic='./pics/animation.gif')
        return Response(r'C:\Users\hocke\Desktop\UofT\Third Year\CSC309\moveNet\src\backend\movenet\giffy2.gif')
        '''
        plt.plot(angles, marker='o', linestyle='-')
        plt.title('Elbow Angle Over Frames')
        plt.xlabel('Frame Index')
        plt.ylabel('Elbow Angle (degrees)')
        plt.grid(True)
        plt.show()

        # Detect position changes
        position_changes = []
        for i in range(1, len(ankle_positions)):
            position_change = np.linalg.norm(np.array(ankle_positions[i]) - np.array(ankle_positions[i - 1]))
            if position_change > threshold:
                position_changes.append(i)
        # Visualize the ankle positions and position changes
        ankle_x, ankle_y = zip(*ankle_positions)
        plt.plot(range(len(ankle_y)), ankle_y, marker='o', linestyle='-', label='Ankle Position')

        # Check if there are position changes before trying to unpack
        if position_changes:
            change_x, change_y = zip(*[ankle_positions[i] for i in position_changes])
            # plt.scatter(change_x, change_y, color='red', label='Position Change')
            plt.scatter(range(len(change_y)), change_y, color='red', label='Position Change')

        plt.title('Ankle Position Over Time')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.show()

        print('switches: ' + str(switch))

        return plt'''