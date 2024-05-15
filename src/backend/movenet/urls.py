from django.urls import path
from .views import AnalysisView
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    #newuser
    #path('create/', UserCreateView.as_view(), name='user-create'),
    path('test/', AnalysisView.as_view(), name='user-create'),

]
