from django.urls import path,include
from .views import AnalysisView
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
#from .views import UserProfileList, UserProfileDetail
from .views import RegisterView, LoginView, UserProfileView, UserDetailView, UserProfileCreateView

urlpatterns = [
    #newuser
    #path('create/', UserCreateView.as_view(), name='user-create'),
    path('api/test/', AnalysisView.as_view(), name='user-create'),
    #path('api/profiles/', UserProfileList.as_view(), name='userprofile-list'),
    #path('api/profiles/<int:pk>/', UserProfileDetail.as_view(), name='userprofile-detail'),
    path('api/register/', RegisterView.as_view(), name='register'),
    path('api/login/', LoginView.as_view(), name='login'),
    path('api/profile/', UserProfileView.as_view(), name='profile'),
    path('api/prof/', UserDetailView.as_view(), name='profile'),
    path('api/create/', UserProfileCreateView.as_view(), name='create'),
]
