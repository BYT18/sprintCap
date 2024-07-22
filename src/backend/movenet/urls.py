from django.urls import path,include
from .views import AnalysisView
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
#from .views import UserProfileList, UserProfileDetail
from .views import RegisterView, LoginView, UserProfileView, UserDetailView

urlpatterns = [
    #newuser
    #path('create/', UserCreateView.as_view(), name='user-create'),
    path('test/', AnalysisView.as_view(), name='user-create'),
    #path('api/profiles/', UserProfileList.as_view(), name='userprofile-list'),
    #path('api/profiles/<int:pk>/', UserProfileDetail.as_view(), name='userprofile-detail'),
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('profile/', UserProfileView.as_view(), name='profile'),
    path('prof/', UserDetailView.as_view(), name='profile'),
]
