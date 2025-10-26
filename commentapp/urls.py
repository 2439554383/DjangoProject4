from django.contrib import admin
from django.urls import path, include
import commentapp
from commentapp.views import get_comment, get_listmodel, switch_isset, get_hospitallist, get_filelist, get_image, \
    get_text, post_content, post_aiface, get_aiimage, get_unmarkvideo, post_audio, get_code, extract_voice_from_video, \
    query_voice_separation_task

urlpatterns = [
    path('get_comment/',get_comment),
    path('get_listmodel/', get_listmodel),
    path('switch_isset/', switch_isset),
    path('get_hospitallist/', get_hospitallist),
    path('get_filelist/', get_filelist),
    path('get_image/', get_image),
    path('get_text/', get_text),
    path('get_code/', get_code),
    path('post_content/', post_content),
    path('post_aiface/', post_aiface),
    path('get_aiimage/', get_aiimage),
    path('get_unmarkvideo/', get_unmarkvideo),
    path('post_audio/', post_audio),
    path('extract_voice/', extract_voice_from_video),  # 新增人声分离接口
    path('query_voice_task/', query_voice_separation_task),  # 查询人声分离任务状态接口
]