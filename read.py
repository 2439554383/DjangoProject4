import json
import subprocess
import tempfile
import pymupdf
from docx import Document
from django.http import JsonResponse, StreamingHttpResponse
from django.conf import settings
import pandas as pd
import requests
import whisper
import requests
import yt_dlp
from django.http import JsonResponse, FileResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.db import connection
import whisper
import clip
import torch
from PIL import Image
import ffmpeg
from moviepy import VideoFileClip
from torch._dynamo.config import base_dir
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import re
import os
import requests
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from openai import OpenAI
import requests
import base64
from commentapp.views import sanitize_filename1
def get_response(start_url):
    dataurl = start_url
    if "douyin" in dataurl:
        type_text = "抖音"
        url = "http://api.moreapi.cn/api/douyin/aweme_detail"
    elif "kuaishou" in dataurl:
        type_text = "快手"
        url = "http://api.moreapi.cn/api/ks/aweme_detail"
    elif "xhslink" in dataurl:
        type_text = "小红书"
        url = "http://api.moreapi.cn/api/xhs/note_detail"
    elif "toutiao" in dataurl:
        type_text = "头条"
        url = "http://api.moreapi.cn/api/toutiao/aweme_detail_v2"
    # elif "b23" in dataurl:
    #     type_text = "哔哩哔哩"
    # elif "weibo" in dataurl:
    #     type_text = "微博"
    #     url = "http://api.moreapi.cn/api/weibo/post_detail"
    # elif "xigua" in dataurl:
    #     type_text = "西瓜视频"
    #     url = "http://api.moreapi.cn/api/xigua/aweme_detail"
    print(type_text)
    pattern = r'https?://[^\s，。、！？“”‘’<>（）【】()"\']+'
    match = re.search(pattern, dataurl)
    if match:
        video_url = match.group()
        print("提取到的视频地址:", video_url)
    else:
        print("未找到视频地址")
    payload = json.dumps({
        "aweme_id": "",
        "share_text": video_url,
        "proxy": ""
    })
    headers = {
        "Authorization": "Bearer O1Y4f9r8sbNdbSqzmpb5MUk3jMS98Hs6exTLosz8bYK0SQyyiQS6nlV2kDDVMghX",
        'Content-Type': 'application/json'
    }
    print(url)
    response = requests.request("POST", url, headers=headers, data=payload)
    text_s = json.loads(response.text)
    print(text_s)
    print(json.dumps(text_s, indent=2, ensure_ascii=False))
    return text_s,type_text

st_url = 'https://weibo.com/2003516887/5177243451263570'
text_s , type = get_response(st_url)
if(type =='抖音'):
    v_url = text_s['data']['aweme_detail']['video']['play_addr']['url_list'][2]
    cover = text_s['data']['aweme_detail']['video']['cover']['url_list'][1]
    print(v_url)
    print(cover)
    title_name = text_s['data']['aweme_detail']['desc']
    title = sanitize_filename1(title_name)
    print(title_name)
elif (type == '小红书'):
    v_url = text_s['data']['response_body']['data']['items'][0]['note_card']['video']['media']['stream']['h264'][0]['master_url']
    cover = text_s['data']['response_body']['data']['items'][0]['note_card']['image_list'][0]['info_list'][0]['url']
    print(v_url)
    print(cover)
    title_name = text_s['data']['response_body']['data']['items'][0]['note_card']['title']
    desc_name = text_s['data']['response_body']['data']['items'][0]['note_card']['desc']
    title = sanitize_filename1(title_name)
    desc = sanitize_filename1(desc_name)
    print(title)
    print(desc)
elif (type == '快手'):
    v_url = text_s['data'][0]['manifest']['adaptationSet'][0]['representation'][0]['url']
    cover = text_s['data'][0]['coverUrls'][0]['url']
    print(v_url)
    print(cover)
    title_name = text_s['data'][0]['caption']
    title = sanitize_filename1(title_name)
    print(title_name)
elif (type == '头条'):
    v_url = text_s['data']['data']['video']['play_addr']['url_list'][0]
    cover = text_s['data']['data']['video']['origin_cover']['url_list'][0]
    print(v_url)
    print(cover)
    title_name = text_s['data']['data']['title']
    title = sanitize_filename1(title_name)
    print(title_name)


