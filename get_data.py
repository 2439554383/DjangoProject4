import requests
import json

url = "https://api.yuk0.com/vedio_group/?url=https://v.douyin.com/20V8gAEx_IU/"

response = requests.get(url)
print(response)
text_s = json.loads(response.text)
print(text_s)