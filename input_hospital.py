# elif (type == '哔哩哔哩'):
#     short_url = "video_url"
#     response = requests.head(short_url, allow_redirects=True)
#     print("重定向后的真实地址:", response.url)
#     match = re.search(r'/video/(BV[0-9A-Za-z]+)', response.url)
#     if match:
#         bvid = match.group(1)
#         print("BVID:", bvid)
#     else:
#         print("未找到 BVID")
#     url = "http://api.moreapi.cn/api/bilibili/video_data"
#
#     payload = json.dumps({
#         "bvid": bvid,
#         "proxy": ""
#     })
#     headers = {
#         "Authorization": "Bearer O1Y4f9r8sbNdbSqzmpb5MUk3jMS98Hs6exTLosz8bYK0SQyyiQS6nlV2kDDVMghX",
#         'Content-Type': 'application/json'
#     }
#     response = requests.request("POST", url, headers=headers, data=payload)
#     text_s = json.loads(response.text)
#     print(json.dumps(text_s, indent=2, ensure_ascii=False))
#     v_url = text_s['data']['aweme_detail']['video']['play_addr']['url_list'][2]
#     cover = text_s['data']['aweme_detail']['video']['cover']['url_list'][1]
#     print(v_url)
#     print(cover)
#     title_name = text_s['data']['aweme_detail']['desc']
#     title = sanitize_filename1(title_name)
#     print(title_name)
#
#
#
#
# # ✅sh api解析视频地址
# # ✅ake api解析视频地址
# # key = "6849cc2c1206b1978Wt433"
# # response = requests.get(f"https://watermark-api.hlyphp.top/Live/Index?appid={key}&link={video_url}")
# # if response.status_code == 200:
# #     print(response.text)
# #     data = json.loads(response.text)
# #     url = data['data']['videoSrc']
# #     cover = data['data']['imageSrc']
# #     title_name = data['data']['title']
# #     title = sanitize_filename1(title_name)
# #     print("url:", url)
# #     print("cover:", cover)
# #     print("title:", cover)
# # ✅more api解析视频地址
#
#
# elif (type == '西瓜视频'):
#     v_url = text_s['data']['aweme_detail']['video']['play_addr']['url_list'][2]
#     cover = text_s['data']['aweme_detail']['video']['cover']['url_list'][1]
#     print(v_url)
#     print(cover)
#     title_name = text_s['data']['aweme_detail']['desc']
#     title = sanitize_filename1(title_name)
#     print(title_name)