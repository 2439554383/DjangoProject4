
video_url = "https://api-play.amemv.com/aweme/v1/play/?video_id=v1e00fgi0000d0r3v7nog65j7p35b6u0&line=0&file_id=8215f21847114e6c9998e4029b210202&sign=a5b5958c285e63e1ae6b77062f3ebf97&is_play_url=1&source=PackSourceEnum_AWEME_DETAIL"

with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
    cmd = [
        "ffmpeg",
        "-y",
        "-t", "30",
        "-i", video_url,       # 🔁 直接从 URL 流式读
        "-vn",                 # 去掉视频
        "-ar", "16000",        # Whisper 推荐采样率
        "-ac", "1",            # 单声道
        "-f", "wav",           # 输出格式
        tmp_audio.name
    ]

    subprocess.run(cmd, check=True)
    url = "https://api.siliconflow.cn/v1/audio/transcriptions"
    token = "sk-ebbcpdkopzbtwmeyedlqepvdeppbbkpgllhqudjuolvirxru"
    headers = {
        "Authorization": f"Bearer {token}",
        # 不需要指定 Content-Type，requests 会自动添加正确的 multipart boundary
    }
    data = {
        "model": "FunAudioLLM/SenseVoiceSmall",
    }
    with open(tmp_audio.name, "rb") as f:
        files = {
            "file": (os.path.basename(tmp_audio.name), f, "audio/wav")
        }
        response = requests.post(url, headers=headers, data=data, files=files)
        result = json.loads(response.text)
    print(result['text'])