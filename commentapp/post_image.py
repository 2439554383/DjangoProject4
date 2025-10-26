import base64
import uuid
import requests

def image_to_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# 编码两张图片
input_image_base64 = image_to_base64("face.jpg")
target_face_base64 = image_to_base64("source.jpg")

# 唯一任务ID
task_id = str(uuid.uuid4())

# 先加载模型（必须调用一次 prepare）
prepare_res = requests.post("http://localhost:8050/prepare")
print("模型加载:", prepare_res.json())

# 发送换脸请求
payload = {
    "id": task_id,
    "inputImage": input_image_base64,
    "targetFace": target_face_base64
}
res = requests.post("http://localhost:8050/task", json=payload)

if res.status_code == 200 and 'result' in res.json():
    # 解码并保存换脸后的图像
    result_base64 = res.json()['result']
    with open("result1.jpg", "wb") as f:
        f.write(base64.b64decode(result_base64))
    print("✅ 换脸完成，结果保存在 result1.jpg")
else:
    print("❌ 请求失败")
    print(res.status_code, res.text)
