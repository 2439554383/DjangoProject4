import base64
import json
import os
import tempfile

from async_tasks import AsyncTask
from bottle import Bottle, request, response

from .face import load_models, swap_face

app = Bottle()

# https://github.com/bottlepy/bottle/issues/881#issuecomment-244024649
app.plugins[0].json_dumps = lambda *args, **kwargs: json.dumps(
    *args, ensure_ascii=False, **kwargs
).encode("utf8")


# Enable CORS
@app.hook("after_request")
def enable_cors():
    response.set_header("Access-Control-Allow-Origin", "*")
    response.set_header("Access-Control-Allow-Methods", "*")
    response.set_header("Access-Control-Allow-Headers", "*")


@app.route("<path:path>", method=["GET", "OPTIONS"])
def handle_options(path):
    response.status = 200
    return "MagicMirror ✨"


@app.get("/status")
def status():
    return {"status": "running"}


@app.post("/prepare")
def prepare():
    return {"success": load_models()}


def save_base64_to_tempfile(base64_str, suffix=".jpg"):
    img_data = base64.b64decode(base64_str)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(img_data)
    tmp_file.close()
    return tmp_file.name

@app.post("/task")
def create_task():
    try:
        task_id = request.json["id"]
        input_image_b64 = request.json["inputImage"]
        target_face_b64 = request.json["targetFace"]
        assert all([task_id, input_image_b64, target_face_b64])

        input_image_path = save_base64_to_tempfile(input_image_b64)
        target_face_path = save_base64_to_tempfile(target_face_b64)

        output_path, _ = AsyncTask.run(
            lambda: swap_face(input_image_path, target_face_path), task_id=task_id
        )

        os.unlink(input_image_path)
        os.unlink(target_face_path)

        if output_path is None:
            response.status = 400
            return {"error": "Face swap failed"}

        # 读取输出文件内容，转成base64字符串返回给前端
        with open(output_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # 可选择删除临时输出文件
        os.unlink(output_path)

        return {"result": img_base64}

    except Exception as e:
        response.status = 400
        print(f"Exception in create_task: {e}", flush=True)
        return {"error": "Something went wrong!"}

@app.delete("/task/<task_id>")
def cancel_task(task_id):
    AsyncTask.cancel(task_id)
    return {"success": True}
