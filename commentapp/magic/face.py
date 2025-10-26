import base64
import os
import traceback
from functools import lru_cache
import cv2
import numpy as np
from tinyface import TinyFace

_tf = TinyFace()


def load_models():
    print("load_models start", flush=True)
    try:
        _tf.config.face_detector_model = _get_model_path("scrfd_2.5g.onnx")
        _tf.config.face_embedder_model = _get_model_path("arcface_w600k_r50.onnx")
        _tf.config.face_swapper_model = _get_model_path("inswapper_128_fp16.onnx")
        _tf.config.face_enhancer_model = _get_model_path("gfpgan_1.4.onnx")
        print("scrfd_2.5g.onnx path:", _get_model_path("scrfd_2.5g.onnx"))
        _tf.prepare()
        return True
    except BaseException as e:
        print(e)
        traceback.print_exc()
        return False

def file_to_cv2_image(file_obj):
    # 读取上传的文件内容
    file_bytes = file_obj.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img
def swap_face_from_cv2(input_img, face_img):
    try:
        reference_face = _tf.get_one_face(input_img)
        destination_face = _tf.get_one_face(face_img)

        result_img = _tf.swap_face(
            vision_frame=input_img,
            reference_face=reference_face,
            destination_face=destination_face,
        )

        # 保存图片到文件
        cv2.imwrite("/www/wwwroot/DjangoProject4/result.jpg", result_img)

        # 转成 base64 返回
        _, buffer = cv2.imencode(".jpg", result_img)
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        print(f"result_base64:{result_base64}")
        return result_base64
    except Exception as e:
        print("换脸失败:", e)
        traceback.print_exc()
        return None
@lru_cache(maxsize=12)
def swap_face(input_path, face_path):
    try:
        save_path = _get_output_file_path(input_path)
        output_img = _swap_face(input_path, face_path)
        _write_image(save_path, output_img)
        return save_path
    except BaseException as _:
        return None


@lru_cache(maxsize=12)
def _swap_face(input_path, face_path):
    return _tf.swap_face(
        vision_frame=_read_image(input_path),
        reference_face=_get_one_face(input_path),
        destination_face=_get_one_face(face_path),
    )


@lru_cache(maxsize=12)
def _get_one_face(face_path: str):
    face_img = _read_image(face_path)
    return _tf.get_one_face(face_img)


@lru_cache(maxsize=12)
def _read_image(img_path: str):
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)


def _write_image(img_path: str, img):
    suffix = os.path.splitext(img_path)[-1]
    cv2.imencode(suffix, img)[1].tofile(img_path)


def _get_output_file_path(file_name):
    base_name, ext = os.path.splitext(file_name)
    return base_name + "_output" + ext


def _get_model_path(file_name: str):
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "models", file_name)
    )
