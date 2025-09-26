import time
import os

import cv2
import numpy as np
from PIL import Image

from segformer import SegFormer_Segmentation


if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   初始化SegFormer模型（PyTorch版本）
    # -------------------------------------------------------------------------#
    segformer = SegFormer_Segmentation()
    # ----------------------------------------------------------------------------------------------------------#
    #   模式设置为"video"以处理视频，"image"处理单张图片
    # ----------------------------------------------------------------------------------------------------------#
    mode = "image"
    # -------------------------------------------------------------------------#
    #   分类参数
    # -------------------------------------------------------------------------#
    count = False
    name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                    "tvmonitor"]
    # ----------------------------------------------------------------------------------------------------------#
    #   视频参数配置
    # -------------------------------------------------------------------------#
    video_path = "img/1.mp4"  # 输入视频路径（可替换为0调用摄像头）
    video_save_path = "result.mp4"  # 输出视频保存路径（为空则不保存）
    video_fps = 25.0  # 保存视频的帧率
    # ----------------------------------------------------------------------------------------------------------#
    #   图片参数配置
    # ----------------------------------------------------------------------------------------------------------#
    image_save_path = "result.jpg"  # 处理后图片保存路径
    # ----------------------------------------------------------------------------------------------------------#
    #   背景替换参数配置
    # ----------------------------------------------------------------------------------------------------------#
    background_image_path = "background.jpg"  # 背景图片路径
    replace_background = True  # 是否启用背景替换
    # 选择需要保留的前景类别（这里保留人主要前景）
    foreground_classes = [15]  # person
    # ----------------------------------------------------------------------------------------------------------#
    #   其余参数保持默认（无需修改）
    # -------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    simplify = True
    onnx_save_path = "model_data/models.onnx"
    background_img = None  # 初始化背景图片变量

    # 新增：图片处理模式
    if mode == "image":
        # 加载背景图片
        if replace_background:
            try:
                background_image_path = input('Input background filename:')
                background_img = cv2.imread(background_image_path)
                if background_img is None:
                    raise FileNotFoundError(f"无法加载背景图片: {background_image_path}")
                print(f"成功加载背景图片: {background_image_path}")
            except Exception as e:
                print(f"加载背景图片失败: {str(e)}")
                replace_background = False  # 禁用背景替换

        # 输入图片路径
        image_path = input('Input image filename:')
        try:
            # 读取图片
            frame = cv2.imread(image_path)
            if frame is None:
                raise FileNotFoundError(f"无法加载图片: {image_path}")

            # 格式转换：BGR→RGB（适配PIL）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(np.uint8(frame_rgb))

            # 模型推理：获取掩码
            _, mask = segformer.detect_image(frame_pil, count=False)
            processed_frame = frame_rgb  # 使用原始图像作为前景

            # 背景替换处理
            if replace_background and background_img is not None:
                # 调整背景图片尺寸以匹配输入图片
                background_img = cv2.resize(background_img, (frame.shape[1], frame.shape[0]))

                # 创建前景掩码
                foreground_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
                for cls in foreground_classes:
                    foreground_mask[mask == cls] = 255

                # 对掩码进行模糊处理，使边缘更自然
                foreground_mask = cv2.GaussianBlur(foreground_mask, (15, 15), 0) / 255.0
                foreground_mask = np.expand_dims(foreground_mask, axis=-1)

                # 格式转换：RGB→BGR（适配OpenCV）
                processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

                # 融合前景和新背景
                result_frame = (processed_frame_bgr * foreground_mask +
                                background_img * (1 - foreground_mask)).astype(np.uint8)
            else:
                # 不替换背景时直接转换格式
                result_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

            # 保存结果图片
            cv2.imwrite(image_save_path, result_frame)
            print(f"图片处理完成，已保存至: {image_save_path}")

            # 显示结果图片
            cv2.imshow("result", result_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f'处理图片时出错: {str(e)}')





    elif mode == "video":

        while True:
            # 加载背景图片
            if replace_background:
                try:
                    background_image_path = input('Input background filename (or 0 for camera):')
                    background_img = cv2.imread(background_image_path)
                    if background_img is None:
                        raise FileNotFoundError(f"无法加载背景图片: {background_image_path}")
                    print(f"成功加载背景图片: {background_image_path}")
                except Exception as e:
                    print(f"加载背景图片失败: {str(e)}")
                    replace_background = False  # 禁用背景替换

            # 循环输入视频路径
            video_path = input('Input video filename (or 0 for camera):')
            # 处理摄像头输入（0）
            if video_path.strip() == '0':
                video_path = 0
            try:
                # 尝试打开视频
                capture = cv2.VideoCapture(video_path)
                if not capture.isOpened():
                    raise ValueError("无法打开视频文件或摄像头")
            except Exception as e:
                print(f'Open Error! {str(e)} Try again!')
                continue

            # 替换原视频保存部分的编码设置
            if video_save_path != "":
                # 使用MP4兼容的编码格式
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 替换 'XVID' 为 'mp4v'
                size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
                # 调整背景图片尺寸以匹配视频
                if replace_background and background_img is not None:
                    background_img = cv2.resize(background_img, size)

            ref, frame = capture.read()
            if not ref:
                raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

            fps = 0.0
            print("视频处理中，按ESC键退出...")
            while (True):
                t1 = time.time()
                ref, frame = capture.read()
                if not ref:
                    break
                # 格式转换：BGR→RGB（适配PIL）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(np.uint8(frame_rgb))
                # 模型推理：只需要掩码（mask），不需要带粉色的processed_frame
                _, mask = segformer.detect_image(frame_pil, count=False)  # 只取掩码
                processed_frame = frame_rgb  # 直接用原始RGB图像
                # 背景替换处理
                if replace_background and background_img is not None:
                    # 创建前景掩码（只保留指定类别的区域）
                    foreground_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
                    for cls in foreground_classes:
                        foreground_mask[mask == cls] = 255
                    # 对掩码进行模糊处理，使边缘更自然
                    foreground_mask = cv2.GaussianBlur(foreground_mask, (15, 15), 0) / 255.0
                    foreground_mask = np.expand_dims(foreground_mask, axis=-1)
                    # 格式转换：RGB→BGR（适配OpenCV显示）
                    processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    # 融合前景和新背景
                    frame = (processed_frame_bgr * foreground_mask +
                             background_img * (1 - foreground_mask)).astype(np.uint8)
                else:
                    # 不替换背景时直接转换格式
                    frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                # 计算FPS
                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %.2f" % (fps))
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
                # 显示视频
                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                # 保存视频
                if video_save_path != "":
                    out.write(frame)
                # 按ESC退出
                if c == 27:
                    capture.release()
                    break
            print("Video Detection Done!")
            capture.release()
            if video_save_path != "":
                print("Save processed video to the path :" + video_save_path)
                out.release()
            cv2.destroyAllWindows()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'image', 'fps' or 'dir_predict'.")