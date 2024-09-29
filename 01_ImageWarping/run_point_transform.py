import cv2
import numpy as np
import gradio as gr
import copy
from scipy.interpolate import Rbf

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    h, w = image.shape[:2]
    warped_image = np.zeros_like(image)
    n = len(source_pts)

    
    src_x, src_y = target_pts[:, 1], target_pts[:, 0]
    dst_x, dst_y = source_pts[:, 1], source_pts[:, 0]



    for i in range(h):
        for j in range(w):
            v_x, v_y = i, j

            
            if np.any(np.all(target_pts == [v_y, v_x], axis=1)):
                idx = np.where(np.all(target_pts == [v_y, v_x], axis=1))[0][0]
                warped_image[v_x, v_y, :] = image[dst_x[idx], dst_y[idx], :]
                continue

            wi = 1 / np.linalg.norm(np.array([src_x - v_x, src_y - v_y]), axis=0)**2
            total_weight = np.sum(wi)

            p_ave_x = np.sum(wi * src_x) / total_weight
            p_ave_y = np.sum(wi * src_y) / total_weight
            q_ave_x = np.sum(wi * dst_x) / total_weight
            q_ave_y = np.sum(wi * dst_y) / total_weight

            src_hat_x = src_x - p_ave_x
            src_hat_y = src_y - p_ave_y
            dst_hat_x = dst_x - q_ave_x
            dst_hat_y = dst_y - q_ave_y

            A = np.array([wi[k] * np.dot(np.array([[src_hat_x[k], src_hat_y[k]], 
                                          [src_hat_y[k], -src_hat_x[k]]]), 
                                         np.array([[v_x - p_ave_x, v_y - p_ave_y], 
                                          [v_y - p_ave_y, p_ave_x - v_x]]).T)
                          for k in range(n)])

            v_hat = np.sum([np.dot([dst_hat_x[k], dst_hat_y[k]], A[k]) for k in range(n)], axis=0)
            v_hat = np.linalg.norm([v_x - p_ave_x, v_y - p_ave_y]) / np.linalg.norm(v_hat + eps) * v_hat + [q_ave_x, q_ave_y]
            v_hat = np.clip(np.round(v_hat), [0, 0], [h - 1, w - 1]).astype(int)

            warped_image[v_x, v_y, :] = image[v_hat[0], v_hat[1], :]

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
