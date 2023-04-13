import pyrealsense2 as rs
import numpy as np
import cv2

def mouse_callback(event, x, y, flags, param):
    global hsv1, flag1
    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 hsv1로 변환합니다.
    if event == cv2.EVENT_LBUTTONDOWN:
        color = resized_color_image[y, x]
        depth = depth_image[y,x]
        one_pixel_color = np.uint8([[color]])  # numpy array 로 변환  3차원 배열로 변환
        one_pixel_depth = np.uint8([[depth]])  # numpy array 로 변환  3차원 배열로 변환
        print(one_pixel_color, one_pixel_depth)
        flag1 = 1

cv2.namedWindow('RealSense_color')
cv2.setMouseCallback('RealSense_color', mouse_callback)
cv2.namedWindow('RealSense_depth')
cv2.setMouseCallback('RealSense_depth', mouse_callback)

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline) # pipeline을 랜더링하기위해 알맞은 형태로 변환
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
found_rgb = False
for s in device.sensors:
# s => pipeline의 device정보 객체임.
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)
config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)

if device_product_line == "L500": # 이게 뭔지
    config.enable_stream(rs.stream.color, 960,540,rs.format.bgr8,30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# cv2.namedWindow('img_color')
try:
    while True:
        frames = pipeline.wait_for_frames()
        # print("frame",frames)
        depth_frame = frames.get_depth_frame()# depth frame 객체
        color_frame = frames.get_color_frame()# color frame 객체
        # print("depth frame_data : ",depth_frame.get_data())
        # print("color frame_data : ",color_frame.get_data())
        # depth_frame.get_data()했을때 depth_frame의 데이터 정보는 bufData에 있다.
        if not depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data()) # frame데이터를 행렬화 시켜줌.
        color_image = np.asanyarray(color_frame.get_data())

        print(depth_image.shape)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),cv2.COLORMAP_JET)
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        resized_color_image = cv2.resize(color_image,dsize = (depth_colormap_dim[1],depth_colormap_dim[0]),interpolation = cv2.INTER_AREA)
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image,dsize = (depth_colormap_dim[1],depth_colormap_dim[0]),interpolation = cv2.INTER_AREA)
        #     images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense_color', resized_color_image)
        cv2.imshow('RealSense_depth', depth_colormap)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # 나가기
            cv2.destroyAllWindows()  # 윈도우 제거
            break

finally:
    pipeline.stop()
