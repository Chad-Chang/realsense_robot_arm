import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline() # pipeline클래스는 user interaction with the device가 잘 이루어지게 만들어짐.
# 복잡성 device/computer vision module을 단순화함. -> user/application에 집중할 수 있음.
# 하나의 블록 interface로 구성되어있음.
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
# argument: streaming type, width, height, format, framerate

if device_product_line == "L500": # 이게 뭔지
    config.enable_stream(rs.stream.color, 960,540,rs.format.bgr8,30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        print("frame",frames)
        depth_frame = frames.get_depth_frame()# depth frame 객체
        color_frame = frames.get_color_frame()# color frame 객체
        print("depth frame_data : ",depth_frame.get_data())
        print("color frame_data : ",color_frame.get_data())
        # depth_frame.get_data()했을때 depth_frame의 데이터 정보는 bufData에 있다.
        if not depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data()) # frame데이터를 행렬화 시켜줌.
        color_image = np.asanyarray(color_frame.get_data())
        # depth_colormap = cv2.applColorMap(depth_image, cv2.COLORMAP_JET)

        # depth_frame은 거리 정보를 담고 있는데 get_data()를 하고 numpyarray로 변환하면 각 원소의 범위가 0~255가 아니게 됨. + 음수가 나올수 잇음.
        # 그래서 scale 변환을 해줘야함.                                                  ㄱ
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),cv2.COLORMAP_JET)
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image,dsize = (depth_colormap_dim[1],depth_colormap_dim[0]),interpolation = cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # 나가기
            cv2.destroyAllWindows()  # 윈도우 제거
            break

finally:
    pipeline.stop()
