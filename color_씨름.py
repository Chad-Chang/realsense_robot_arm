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

lower_green = (50, 255, 60)
upper_green = (70, 255, 255)
lower_blue = (97, 100, 31)
upper_blue = (117, 255, 255)

centeRx_g = 0  # 도심값
centeRy_g = 0
centeRx_b = 0
centeRy_b = 0
ang = 85
maxArea_g = 0

contour_update_g = 0  # 2개의 컨투어중 큰거를 물체로 감지
contour_update_r = 0  # 2개의 컨투어중 큰거를 물체로 감지
contour_update_b = 0  # 2개의 컨투어중 큰거를 물체로 감지
dir_g = "0"  # 카메리 어느쪽에 물체가 있는지 (카메라 상에서 물체의 위치만 => 서보모터 돌리는 가이드)
vec_g = "0"  # 뉴클레오에 통신해주는 방향 (서보모터 각도를 기준으로 보내주는 값)
isObject_g = "0"  # 물체 있는지 없는지 0 = false, 1 = true, 'L' : 왼쪽에서 사라졌다. , 'R' : 오른족에서 사라졌다.
search = True
start_game = False
size_g = "0"  # 물체의 크기
inc2 = 5  # 물체가 사라졌을때의 서보 증분값
inc = 0
count = 0
isBlue = '0'
area_b = 0

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
        img_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        img_mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
        img_mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

        # depth_colormap = cv2.applColorMap(depth_image, cv2.COLORMAP_JET)

        # depth_frame은 거리 정보를 담고 있는데 get_data()를 하고 numpyarray로 변환하면 각 원소의 범위가 0~255가 아니게 됨. + 음수가 나올수 잇음.
        # 그래서 scale 변환을 해줘야함.                                                  ㄱ
        test = cv2.convertScaleAbs(depth_image, alpha=0.03)
        test2 = np.where(test > 0, 150, 0)
        # print(test2.shape)
        cv2.imshow('asdef',test)
        # print(np.where(test>0,255,0))
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),cv2.COLORMAP_JET)
        kernel = np.ones((5, 5), np.uint8)  # 모폴로지  노이즈 필터링
        img_mask_green = cv2.morphologyEx(img_mask_green, cv2.MORPH_OPEN, kernel)
        img_mask_green = cv2.morphologyEx(img_mask_green, cv2.MORPH_CLOSE, kernel)
        contours_g, _ = cv2.findContours(img_mask_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # 반환값 계층 정보

        for i, _ in enumerate(contours_g):  # 배열이름만 range부분에 넣으면 인덱스 추출
            cnt = contours_g[i]
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            cx = int(x + w / 2)  # x중심 위치
            cy = int(y + h / 2)  # y중심 위치
            if area > maxArea_g:  # 가장 큰 물체를 업데이트 해줌.
                maxArea_g = area
                maxCx_g = cx
                maxCy_g = cy
                Rx_g = x
                Ry_g = y
                Rw_g = w
                Rh_g = h
                cv2.circle(color_image, (maxCx_g, maxCy_g), 10, (0, 0, 255), 2)
                print("g")

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
