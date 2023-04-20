import pyrealsense2 as rs
import numpy as np
import cv2


hsv1 = 0
lower1 = 0
upper1 = 0
lower2 = 0
upper2 = 0
lower3 = 0
upper3 = 0
flag1 = 0
state_data = "0"
def nothing(x):
    pass
def mouse_callback(event, x, y, flags, param):
    global hsv1, flag1
    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 hsv1로 변환합니다.
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(resized_color_image[y, x])
        color = resized_color_image[y, x]
        # print(x, ',', y)
        # print(param)
        one_pixel = np.uint8([[color]])  # numpy array 로 변환  3차원 배열로 변환
        # print(one_pixel)
        hsv1 = cv2.cvtColor(one_pixel, cv2.COLOR_BGR2HSV)  # [[[a b c]]] 3차원이지만 1차원에 1행 밖에 없는 행렬 생성 ==> 형식 맞춰주기 위해 hsv1[0][0]
        hsv1 = hsv1[0][0]  # 색값
        print(hsv1)
        # print(hsv1)
        flag1=1

    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 hsv1로 변환합니다.
    # if event == cv2.EVENT_LBUTTONDOWN:
    #     color = resized_color_image[y, x]
    #     depth = depth_image[y,x]
    #     one_pixel_color = np.uint8([[color]])  # numpy array 로 변환  3차원 배열로 변환
    #     one_pixel_depth = np.uint8([[depth]])  # numpy array 로 변환  3차원 배열로 변환
    #     print(one_pixel_color, one_pixel_depth)
    #     flag1 = 1

cv2.namedWindow('resized_color_image')
cv2.setMouseCallback('resized_color_image', mouse_callback)
# cv2.namedWindow('RealSense_depth')
# cv2.setMouseCallback('RealSense_depth', mouse_callback)
cv2.namedWindow('img_result')
cv2.createTrackbar('thresholdLs1', 'img_result', 0, 255, nothing)
cv2.setTrackbarPos('thresholdLs1', 'img_result', 30)
cv2.createTrackbar('thresholdUs1', 'img_result', 0, 255, nothing)
cv2.setTrackbarPos('thresholdUs1', 'img_result', 30)
cv2.createTrackbar('thresholdLv1', 'img_result', 0, 255, nothing)
cv2.setTrackbarPos('thresholdLv1', 'img_result', 30)
cv2.createTrackbar('thresholdUv1', 'img_result', 0, 255, nothing)
cv2.setTrackbarPos('thresholdUv1', 'img_result', 30)



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

try:
    while True:
        if flag1 == 1:
            thresholdLs1 = cv2.getTrackbarPos('thresholdLs1', 'img_result')  # Ls
            thresholdLv1 = cv2.getTrackbarPos('thresholdLv1', 'img_result')
            thresholdUs1 = cv2.getTrackbarPos('thresholdUs1', 'img_result')  # Us
            thresholdUv1 = cv2.getTrackbarPos('thresholdUv1', 'img_result')  # Uv`

            if hsv1[0] < 10:  # 빨간색
                lower1 = np.array([hsv1[0] - 10 + 180, thresholdLs1, thresholdLv1])
                upper1 = np.array([180, thresholdUs1, thresholdUv1])
                lower2 = np.array([0, thresholdLs1, thresholdLv1])
                upper2 = np.array([hsv1[0], thresholdUs1, thresholdUv1])
                lower3 = np.array([hsv1[0], thresholdLs1, thresholdLv1])
                upper3 = np.array([hsv1[0] + 10, thresholdUs1, thresholdUv1])

            elif hsv1[0] > 170:  # 빨간색
                # print("case2")
                lower1 = np.array([hsv1[0], thresholdLs1, thresholdLv1])
                upper1 = np.array([180, thresholdUs1, thresholdUv1])
                lower2 = np.array([0, thresholdLs1, thresholdLv1])
                upper2 = np.array([hsv1[0] + 10 - 180, thresholdUs1, thresholdUv1])
                lower3 = np.array([hsv1[0] - 10, thresholdLs1, thresholdLv1])
                upper3 = np.array([hsv1[0], thresholdUs1, thresholdUv1])

            else:
                # print("case3")                     # 그 밖의 부분은 경계를 구분할 필요가 없음.
                lower1 = np.array([hsv1[0] - 10, thresholdLs1, thresholdLv1])
                upper1 = np.array([hsv1[0] + 10, thresholdUs1, thresholdUv1])

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()# depth frame 객체
        color_frame = frames.get_color_frame()# color frame 객체
        # print("depth frame_data : ",depth_frame.get_data())
        # print("color frame_data : ",color_frame.get_data())
        # depth_frame.get_data()했을때 depth_frame의 데이터 정보는 bufData에 있다.
        if not depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data()) # frame데이터를 행렬화 시켜줌.
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                         interpolation=cv2.INTER_AREA)
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image,dsize = (depth_colormap_dim[1],depth_colormap_dim[0]),interpolation = cv2.INTER_AREA)
        #     images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        img_hsv1 = cv2.cvtColor(resized_color_image, cv2.COLOR_BGR2HSV)
        img_mask1 = cv2.inRange(img_hsv1, lower1, upper1)  # 범위가 음수인 경우는 재지정해줘야하기 때문에 범위를 3개로 나눔
        # print(np.where(resized_color_image>255))
        # print(lower1, lower2)
        img_mask2 = cv2.inRange(img_hsv1, lower2, upper2)  # 범위 지정 이진화 함수
        img_mask3 = cv2.inRange(img_hsv1, lower3, upper3)
        img_mask = img_mask1 | img_mask2 | img_mask3

        kernel = np.ones((5, 5), np.uint8)  # 모폴로지
        img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
        img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)

        img_result = cv2.bitwise_and(color_image, color_image, mask=img_mask)
        numOfLabels, _, stats, centroids = cv2.connectedComponentsWithStats(img_mask)
        max_area = 0
        max_centerX = 0
        max_centerY = 0
        for idx, centroid in enumerate(centroids):  # 한 row씩만 뽑아줌
            if stats[idx][0] == 0 and stats[idx][1] == 0:  # roi 이미지가 없으면 pass
                continue

            if np.any(np.isnan(centroid)):  # ????
                continue

            x, y, width, height, area = stats[idx]  # stats는 2차원 배열에서  a[0]이면 1행만 출력!!
            centerX, centerY = int(centroid[0]), int(
                centroid[1])  # centerX, centerY = int(centroid[0]), int(centroid[1])??
            #  print(centerX, centerY)
            # print(area)
            if (area > max_area):
                max_area = area
                max_centerX = centerX
                max_centerY = centerY

        # print(max_area)
        if max_area > 50:  # 크기가 50 이상이면 동그라미, 사각형인정
            cv2.circle(resized_color_image, (max_centerX, max_centerY), 10, (0, 0, 255), 10)
            # cv2.minAreaRect()


        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('resized_color_image', resized_color_image)
        cv2.imshow('img_mask1', img_mask1)
        # cv2.imshow('img_hsv1', img_hsv1)
        # cv2.imshow('img_hsv1', img_hsv1)
        # cv2.imshow('img_hsv1', img_hsv1)
        cv2.imshow('RealSense_depth', depth_colormap)
        cv2.imshow('img_result', img_result)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # 나가기
            cv2.destroyAllWindows()  # 윈도우 제거
            break

finally:
    pipeline.stop()
