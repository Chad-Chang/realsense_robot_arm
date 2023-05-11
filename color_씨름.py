# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import numpy.linalg as LA
import matplotlib.pyplot as plt
import socket
import time

detect = False

def mouse_callback(event, x, y, flags, param):
    global detect, flag1
    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 hsv1로 변환합니다.
    if event == cv2.EVENT_LBUTTONDOWN:
        detect = True
        print(detect)

cv2.namedWindow('resized_color_image')
cv2.setMouseCallback('resized_color_image', mouse_callback)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.connect((TCP_IP, TCP_PORT))##
server_socket.bind(('192.168.0.78', 2018))    # ip주소, 포트번호 지정
server_socket.listen(0)                          # 클라이언트의 연결요청을 기다리는 상태
# server_socket.timeout(1)


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6, 4))

ax = fig.add_subplot(111, projection='3d')
np.set_printoptions(precision=6, suppress=True)

########################hsv threshold#############################
# lower_green = (50, 40, 40)
# upper_green = (70, 255, 255)
lower_green = (93, 120, 60)
upper_green = (113, 255, 255)

maxArea_g = 0
inc = 0
count = 0
isBlue = '0'
area_b = 0
max_x = 0
max_y = 0
max_w = 0
max_h = 0
maxArea_b = 0
centerX_b = 0
maxCx_b = 0
maxCy_b = 0
vx,vy,vz = 0,0,0
vx_d,vy_d,vz_d = 0,0,0
# focal_len = 389.318 # mm 단위 -> 문제 있을수 있음.
focal_len_y = 592.135 # y 방향의 focal length
focal_len_x = 589.12

# focal_len = 1.56 # mm 단위 -> 문제 있을수 있음.
point_center = np.array([0,0,0])
#############################################################
# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)

if device_product_line == "L500": # 이게 뭔지
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        point_LD = np.array([0,0,0])
        point_LU = np.array([0, 0, 0])
        point_RD = np.array([0, 0, 0])
        point_RU = np.array([0, 0, 0])
        vx, vy, vz = 0, 0, 0
        vd, vx_d, vy_d, vz_d = 0, 0, 0,0
        maxArea_g = 0
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),interpolation=cv2.INTER_AREA)

        img_hsv = cv2.cvtColor(resized_color_image, cv2.COLOR_BGR2HSV)
        img_mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
        kernel = np.ones((5, 5), np.uint8)  # 모폴로지  노이즈 필터링
        ###############################morphologyEx############################
        img_mask_green = cv2.morphologyEx(img_mask_green, cv2.MORPH_OPEN, kernel)
        img_mask_green = cv2.morphologyEx(img_mask_green, cv2.MORPH_CLOSE, kernel)

        img_result = cv2.bitwise_and(resized_color_image, resized_color_image, mask=img_mask_green)
        # img_result = img_result_r | img_result_b | img_result_g
        ##############################컨투어 따기#####################################
        contours_g, _ = cv2.findContours(img_mask_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)  # 반환값 계층 정보

        for i, _ in enumerate(contours_g):  # 배열이름만 range부분에 넣으면 인덱스 추출
            cnt = contours_g[i]
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.minAreaRect(cnt)

            area = w * h
            cx = int(x + w / 2)  # x중심 위치
            cy = int(y + h / 2)  # y중심 위치
            # print(area)
            if area > maxArea_g:  # 가장 큰 물체를 업데이트 해줌.
                maxArea_g = area
                maxCx_g = cx
                maxCy_g = cy
        if maxArea_g > 80:  # 80 점만 보여도 서보가 돌아가도록 하는 것이 좋을듯 ** 수정 사항 (mbed에서 size_g 값이 0이면 dc모터가 그 방향으로 돌지 않도록 or 파이썬에서 해결)
            #################isObject_g=true : ang = ######################
            cv2.circle(resized_color_image, (maxCx_g, maxCy_g), 1, (0, 0, 255), 2)
            cv2.putText(resized_color_image, "center" +'[' +str(maxCx_g)+','+str(maxCy_g)+','+str(depth_image[maxCy_g, maxCx_g])+']', (maxCx_g-70,maxCy_g+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1,
                        cv2.LINE_AA)
            point_center[0] = maxCx_g
            point_center[1] = maxCy_g
            point_center[2] = depth_image[maxCy_g,maxCx_g]

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # print(box.shape)
            # print(box) # 네 점을 리스트 형태로 가지고 있음.
            cv2.drawContours(resized_color_image,[box],-1,(0,0,255),1)
            # print("box0=",box[0],"box1=",box[1],",box2=",box[2],"box3=",box[3])


            # point3[:2] = box[3]
            # x,y는 반대로
            np.putmask(box[:, 0], box[:, 0] >= resized_color_image.shape[1], resized_color_image.shape[1]-1)
            np.putmask(box[:, 1], box[:, 1] >= resized_color_image.shape[0], resized_color_image.shape[0] - 1)

            v3_0 = box[0] - box[3]
            v3_2 = box[2] - box[3]
            if LA.norm(v3_0) >= LA.norm(v3_2):
                Vx = v3_2
                Vy = v3_0
                vy = v3_0/LA.norm(Vy)
                vx = v3_2/LA.norm(Vx)
                vz = np.cross(vx, vy)
                vz = -vz/LA.norm(vz)
                # print(vz)
                point_LD[:2] = box[3]
                point_LD[2] = depth_image[box[3, 1], box[3, 0]]
                # cv2.circle(resized_color_image, (point_LD[0], point_LD[1]), 10, (0, 0, 255), 2)
                point_LU[:2] = box[0]
                point_LU[2] = depth_image[box[0, 1], box[0, 0]]
                point_RD[:2] = box[2]
                point_RD[2] = depth_image[box[2, 1], box[2, 0]]
                point_RU[:2] = box[1]
                point_RU[2] = depth_image[box[1, 1], box[1, 0]]


                # ax.plot(vx_d, vy_d, vz_d)

            elif LA.norm(v3_2) > LA.norm(v3_0):
                Vx = -v3_0
                Vy = v3_2
                vy = Vy/LA.norm(Vy)
                vx = Vx/LA.norm(Vx) # v0를 reference 포인트로 만들기시
                vz = np.cross(vx, vy)
                vz = -vz / LA.norm(vz)
                # print(vz)
                point_LD[:2] = box[0]
                # point_LD[2] = depth_image[box[0, 1], box[0, 0]]
                point_LU[:2] = box[1]
                # point_LU[2] = depth_image[box[1, 1], box[1, 0]]
                point_RD[:2] = box[3]
                # point_RD[2] = depth_image[box[3, 1], box[3, 0]]
                point_RU[:2] = box[2]

            else:
                pass
            Depth_point_LD, Depth_point_LU, Depth_point_RD, Depth_point_RU = np.array([0, 0, 0]), np.array(
                [0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])
            Depth_point_LD[:2] = point_LD[:2] + Vx / 3 + Vy / 3
            Depth_point_RD[:2] = Depth_point_LD[:2] + Vx / 3
            Depth_point_LU[:2] = Depth_point_LD[:2] + Vy / 3
            Depth_point_RU[:2] = Depth_point_LD[:2] + Vx / 3 + Vy / 3

            Depth_point_LD[2] = depth_image[Depth_point_LD[1], Depth_point_LD[0]]
            Depth_point_RD[2] = depth_image[Depth_point_RD[1], Depth_point_RD[0]]
            Depth_point_LU[2] = depth_image[Depth_point_LU[1], Depth_point_LU[0]]
            Depth_point_RU[2] = depth_image[Depth_point_RU[1], Depth_point_RU[0]]

            vx_d = (Depth_point_RD-Depth_point_LD) / LA.norm(Depth_point_RD-Depth_point_LD)
            vy_d = -(Depth_point_LU-Depth_point_LD) / LA.norm(Depth_point_LU-Depth_point_LD)
            vz_d = np.cross(vx_d, vy_d) / LA.norm(np.cross(vx_d, vy_d))  # 위 방향인지 아래방향인지 아직 모름.

            # depth가 안정적인 부분만 뽑기 위한 점을 시각화
            cv2.circle(depth_colormap, (Depth_point_LD[0], Depth_point_LD[1]), 1, (255, 255, 0), 1)
            cv2.circle(depth_colormap, (Depth_point_RD[0], Depth_point_RD[1]), 1, (0, 255, 0), 1)
            cv2.circle(depth_colormap, (Depth_point_LU[0], Depth_point_LU[1]), 1, (0, 255, 0), 1)
            cv2.circle(depth_colormap, (Depth_point_RU[0], Depth_point_RU[1]), 1, (0, 255, 0), 1)
            cv2.circle(depth_colormap, (point_center[0], point_center[1]), 1, (0, 255, 0), 1)

            # 벡터 시각화
            cv2.arrowedLine(resized_color_image, (point_LD[0], point_LD[1]), (point_LD[0] + Vx[0], point_LD[1] + Vx[1]),(0, 0, 255),thickness=2)  # X축 표시
            cv2.arrowedLine(resized_color_image, (point_LD[0], point_LD[1]),(point_LD[0] + Vy[0], point_LD[1] + Vy[1]),(255, 0, 0),thickness=2)  # Y축 표시

#====중심 원점 표시===========================================================================================================================
            cv2.circle(resized_color_image,(resized_color_image.shape[1] // 2, resized_color_image.shape[0] // 2),1, (0, 255, 0), 1)
            # cv2.arrowedLine(resized_color_image, (resized_color_image.shape[1] // 2, resized_color_image.shape[0] // 2),(resized_color_image.shape[1] // 2+30, resized_color_image.shape[0] // 2),(0, 255, 0),thickness=2)  # X축 표시
            # cv2.arrowedLine(resized_color_image, (resized_color_image.shape[1] // 2, resized_color_image.shape[0] // 2),(resized_color_image.shape[1] // 2 , resized_color_image.shape[0] // 2 + 30),(0, 255, 0),thickness=2)  # X축 표시
            cv2.putText(resized_color_image,"center" + '[' + str(resized_color_image.shape[1] // 2) + ',' + str(resized_color_image.shape[0] //2)+','+str(depth_image[resized_color_image.shape[1]//2, resized_color_image.shape[0]//2])+']',(resized_color_image.shape[1] // 2, resized_color_image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#====x중심 고정===========================================================================================================================
            cv2.circle(resized_color_image, (resized_color_image.shape[1]//2, point_center[1]), 1, (0, 0, 255), 1)
            cv2.putText(resized_color_image, "center" + '[' + str(resized_color_image.shape[1]//2) + ',' + str(point_center[1]) + ',' + str(depth_image[resized_color_image.shape[0]//2, point_center[0]])+']', (resized_color_image.shape[1]//2, point_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# normalized image plane
            x_normalized = (point_center[0] - resized_color_image.shape[1]//2) / focal_len_x
            y_normalized = (point_center[1] - resized_color_image.shape[0]//2) / focal_len_y
            Zc = point_center[2]
            Xc = round(x_normalized * Zc, 3)
            Yc = round(y_normalized * Zc, 3)
            Xc2 = round((point_center[0] - resized_color_image.shape[1]//2)*2*Zc/1225, 3)
            Yc2 = round((point_center[1] - resized_color_image.shape[0]//2)*2*Zc/1225, 3)
            # print(Xc, Xc2)
            # print(Yc, Yc2)

            x_normalized_aline = (resized_color_image.shape[1]//2 - resized_color_image.shape[1] // 2) / focal_len_x
            y_normalized_aline = (point_center[1] - resized_color_image.shape[0] // 2) / focal_len_y
            Zc_aline = depth_image[resized_color_image.shape[1]//2, point_center[1]]
            Xc_aline = round(x_normalized_aline * Zc_aline, 3)
            Yc_aline = round(y_normalized_aline * Zc_aline, 3)

            x_normalized_zero = (resized_color_image.shape[1] // 2 - resized_color_image.shape[1] // 2) / focal_len_x
            y_normalized_zero = (resized_color_image.shape[0] // 2 - resized_color_image.shape[0] // 2) / focal_len_y
            Zc_zero = depth_image[resized_color_image.shape[1] // 2, resized_color_image.shape[0] // 2]
            Xc_zero = round(x_normalized_zero * Zc_zero, 3)
            Yc_zero = round(y_normalized_zero * Zc_zero, 3)
            # print("zero=", Xc_zero, Yc_zero, Zc_zero)

            # 3축 만들기(matplot)
            vxx_d_1 = np.linspace(Xc, Xc + vx_d[0]*100, 10)
            vxy_d_1 = np.linspace(Yc, Yc + vx_d[1]*100, 10)
            vxz_d_1 = np.linspace(Zc, Zc + vx_d[2]*100, 10)
            vyx_d_1 = np.linspace(Xc, Xc + vy_d[0]*100, 10)
            vyy_d_1 = np.linspace(Yc, Yc + vy_d[1]*100, 10)
            vyz_d_1 = np.linspace(Zc, Zc + vy_d[2]*100, 10)
            vzx_d_1 = np.linspace(Xc, Xc + vz_d[0]*100, 10)
            vzy_d_1 = np.linspace(Yc, Yc + vz_d[1]*100, 10)
            vzz_d_1 = np.linspace(Zc, Zc + vz_d[2]*100, 10)

            # 회전 행렬 만들기
            # rotation= np.concatenate(vx.T, vy.T, vz.T)
            rotation = np.concatenate([np.array([vx_d]).T,np.array([vy_d]).T,np.array([vz_d]).T],axis=1)
            # print("rotation= ", rotation, rotation.shape)
            # print('vx =', vx_d, 'vy =', vy_d,'vz =', vz_d,)
            position = np.array([[Xc,Yc,Zc]]).T
            # print("rotation=", rotation, 'shape=', rotation.shape)
            # print('position=', position, 'shape=',position.shape)
            null_m = np.array([[0,0,0,1]])
            T = np.around(np.concatenate([rotation, position], axis=1),4)
            H = np.concatenate([T,null_m],axis=0)
            ext_T = np.array([[-1, 0, 0, 265], [0, 0, -1, 115], [0, -1, 0, 135],[0, 0, 0, 1]])

            # print(ext_T)
            # print("camera=", H)
            # print("ext =", ext_T)
            # print("world=",ext_T*H)
            # print("world_rotation =", ext_T[:3,:3]*H[:3,:3])
            # print(vx_d, LA.norm(vx_d))
            # print(H)
# 동차 행렬 행별로 나눠주기
#             H_row= 0
            H_row = str(H[0,0])+','+str(H[0,1])+','+str(H[0,2])+','+str(H[0,3])+'/'+ str(H[1,0])+','+str(H[1,1])+','+str(H[1,2])+','+str(H[1,3])+'/'+str(H[2,0])+','+str(H[2,1])+','+str(H[2,2])+','+str(H[2,3])+'/'+str(H[3,0])+','+str(H[3,1])+','+str(H[3,2])+','+str(H[3,3])+'/'
            # H_str =str(H)[1:-1]
            # print(H_row)

# 매트랩에 통신 보내기

            if detect:
                client_socket, addr = server_socket.accept()
                client_socket.send(H_row.encode())
                print('sent')
                client_socket.close()
                # 축 라벨
            ax.set_xlabel("x", size=14)
            ax.set_ylabel("y", size=14)
            ax.set_zlabel("z", size=14)

            # 축 눈금 지정
            ax.plot(0, 0, 0, color="purple")
            ax.view_init(elev=-90, azim=-90)
            ax.plot(vxx_d_1, vxy_d_1, vxz_d_1, color="red")# y축
            ax.plot(vyx_d_1, vyy_d_1, vyz_d_1, color="blue")# x축
            ax.plot(vzx_d_1, vzy_d_1, vzz_d_1, color="green")# z 축

            plt.pause(0.0001)
        detect = False
        # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('resized_color_image', resized_color_image)
        # cv2.imshow('depth_colormap', depth_colormap)
        # cv2.imshow('img_result', img_result)
        plt.cla()

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
