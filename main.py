import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()    # 이미지 가져옴
print(pipeline)
config = rs.config()        # 설정 파일 생성
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # parity stream크기 , 포맷, 프레임 설정
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # color stream만들기

profile = pipeline.start(config)   #설정을 적용하여 이미지 취득 시작, 프로파일 얻음

depth_sensor = profile.get_device().first_depth_sensor()    # 깊이 센서를 얻음
depth_scale = depth_sensor.get_depth_scale()                # 깊이 센서의 깊이 스케일 얻음
print("Depth Scale is: ", depth_scale)

clipping_distance_in_meters = 1    # 1 meter, 클리핑할 영역을 1m로 설정
clipping_distance = clipping_distance_in_meters / depth_scale   #스케일에 따른 클리핑 거리

align_to = rs.stream.color      #depth 이미지를 맞추기 위한 이미지, 컬러 이미지
align = rs.align(align_to)      #depth 이미지와 맞추기 위해 align 생성

try:
    while True:
        frames = pipeline.wait_for_frames() #color와 depth의 프레임셋을 기다림
        #frames.get_depth_frame() 은 640x360 depth 이미지이다.

        aligned_frames= align.process(frames)   #모든(depth 포함) 프레임을 컬러 프레임에 맞추어 반환

        aligned_depth_frame = aligned_frames.get_depth_frame()  #  aligned depth 프레임은 640x480 의 depth 이미지이다
        color_frame = aligned_frames.get_color_frame()      #컬러 프레임을 얻음

        if not aligned_depth_frame or not color_frame:      #프레임이 없으면, 건너 뜀
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())     #depth이미지를 배열로,
        color_image = np.asanyarray(color_frame.get_data())             #color 이미지를 배열로

        #백그라운드 제거
        grey_color = 153
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  #depth image는 1채널, 컬러 이미지는 3채널
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        # np.where(A>0,a,b)  => A>0이면 a, 아니면 b
        # 클리핑 거리를 깊이 _이미지가 넘어서거나, 0보다 적으면, 회색으로 아니면 컬러 이미지로 반환

        #이미지 렌더링
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) # depth이미지의 맵을 컬러맵으로 보여줘라
            # applyColorMap(src, 필터) 필터를 적용함 , COLORMAP_JET=  연속적인 색상, blue -> red
            # convertScaleAbs: 인자적용 후 절대값, 8비트 반환
        images = np.hstack((bg_removed, depth_colormap))  #두 이미지를 수평으로 연결
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)   #이미지 윈도우 정의
        cv2.imshow('Align Example', images)         #이미지를 넣어 윈도우에 보임
        key = cv2.waitKey(1)                                #키 입력
        if key & 0xFF == ord('q') or key == 27:     #나가기
            cv2.destroyAllWindows()                        #윈도우 제거
            break
finally:
    pipeline.stop()     #리얼센스 데이터 스트리밍 중지
