import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

#get device product line for setting a supporting resolution
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

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

## Getting the depth sensor's depth scale (see rs-align example for explanation
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is = ", depth_scale)

clipping_distance_in_meters = 1 #1 meter
# clipping distance 란 측정 신뢰도가 높은 최대한의 거리
clipping_distance = clipping_distance_in_meters / depth_scale

# align object:
# rs.align allows us to perform alignment of depth frames to others frames
# depth카메라의 frame을 기준으로 color 카메라의 frame을 맞춰줌
# the "align_to" is the stream type to which we plan to align depth frames
# 그런데 어떤 원리로 frame을 맞출수 있는가
align_to = rs.stream.color
align = rs.align(align_to)
print(align_to) # align 객체
print(align)

# try:
while True:
    frames = pipeline.wait_for_frames()
    # align the depth framme to color frame
    aligned_frames = align.process(frames)
    # print("alignd_frames: ", aligned_frames)

    # get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    # print("aligned_depth_frame: ", aligned_depth_frame)
    # print("original_depth",frames.get_depth_frame())
    # print("original_color", frames.get_color_frame())
    color_frame = aligned_frames.get_color_frame()

    #validat that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    #remove background - Set pixels further than clipping distance to grey
    grey_color = 255
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image)) #depth image는 1채널, color image는 3채널

    bg_removed = np.where((depth_image_3d>clipping_distance)|(depth_image_3d <= 0), grey_color,color_image)
    # print(bg_removed)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.03),cv2.COLORMAP_JET)
    images = np.hstack((bg_removed,depth_colormap))

    cv2.namedWindow('Aline Example',cv2.WINDOW_NORMAL)
    # cv2.imshow('Aline Example',depth_colormap)
    # cv2.imshow('Aline Example2', color_image)
    cv2.imshow('Aline Example', images)
    key = cv2.waitKey(1)
    if key & 0xFF==ord('q') or key ==27:
        cv2.destroyAllWindows()
        break


