import pyrealsense2 as rs

try:
    pipeline = rs.pipeline() #카메라 파이프라인 설정
    config = rs.config() # 카메라 설정 추상화
    config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30) # 카메라 시작 config 설정(카메라 모드, 해상도, 형식, )
    pipeline.start(config) # config설정대로 파이프라인 실행

    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame() # <class 'pyrealsense2.pyrealsense2.depth_frame'> 추상화 데이터 정보 객체, 몇번 프레임인지도 세줌.
        print(depth)
        if not depth: continue

        coverage = [0]*64
        for y in range(480):
            for x in range(640):
                dist = depth.get_distance(x,y) # 화면 depth 객체에서 거리를 추출함(픽셀 어디에서)
                if 0 < dist and dist < 1:
                    coverage[x//10]+=1 # ?

                if y%20 is 19: # 같은 메모리 주소에 있는가.
                    line = ""
                    for c in coverage: # coverage로 뭘하지?/,,,
                        line += ".:nbBXWW"[c//25]
                    coverage = [0]*64
                    print(line)

    exit(0)
except Exception as e:
    print(e)

    pass