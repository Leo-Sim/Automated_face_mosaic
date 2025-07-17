import cv2
from ultralytics import YOLO



model = YOLO('../yolo/runs/train/yolo_train6/weights/best.pt')


# 비디오 파일 열기
video_path = 'street.webm'  # 비디오 파일 경로
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# 비디오의 프레임 크기와 FPS 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Video properties: {frame_width}x{frame_height}, FPS: {fps}")

# 비디오 출력 설정 (결과를 저장하고 싶을 경우)
output_path = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 프레임 처리
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video processing complete.")
        break

    # YOLO 모델을 사용해 객체 탐지
    results = model(frame)

    # 탐지된 객체를 프레임에 표시
    annotated_frame = results[0].plot()  # 결과를 시각화한 프레임

    # 프레임 표시
    cv2.imshow('YOLO Detection', annotated_frame)

    # 비디오 저장
    out.write(annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()