import cv2
import os

def process_video(frame_processing_function, 
                  input_video_path, 
                  output_video_path="output.mp4", 
                  start_time=0, # seconds 
                  duration=None):
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open the input video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.3)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.3)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps)
    end_frame = total_frames if duration is None else int((start_time + duration) * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = frame_processing_function(frame)
        out.write(processed_frame)
        current_frame += 1

    cap.release()
    out.release()

    print(f"Processed video saved to: {output_video_path}")
    return output_video_path
