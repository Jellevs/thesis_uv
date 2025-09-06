import supervision as sv
import cv2

def read_video(input_video_path):
    frame_generator = sv.get_video_frames_generator(input_video_path)
    frames = [frame for frame in frame_generator]
    return frames

def save_video(input_video_path, output_video_path, annotated_frames):
    video_info = sv.VideoInfo.from_video_path(input_video_path)
    with sv.VideoSink(f"{output_video_path}/output_video.mp4", video_info=video_info) as sink:
        for frame in annotated_frames:
            sink.write_frame(frame)


def save_images(annotated_frames, output_folder):
    for idx, frame in enumerate(annotated_frames):
        cv2.imwrite(f"{output_folder}/frame_{idx:04d}.png", frame)