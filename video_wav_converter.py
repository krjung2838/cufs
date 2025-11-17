import subprocess
import os
import tkinter as tk
from tkinter import filedialog


def extract_audio_from_video(video_files, output_dir):
    # ffmpeg 경로를 절대 경로로 지정
    ffmpeg_path = r"C:\Users\cufs\Desktop\업무\subtitle\ffmpeg\ffmpeg.exe"  # ffmpeg 경로를 실제 설치된 경로로 수정
    

    for video_file in list(video_files):
        if video_file.endswith(".mp4"):
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_file))[0]}.wav")

            command = [ffmpeg_path, "-i", video_file, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_file]
            try:
                subprocess.run(command, check=True)
                print(f"오디오 추출 완료: {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"에러 발생: {e}")
        

#Tkinter 창 숨기기(백 그라운드로)
root = tk.Tk()
root.withdraw()

# 예시 사용
video_files = filedialog.askopenfilenames(
            title="mp4 파일 선택 (다중 선택 가능)",
            filetypes=[("mp4 Files", "*.mp4")]
        )
output_directory = filedialog.askdirectory(title="WAV파일을 저장할 경로 선택")

extract_audio_from_video(video_files, output_directory)
