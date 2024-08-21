import os
import pandas as pd
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import whisper
from moviepy.editor import VideoFileClip
import pickle

path = r'C:\\Users\\User\\Desktop\\01_프로젝트\\음성인식\\기만탐지모델'

def download_ytb(ytb_url):    
    # FFmpeg 위치 설정
    ffmpeg_location = r"C:\\Users\\User\\Desktop\\00_Tools\\ffmpeg-2024-07-10-git-1a86a7a48d-full_build\\bin\\ffmpeg.exe"
    
    # 오디오 yt-dlp 옵션 설정
    ydl_opts = {
        'format': 'bestaudio/best',  # 최고의 오디오 품질로 다운로드
        'outtmpl': 'temp.%(ext)s',  # 임시 파일 생성
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
            'preferredquality': '192',
        }],
        'ffmpeg_location': ffmpeg_location  # FFmpeg 위치 지정
    }
    
    def download_and_process_audio(url):
        with YoutubeDL(ydl_opts) as ydl:
            # YouTube URL에서 오디오 다운로드
            info_dict = ydl.extract_info(url, download=True)
            temp_filename = ydl.prepare_filename(info_dict).replace('.webm', '.m4a')
    
        # 출력 파일 경로
        output_filename = r'C:\\Users\\User\\Desktop\\01_프로젝트\\음성인식\\기만탐지모델\\download_audio.wav'
    
        # pydub를 사용하여 오디오 파일을 읽어오기
        try:
            audio_segment = AudioSegment.from_file(temp_filename, format="m4a")
            # WAV 파일로 변환하여 저장
            audio_segment.export(output_filename, format="wav")
        except Exception as e:
            print(f"오디오 파일 변환 중 오류 발생: {e}")
    
        # 다운로드 후 임시 파일 삭제
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
        # 처리된 오디오 파일의 경로 반환
        return output_filename
    
    # 오디오 다운로드 및 처리 호출
    audio_file_path = download_and_process_audio(ytb_url)
    print(f"오디오 파일 저장 경로: {audio_file_path}")
    
    # 비디오 파일 경로 설정
    video_file_path = r'C:\\Users\\User\\Desktop\\01_프로젝트\\음성인식\\기만탐지모델\\downloaded_video.mp4'
    
    # 기존 비디오 파일이 있으면 삭제
    if os.path.exists(video_file_path):
        os.remove(video_file_path)
        print(f"기존 비디오 파일이 삭제되었습니다: {video_file_path}")
    
    # 비디오 yt-dlp 옵션 설정
    ydl_video_opts = {
        'format': 'mp4',  # MP4 형식으로 다운로드
        'outtmpl': video_file_path,
        'ffmpeg_location': ffmpeg_location,  # FFmpeg 위치 지정
        'noplaylist': True,  # 플레이리스트의 경우 첫 번째 비디오만 다운로드
        'progress_hooks': [lambda d: print(f"진행 상황: {d['status']}")]  # 다운로드 진행 상황 출력
    }
    
    # yt-dlp 객체 생성 및 비디오 다운로드
    with YoutubeDL(ydl_video_opts) as ydl:
        ydl.download([ytb_url])
    
    # 비디오 파일 로드
    video_clip = VideoFileClip(video_file_path)


ytb_url = "https://www.youtube.com/watch?v=mWyXyqxxAM4"  

download_ytb(ytb_url)
