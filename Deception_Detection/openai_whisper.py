import whisper
import pickle
import pandas as pd

audio_file_path = 'C:\\Users\\User\\Desktop\\01_프로젝트\\음성인식\\기만탐지모델\\download_audio.wav'

def openai_whisper(audio_file_path):
    # Whisper 모델 로드
    model = whisper.load_model("base")

    # 결과를 저장할 리스트 초기화
    results = []
    
    # Process an audio file
    result = model.transcribe(audio_file_path)
    
    text_data = result['text']
    
    return text_data
    
text_data = openai_whisper(audio_file_path)

text_df = pd.DataFrame({'text': [text_data]})

text_df.to_csv('C:\\Users\\User\\Desktop\\01_프로젝트\\음성인식\\기만탐지모델\\text_data.csv')

