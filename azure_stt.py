import os
import time
import datetime
import json
import azure.cognitiveservices.speech as speechsdk
import tkinter as tk
from tkinter import filedialog

#Tkinter 창 숨기기(백 그라운드로)
root = tk.Tk()
root.withdraw()

# ---------------CONFIGURATION---------------
# Azure Speech Service 키와 지역을 입력하세요.
SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION")

# 변환할 오디오 파일 경로를 지정하세요.
AUDIO_FILE_PATH = filedialog.askopenfilenames(title="STT 처리할 음성파일 선택", filetypes=[("WAV files", "*.wav")])
if not AUDIO_FILE_PATH:
    print("오디오 파일을 선택하지 않았습니다. 종료합니다.")
    exit()

# 음성인식 언어를 설정합니다.
RECOGNITION_LANGUAGE = "ko-KR"

# 자막을 나눌 기준을 직접 설정하세요.
MAX_SECOND_PER_SEGMENT = 5.0
MAX_CHARS_PER_SEGMENT = 40


def to_srt_timestamp(total_seconds):
    """총 초를 SRT 타임스탬프 형식 (HH:MM:SS,ms)으로 변환합니다."""
    td = datetime.timedelta(seconds=total_seconds) # 정수 초만 seconds에 저장하고 마이크로초는 microseconds에 저장
    total_seconds_float = td.total_seconds()
    hours, remainder = divmod(total_seconds_float, 3600)
    minutes, seconds = divmod(remainder, 60)

    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def generate_srt_from_audio_simple(input_wav, output_path):
    """오디오 파일 전체를 한 번에 처리하여 단어 수준 타임스탬프 기반의 SRT 파일을 생성합니다."""

    print(f"'{os.path.basename(input_wav)}' 파일의 SRT 생성을 시작합니다...")

    try:
        # 1. Speech SDK 설정
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        speech_config.speech_recognition_language = RECOGNITION_LANGUAGE
        speech_config.request_word_level_timestamps()
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_OutputFormatOption, "detailed")

        # 2. 오디오 파일 입력 설정
        audio_config = speechsdk.audio.AudioConfig(filename=input_wav)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        # 3. 결과 저장을 위한 리스트
        all_word_result = []

        # 'recognised' 이벤트 핸들러
        def handle_recognized(evt):
            nonlocal all_word_result
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                result_json = evt.result.properties[speechsdk.PropertyId.SpeechServiceResponse_JsonResult]
                result_data = json.loads(result_json)

                # 'NBest' (대문자 B) 키가 있는지 확인합니다.
                if 'NBest' in result_data and len(result_data['NBest']) > 0:
                    nbest_item = result_data['NBest'][0]
                    if 'Words' in nbest_item:
                        words = nbest_item['Words']
                        for word_info in words:
                            all_word_result.append({
                                'text': word_info['Word'],
                                'start_time': word_info['Offset'] / 10_000_000,
                                'end_time': (word_info['Offset'] + word_info['Duration']) / 10_000_000
                            })

        # 이벤트 핸들러 연결
        speech_recognizer.recognized.connect(handle_recognized)

        # 세션 중지 및 에러 처리를 위한 동기화 설정
        from threading import Event
        stop_event = Event()
        speech_recognizer.session_stopped.connect(lambda evt: stop_event.set())
        speech_recognizer.canceled.connect(lambda evt: stop_event.set())

        # 4. 음성 인식 실행
        speech_recognizer.start_continuous_recognition()
        stop_event.wait() # 작업이 끝날 때까지 대기

        # 5. 수집된 단어들을 기준으로 SRT 세그먼트 재구성
        print("음성 인식이 완료되었습니다. 자막 재구성을 시작합니다...")

        if not all_word_result:
            print("오류: 인식된 단어가 없어 SRT 파일을 생성할 수 없습니다.")
            return
        
        # 순서가 뒤섞일 경우를 대비해 시간순으로 정렬
        all_word_result.sort(key=lambda x: x['start_time'])

        output_filename = os.path.splitext(os.path.basename(input_wav))[0] + ".srt"
        SRT_FILE_PATH = os.path.join(output_path, output_filename)

        with open(SRT_FILE_PATH, "w", encoding="utf-8") as srt_file:
            segment_index = 1
            current_segment_text = ""
            if not all_word_result: return

            segment_start_time = all_word_result[0]['start_time']
            last_word_end_time = segment_start_time

            for i, word in enumerate(all_word_result):
                next_text = current_segment_text + (" " if current_segment_text else "") + word['text']
                segment_duration = word['end_time'] - segment_start_time

                if current_segment_text and (segment_duration > MAX_SECOND_PER_SEGMENT or len(next_text) > MAX_CHARS_PER_SEGMENT):
                    srt_file.write(f"{segment_index}\n")
                    srt_file.write(f"{to_srt_timestamp(segment_start_time)} --> {to_srt_timestamp(last_word_end_time)}\n")
                    srt_file.write(f"{current_segment_text.strip()}\n\n")

                    segment_index += 1
                    current_segment_text = word['text']
                    segment_start_time = word['start_time']
                else:
                    current_segment_text = next_text
                
                last_word_end_time = word['end_time']
            
            if current_segment_text:
                final_end_time = all_word_result[-1]['end_time']
                srt_file.write(f"{segment_index}\n")
                srt_file.write(f"{to_srt_timestamp(segment_start_time)} --> {to_srt_timestamp(final_end_time)}\n")
                srt_file.write(f"{current_segment_text.strip()}\n\n")
        
        print(f"✅ 성공! '{SRT_FILE_PATH}' 파일이 생성되었습니다.")

    except Exception as e:
        print(f"치명적인 오류가 발생했습니다: {e}")

    
# 스크립트 실행
if __name__ == "__main__":
    # 출력 파일 경로를 지정한 디렉토리에, 이름만 바꿔서 저장하도록 수정
    output_path = filedialog.askdirectory(title="SRT 파일을 저장할 폴더 선택")
    if not output_path:
        print("출력 폴더가 선택되지 않았습니다. 종료합니다.")
        exit()
    for wav in AUDIO_FILE_PATH:
        generate_srt_from_audio_simple(wav, output_path)