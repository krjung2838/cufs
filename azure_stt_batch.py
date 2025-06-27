import os
import time
import datetime
import requests
import json
import tkinter as tk
from tkinter import filedialog

# azure-ai-speech SDK 대신 azure-storage-blob만 사용
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

# --- ⚙️ CONFIGURATION ---

# 1. Azure 자격 증명 (환경 변수 사용 권장)
SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

# 2. Blob 컨테이너 이름
BLOB_CONTAINER_NAME = "stt-audio-files" # 이 부분은 보스의 컨테이너 이름으로 유지

# 3. 언어 및 API 엔드포인트 설정
RECOGNITION_LANGUAGE = "ko-KR"
BASE_URL = f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"

# 4. SRT 자막 분할 기준
MAX_SECOND_PER_SEGMENT = 5.0
MAX_CHARS_PER_SEGMENT = 40

# --- HELPER FUNCTIONS ---
# (이 부분은 이전 코드와 동일하므로, 변경 없이 그대로 사용합니다)

def to_srt_timestamp(total_seconds):
    total_seconds_float = float(total_seconds)
    hours, remainder = divmod(total_seconds_float, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"

def upload_and_get_sas_url(blob_service_client, local_file_path):
    blob_name = os.path.basename(local_file_path)
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=blob_name)
    
    print(f"  ➡️ '{blob_name}' 파일을 Azure Blob Storage에 업로드 중...")
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    sas_token = generate_blob_sas(
        account_name=blob_client.account_name, container_name=blob_client.container_name, blob_name=blob_client.blob_name,
        account_key=blob_client.credential.account_key, permission=BlobSasPermissions(read=True),
        expiry=datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
    )
    return f"https://{blob_client.account_name}.blob.core.windows.net/{blob_client.container_name}/{blob_client.blob_name}?{sas_token}"

def generate_srt_from_json_data(result_data, srt_output_path):
    print(f"  ➡️ 다운로드한 결과로 SRT 파일 생성 중...")
    all_word_result = []
    for phrase in result_data.get('recognizedPhrases', []):
        best_phrase = phrase.get('nBest', [{}])[0]
        for word_info in best_phrase.get('words', []):
            all_word_result.append({
                'text': word_info['word'],
                'start_time': word_info['offsetInTicks'] / 10_000_000,
                'end_time': (word_info['offsetInTicks'] + word_info['durationInTicks']) / 10_000_000
            })

    if not all_word_result:
        print("  ⚠️ 인식된 단어가 없어 SRT 파일을 생성할 수 없습니다.")
        return

    all_word_result.sort(key=lambda x: x['start_time'])
    
    with open(srt_output_path, "w", encoding="utf-8") as srt_file:
        segment_index = 1
        current_segment_text, segment_start_time = "", all_word_result[0]['start_time']
        last_word_end_time = segment_start_time
        for i, word in enumerate(all_word_result):
            next_text = current_segment_text + (" " if current_segment_text else "") + word['text']
            segment_duration = word['end_time'] - segment_start_time
            if current_segment_text and (segment_duration > MAX_SECOND_PER_SEGMENT or len(next_text) > MAX_CHARS_PER_SEGMENT):
                srt_file.write(f"{segment_index}\n{to_srt_timestamp(segment_start_time)} --> {to_srt_timestamp(last_word_end_time)}\n{current_segment_text.strip()}\n\n")
                segment_index += 1
                current_segment_text, segment_start_time = word['text'], word['start_time']
            else:
                current_segment_text = next_text
            last_word_end_time = word['end_time']
        if current_segment_text:
            final_end_time = all_word_result[-1]['end_time']
            srt_file.write(f"{segment_index}\n{to_srt_timestamp(segment_start_time)} --> {to_srt_timestamp(final_end_time)}\n{current_segment_text.strip()}\n\n")

    print(f"  ✅ 성공! '{os.path.basename(srt_output_path)}' 파일 생성 완료.")

# --- 🚀 MAIN EXECUTION ---

def main():
    # --- Tkinter를 이용한 파일 및 폴더 선택 ---
    root = tk.Tk()
    root.withdraw()

    print("파일 선택 대화상자를 엽니다...")
    input_audio_files = filedialog.askopenfilenames(
        title="STT 처리할 오디오 파일을 선택하세요",
        filetypes=[("Audio Files", "*.wav *.mp3 *.ogg")]
    )
    if not input_audio_files:
        print("파일을 선택하지 않았습니다. 프로그램을 종료합니다.")
        return

    print("결과를 저장할 폴더 선택 대화상자를 엽니다...")
    output_srt_folder = filedialog.askdirectory(
        title="SRT 파일을 저장할 폴더를 선택하세요"
    )
    if not output_srt_folder:
        print("폴더를 선택하지 않았습니다. 프로그램을 종료합니다.")
        return

    # --- 나머지 로직 실행 ---
    print("\n--- Azure Batch STT (REST API 방식) 프로세스를 시작합니다 ---")

    if not all([SPEECH_KEY, SPEECH_REGION, AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME]):
        print("오류: Azure 자격 증명 또는 컨테이너 이름이 설정되지 않았습니다.")
        return
    
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    headers = {'Ocp-Apim-Subscription-Key': SPEECH_KEY, 'Content-Type': 'application/json'}
    
    print(f"총 {len(input_audio_files)}개의 오디오 파일을 처리합니다.\n")

    for audio_file_path in input_audio_files:
        audio_file_name = os.path.basename(audio_file_path)
        print(f"--- 🎵 파일 처리 시작: {audio_file_name} ---")
        try:
            sas_url = upload_and_get_sas_url(blob_service_client, audio_file_path)
            
            payload = {
                "contentUrls": [sas_url],
                "properties": {
                    "wordLevelTimestampsEnabled": True, "diarizationEnabled": False,
                },
                "locale": RECOGNITION_LANGUAGE, "displayName": f"Transcription for {audio_file_name}"
            }

            print("  ➡️ Batch Transcription 작업을 Azure에 제출합니다 (POST 요청)...")
            response = requests.post(BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            status_url = response.headers['Location']
            print(f"  ✅ 작업 제출 성공! (Status URL: {status_url})")

            print("  ➡️ 작업이 완료될 때까지 대기합니다...")
            while True:
                status_response = requests.get(status_url, headers=headers)
                status_response.raise_for_status()
                status_data = status_response.json()
                if status_data['status'] in ["Succeeded", "Failed"]:
                    break
                print(f"  ... 현재 상태: {status_data['status']} (30초 후 다시 확인)")
                time.sleep(30)
            
            if status_data['status'] == "Succeeded":
                print(f"  ✅ 작업 성공! 결과를 다운로드합니다.")
                files_url = status_data['links']['files']
                files_response = requests.get(files_url, headers=headers)
                files_response.raise_for_status()
                files_data = files_response.json()['values']
                
                result_url = next((f['links']['contentUrl'] for f in files_data if f['kind'] == 'Transcription'), None)
                if result_url:
                    result_response = requests.get(result_url)
                    result_response.raise_for_status()
                    srt_filename = os.path.splitext(audio_file_name)[0] + ".srt"
                    generate_srt_from_json_data(result_response.json(), os.path.join(output_srt_folder, srt_filename))
            else:
                print(f"  ❌ 작업 실패. Status: {status_data['status']}, Details: {status_data.get('properties', {}).get('error', {})}")

        except Exception as e:
            print(f"  ❌ '{audio_file_name}' 처리 중 심각한 오류 발생: {e}")
        
        print(f"--- ⏹️ 파일 처리 종료: {audio_file_name} ---\n")

    print("--- 모든 프로세스가 완료되었습니다. ---")

if __name__ == "__main__":
    main()