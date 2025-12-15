import srt
from datetime import timedelta
import tkinter as tk
from tkinter import filedialog
import shutil
from pathlib import Path
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from pyannote.audio import Pipeline, Audio
from pyannote.core import Annotation, Segment
from pydub import AudioSegment
from speechbrain.inference import EncoderClassifier
import subprocess
import re
import stable_whisper # STT를 위한 라이브러리
from collections import Counter
import traceback
import os
import pandas as pd
from speechbrain.inference import SpeakerRecognition
from inaSpeechSegmenter import Segmenter
import soundfile as sf


# ========== 사용자 설정 파라미터 ==========
HF_TOKEN = "" # 허깅페이스 토큰
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FFMPEG_PATH = r"" # 📌 FFmpeg 실행 파일의 '전체 경로'를 정확하게 입력하세요.
STT_MODEL_SIZE = "large-v3-turbo" # STT 모델 크기 ('tiny', 'base', 'small', 'medium', 'large-v3')
FILENAME_SUFFIX = "" # 최종 파일명에 추가될 접미사
MAIN_LANGUAGE = "ko"
SUB_LANGUAGE = None # 보조언어를 강제로 고정. 기본값은 None. # 보조언어를 두지 않으려면 "no_sub"으로 설정
THIRD_LANG = "no_third" # 
ALLOWED_LANGS = ['ko', 'en', 'vi', 'es', 'zh', 'ja', 'id']


# 1. FFmpeg 볼륨 전처리 파라미터
SILENCE_THRESH_DB = -50  # FFmpeg이 '침묵'으로 판단할 소리의 크기 기준입니다. -40dB보다 작은 소리는 침묵으로 간주합니다.
MIN_SILENCE_DURATION_S = 0.05  # 최소 0.05초 이상 지속되는 침묵 구간만 찾아내도록 설정합니다.


# 2. VAD 모델 세부 파라미터
VAD_PARAMS = {
    "min_duration_off": 0.01,  # 음성이 없는 구간(침묵)이 최소 0.01초는 되어야 침묵으로 인정합니다.
    "min_duration_on": 0.05,  # 음성이 있는 구간이 최소 0.01초는 되어야 음성으로 인정합니다.
    "onset": 0.01,  # 음성 시작이라고 판단할 확률의 임계값입니다. (0~1 사이, 높을수록 보수적)
    "offset": 0.01  # 음성 종료라고 판단할 확률의 임계값입니다. (0~1 사이, 높을수록 보수적)
}

# VAD 세부설정
MAX_DURATION = 1 # 충분히 길다고 판단할 병합 전 VAD의 길이 기준입니다.
MAX_GAP = 0.5 # 충분히 길다고 판단할 병합 전 각 VAD의 갭의 길이 기준입니다.
MAX_MERGED_DURATION = 5 # 병합된 세그먼트의 최대길이입니다.

# ▼▼▼ 화자 분리 민감도 튜닝 ▼▼▼
diarization_params = {
    # 1. 잡음/끊김 처리
    "segmentation": {
        "min_duration_on": 0.05, 
        "min_duration_off": 0.01  
    },
    
    # 2. 화자 구분 민감도
    "clustering": {
        "method": "centroid", # 중심점 기준 (기본값)
        "min_cluster_size": 12, # 최소 이 정도 크기는 되어야 화자로 인정 (기본 12~15)
        "threshold": 1.0, # ★ 핵심: 0.0 ~ 1.0 사이 (기본값은 보통 0.7 내외)
    }
}

# 4. 최종 SRT 생성 및 병합 파라미터
MIN_SEGMENT_DURATION = 0.1 # VAD로 찾아낸 음성 구간 중 0.1초보다 짧은 구간은 너무 짧은 노이즈일 가능성이 높으므로 무시하고 제거합니다.
MERGE_MAX_SECONDS = 15.0 # STT를 하기 전, 같은 언어의 음성 구간들을 합칠 때 최대 15초까지만 합치도록 제한합니다. 너무 길면 STT 성능이 떨어질 수 있습니다.


# --- 자막 병합 설정 ---
MERGE_THRESHOLD_SECONDS = 1.0  # 1차 병합 기준: 자막을 합칠 기준 시간 (초)
MAX_CHARS_PER_LINE = 30      # 1차 병합 기준: 합쳐진 자막의 최대 글자 수
MIN_DURATION_SECONDS = 1.0   # 2차 병합 기준: 이 시간(초)보다 짧은 자막은 앞 자막에 강제로 합침


# --- 프롬프트 딕셔너리 ---
en = "Today, we will discuss the importance of renewable energy. The quick brown fox jumps over the lazy dog."
# ja = "오늘은 て形랑 辞書形를 볼 거야. て形는 연결·부탁(〜てください), 辞書形는 기본형. ５番出口에서 만나. 엘리베이터는 エレベーター, 계단은 階段. "
ja = "테형 뒤에 이루를 붙이면 진행형이 돼. 시테 이루는 '하고 있다'라는 뜻이야. 오스스메 메뉴가 뭐예요? 나마비루 두 잔 주세요. 사이후를 잃어버려서 케이사츠에 신고했어."
vi = "씬짜오 깜언 또이 드억조이 퍼 반미 응온 아잉 엠 자오비엔 바오니에우"
es = "이것은 한국어와 스페인어를 사용하는 스페인어 문법 강의입니다. Me llamo Juan. ¿Dónde está la biblioteca? Te quiero mucho."
idn = "슬라맛 빠기 뜨리마 까시 아빠 까바르 쁘르기 싸야 팅갈 디 서울 뜨리마 까시 바냑 삼빠이 줌빠 라기"
zh = "오늘은 把字句랑 被字句를 비교할 거야. 把字句는 처분 강조, 被字句는 피동.三号出口에서 만나. 택시는 打车, 갈아타기는 换乘."
ko = ""
INSTRUCTOR_PROMPT_DICT = {
    'vi': vi,
    'es': es,
    'id': idn,
    'zh': zh,
    'en': en,
    'ja': ja,
    'ko': ko
}


# ========== Symlink 우회 Patch (Windows 환경 호환성) ==========
def force_copy(src, dst):
    if src is None or dst is None: return None
    src_path, dst_path = Path(src), Path(dst)
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir(): shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else: shutil.copy2(src_path, dst_path)
        return dst
    except Exception as e:
        print(f"   [경고] 파일 복사 중 오류 발생: {e}")
        return None

import speechbrain.utils.fetching as sb_fetch
sb_fetch.link_with_strategy = lambda src, dst, strategy: force_copy(src, dst)

# ========== 모델 로딩 (스크립트 시작 시 1회 실행) ==========
print("🔄 모델 로딩 중... (VAD, Language ID, STT)")
# 1. VAD
vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=HF_TOKEN)
vad_pipeline.to(torch.device(DEVICE))
vad_pipeline.instantiate(VAD_PARAMS)
print("✅ VAD 모델 로딩 완료.")

# 2. Language ID
lang_id_model = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp_lang_id"
)
print("✅ Language ID 모델 로딩 완료.")

# 3. STT (Stable Whisper)
stt_model = stable_whisper.load_model(STT_MODEL_SIZE, device=DEVICE)
print(f"✅ STT 모델({STT_MODEL_SIZE}) 로딩 완료.")


# 화자 인식(검증) 전용 모델입니다. 성능이 아주 뛰어납니다.
print("🔄 Speaker Verification 모델 로딩 중... (SpeechBrain)")
verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="tmp_speaker_verification",
    run_opts={"device": DEVICE}
)
print("✅ Speaker Verification 모델 로딩 완료.")

# 음악 감지 모델
print("🔄 Music/Speech 세그먼터 로딩 중... (inaSpeechSegmenter)")
music_segmenter = Segmenter(vad_engine="smn", detect_gender=False)
print("✅ Music/Speech 세그먼터 로딩 완료.")


# ========== 핵심 기능 함수 ==========

def read_xlsx_and_create_dict(xlsx_file_path):
    """강의명과 보조언어가 매칭되어 있는 엑셀파일을 불러와 딕셔너리로 생성합니다."""
    
    df = pd.read_excel(
    io=xlsx_file_path,     # 1. 이 파일을
    header=3,          # 2. 4번째 줄을 헤더로 삼아서
    usecols="C:D"      # 3. C열과 D열만 읽어라
    )     
    df = df.dropna(subset=['보조언어']) # 보조언어가 비어있는 행은 모두 제거합니다.
    
    lang_map = df.set_index('강의명')['보조언어'].to_dict() # 강의명 : 보조언어 의 형식으로 딕셔너리를 생성합니다.
    keys_view = list(lang_map.keys()) # lang_map에서 key값만 뽑아서 리스트로 만듭니다. 이는 한 공통접두어가 다른 공통접두어를 포함할 경우를 대비하기 위함입니다.
    sorted_list = sorted(keys_view, key=len, reverse=True) # 리스트를 길이가 긴 순서대로 정렬합니다.
    
    return lang_map, sorted_list




def get_non_silent_segments_ffmpeg(audio_path):
    print("\n🔊 0. FFmpeg로 침묵 구간 분석 시작...")
    command = [FFMPEG_PATH, '-i', str(audio_path), '-af', f'silencedetect=noise={SILENCE_THRESH_DB}dB:d={MIN_SILENCE_DURATION_S}', '-f', 'null', '-']
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        ffmpeg_output = result.stderr
    except FileNotFoundError:
        print(f"\n[치명적 오류] 'ffmpeg'를 찾을 수 없습니다. FFMPEG_PATH 변수 경로를 확인하세요: {FFMPEG_PATH}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"\n[오류] FFmpeg 실행 중 오류 발생: {e.stderr}")
        return None
    
    silence_starts = [float(t) for t in re.findall(r'silence_start: (\d+\.?\d*)', ffmpeg_output)]
    silence_ends = [float(t) for t in re.findall(r'silence_end: (\d+\.?\d*)', ffmpeg_output)]

    if not silence_starts:
        print("   [정보] FFmpeg가 침묵 구간을 찾지 못했습니다. 전체 파일을 분석합니다.")
        return "full_audio"

    if len(silence_starts) > len(silence_ends):
        silence_starts = silence_starts[:len(silence_ends)]

    non_silent_segments = []
    last_end = 0.0
    for start, end in zip(silence_starts, silence_ends):
        if start > last_end + 0.01:
            non_silent_segments.append({'start': last_end, 'end': start})
        last_end = end

    try:
        duration = len(AudioSegment.from_file(audio_path)) / 1000.0
        if duration > last_end + 0.01:
            non_silent_segments.append({'start': last_end, 'end': duration})
    except Exception as e:
        print(f"   [경고] 오디오 전체 길이 확인 중 오류 발생: {e}")

    print(f"✅ FFmpeg 분석 완료. {len(non_silent_segments)}개의 유성 구간 발견.")
    return non_silent_segments




def detect_music_segments(audio_path):
    """
    inaSpeechSegmenter 결과를 받아서 즉시 딕셔너리 리스트로 변환하여 반환합니다.
    Output: [{'label': 'music', 'start': 0.0, 'end': 10.0}, ...]
    """
    print("\n🎼 0-1. inaSpeechSegmenter로 음악/음성 구간 분석 시작...")
    try:
        # 라이브러리는 튜플 리스트를 뱉습니다: [('music', 0.0, 5.0), ...]
        raw_segments = music_segmenter(str(audio_path))
        print(f"   - inaSpeechSegmenter 세그먼트 개수: {len(raw_segments)}")
        
        # ★ 여기서 딕셔너리로 변환!
        dict_segments = []
        for label, start, end in raw_segments:
            dict_segments.append({
                'label': label,
                'start': float(start),
                'end': float(end)
            })
            
        return dict_segments

    except Exception as e:
        print(f"   [경고] 음악 구간 분석 실패: {e}")
        return []
    
    
    

def build_music_blocks(ina_segments, short_speech_max=1.0):
    if not ina_segments: return []

    blocks = []
    cur_start = None
    cur_end = None


    for item in ina_segments:
        label = item['label']
        start = item['start']
        end = item['end']
        dur = end - start

        if label == "music":
            if cur_start is None: cur_start = start
            cur_end = end
        else:
            if cur_start is not None and label in ("speech", "noise") and dur <= short_speech_max:
                cur_end = end
            else:
                if cur_start is not None:
                    blocks.append((cur_start, cur_end))
                    cur_start = None
                    cur_end = None

    if cur_start is not None:
        blocks.append((cur_start, cur_end))

    if not blocks: return []


    blocks = sorted(blocks)
    merged = []
    for start, end in blocks:
        if not merged:
            merged.append([start, end])
        else:
            last_start, last_end = merged[-1]
            if start <= last_end + 0.2:
                merged[-1][1] = max(last_end, end)
            else:
                merged.append([start, end])

    music_blocks = [(s, e) for s, e in merged]
    print(f"   - 병합된 음악 블록 개수: {len(music_blocks)}")
    return music_blocks




def remove_music_from_non_silent(non_silent_segments, music_blocks, min_len=0.05):
    """
    ffmpeg로 얻은 유성 구간(non_silent_segments)에서
    music_blocks를 전부 빼고 남은 구간만 반환.

    non_silent_segments: [{'start': float, 'end': float}, ...]
    music_blocks: [(start, end), ...]
    """
    if not music_blocks:
        return non_silent_segments

    if not non_silent_segments or non_silent_segments == "full_audio":
        # "full_audio"는 여기서 처리하지 않고, 호출부에서 별도 처리
        return non_silent_segments

    cleaned = []

    for seg in non_silent_segments:
        seg_start = float(seg["start"])
        seg_end   = float(seg["end"])
        parts = [(seg_start, seg_end)]

        for m_start, m_end in music_blocks:
            new_parts = []
            for p_start, p_end in parts:
                # 겹치지 않으면 그대로 유지
                if p_end <= m_start or p_start >= m_end:
                    new_parts.append((p_start, p_end))
                    continue

                # 겹치면 음악 부분만 잘라내고 양쪽만 유지
                if p_start < m_start:
                    new_parts.append((p_start, m_start))
                if p_end > m_end:
                    new_parts.append((m_end, p_end))

            parts = new_parts
            if not parts:
                break

        # 너무 짧은 구간은 버리고, 일정 길이 이상만 채택
        for p_start, p_end in parts:
            if p_end - p_start >= min_len:
                cleaned.append({"start": p_start, "end": p_end})

    print(f"   - 음악 제거 전 유성 구간: {len(non_silent_segments)}개 → 제거 후: {len(cleaned)}개")
    return cleaned




def extract_segments_2stage(waveform, sample_rate, non_silent_segments):
    print("\n🚀 1. 2단계 VAD 기반 세그먼트 추출 시작...")
    final_vad_annotation = Annotation()
    
    if non_silent_segments == "full_audio":
        print("   - 전체 오디오에 대해 VAD를 실행합니다.")
        return vad_pipeline({"waveform": waveform, "sample_rate": sample_rate})

    total_speech_chunks_found = 0
    skipped_chunks = 0
    
    # Pyannote가 처리할 수 있는 최소 길이 (안전하게 0.06초 정도로 잡음)
    MIN_CHUNK_SAMPLES = int(sample_rate * 0.06) 

    for i, segment in enumerate(non_silent_segments):
        start, end = segment['start'], segment['end']
        start_frame, end_frame = int(start * sample_rate), int(end * sample_rate)
        
        # 인덱스 범위 보호
        if end_frame > waveform.shape[1]:
            end_frame = waveform.shape[1]
            
        chunk_waveform = waveform[:, start_frame:end_frame]
        
        # ★ [핵심 수정] 너무 짧은 오디오 조각(0.06초 미만)은 VAD 에러를 유발하므로 건너뜁니다.
        if chunk_waveform.shape[1] < MIN_CHUNK_SAMPLES:
            skipped_chunks += 1
            continue

        file_chunk = {"waveform": chunk_waveform, "sample_rate": sample_rate}
        
        try:
            # VAD 실행 (여기서 에러가 나더라도 프로그램이 죽지 않도록 예외 처리 추가)
            vad_result_chunk = vad_pipeline(file_chunk)
            
            for speech_turn, _, _ in vad_result_chunk.itertracks(yield_label=True):
                offset_speech_turn = Segment(speech_turn.start + start, speech_turn.end + start)
                final_vad_annotation[offset_speech_turn] = "speech"
                total_speech_chunks_found += 1
                
        except Exception as e:
            # 혹시 모를 내부 에러 방지 (로그만 남기고 계속 진행)
            print(f"   [경고] VAD 처리 중 조각 스킵됨 ({start:.2f}~{end:.2f}s): {e}")
            continue
            
    merged_annotation = Annotation()
    for segment in final_vad_annotation.support().itersegments():
        merged_annotation[segment] = "speech"
        
    print(f"✅ 2단계 VAD 분석 완료. 총 {total_speech_chunks_found}개의 음성 조각 발견. (너무 짧아 생략된 조각: {skipped_chunks}개)")
    return merged_annotation




def detect_language_for_vad_segments(vad_annotation, waveform, sample_rate, lang_id_model):
    """
    pyannote VAD Annotation 결과와 이미 로드된 waveform을 사용해 각 세그먼트의 언어를 감지합니다.
    (★수정: 0.1초 미만 구간은 언어 감지 대상에서 사전에 제외합니다)
    """
    print("\n🚀 VAD 구간별 언어 감지 시작 (0.1초 미만 사전 제거)...")
    
    label_encoder = lang_id_model.hparams.label_encoder
    
    # 1. Annotation 객체를 처리하기 쉬운 딕셔너리 리스트로 변환합니다.
    segments_with_lang = []
    skipped_short_count = 0

    for segment in vad_annotation.itersegments():
        duration = segment.end - segment.start
        
        # ⚡ [핵심 수정] 0.1초 미만이면 아예 리스트에 넣지 않고 건너뜁니다.
        if duration < 0.1:
            skipped_short_count += 1
            continue
            
        segments_with_lang.append({'start': segment.start, 'end': segment.end})

    print(f"   - ✂️ 0.1초 미만 초단파 {skipped_short_count}개 사전 제거됨.")

    # 2. 각 세그먼트를 순회하며 언어를 감지합니다.
    for seg in segments_with_lang:
        # 오디오 파일을 다시 읽는 대신, 메모리에 있는 waveform에서 바로 잘라냅니다.
        start_sample = int(seg['start'] * sample_rate)
        end_sample = int(seg['end'] * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]

        # 세그먼트가 너무 짧으면(0.5초 미만) 'unknown'으로 처리합니다.
        if segment_waveform.shape[1] < sample_rate * 0.5:
            seg['lang'] = 'ko'
            continue
        
        # 잘라낸 오디오 조각으로 언어를 예측합니다.
        prediction = lang_id_model.classify_batch(segment_waveform)
        
        # 1. 일단 Top 1 언어를 확인합니다.
        top_full_label = prediction[3][0]
        top_lang_code = top_full_label.split(':')[0].strip().lower()

        if top_lang_code in ALLOWED_LANGS:
            # 2. Top 1이 허용 목록에 있으면, 그대로 사용 (가장 빠름)
            seg['lang'] = top_lang_code
        else:
            # 3. Top 1이 허용 목록에 없으면, 전체 확률을 뒤져봅니다.
            print(f"    - [언어 재조정] Top 1 '{top_lang_code}'(이)가 허용 목록에 없음. '{ALLOWED_LANGS}' 내에서 재검색...")

            if (len(prediction) < 1 or
                    not isinstance(prediction[0], torch.Tensor) or
                    prediction[0].numel() == 0):
                print(f"    - [경고] 확률 텐서 없음/비었음 ({seg['start']:.2f}s~{seg['end']:.2f}s). 'ko' 처리.")
                seg['lang'] = 'ko'
                continue

            probabilities = prediction[0]

            allowed_probs = {}
            num_langs_to_check = min(len(probabilities), len(label_encoder.ind2lab))
            for i in range(num_langs_to_check):
                if i not in label_encoder.ind2lab: continue
                label_str = label_encoder.ind2lab[i]
                lang_code = label_str.split(':')[0].strip().lower()

                if lang_code in ALLOWED_LANGS:
                    if i < len(probabilities):
                         prob = probabilities[i].item()
                         allowed_probs[lang_code] = prob

            if allowed_probs:
                final_lang = max(allowed_probs, key=allowed_probs.get)
                seg['lang'] = final_lang
            else:
                seg['lang'] = 'ko'

    print("✅ 언어 감지 완료")
    return segments_with_lang # 데이터 형태 : {'start':..., 'end':..., 'lang':...}




def tag_noise_by_music_blacklist_iterative(vad_segments, ina_segments, waveform, sample_rate, verification_model, threshold=0.4, max_iterations=2):
    print(f"\n🎼 [Iterative Blacklist] 반복 정제 방식으로 음악 제거 시작 (최대 {max_iterations}회 반복)...")
    
    if not vad_segments: return []

    # VAD 데이터 표준화
    if isinstance(vad_segments, Annotation):
        seg_list = [{'start': s.start, 'end': s.end} for s in vad_segments.itersegments()]
    else: seg_list = vad_segments

    total_len = waveform.shape[1]
    
    # === [Step 1] 초기 음악 몽타주 생성 (inaSpeechSegmenter 기반) ===
    # 이곳에 모인 임베딩들이 '음악 기준점'이 됩니다.
    music_embeddings_pool = [] 
    
    for item in ina_segments:
        label = item['label']
        start = item['start']
        end = item['end']
        
        if label == 'music':
            curr = start
            while curr < end:
                chunk_end = min(curr + 5.0, end)
                if chunk_end - curr < 1.0: break 
                
                s_sample = int(curr * sample_rate)
                e_sample = int(chunk_end * sample_rate)
                
                try:
                    # 음악 구간의 임베딩 추출하여 풀(pool)에 저장
                    emb = verification_model.encode_batch(waveform[:, s_sample:e_sample]).flatten()
                    music_embeddings_pool.append(emb)
                except: pass
                curr += 5.0

    if not music_embeddings_pool:
        print("   ⚠️ 초기 음악 구간이 감지되지 않았습니다. 반복 필터링을 중단합니다.")
        return seg_list

    # === [Step 2] 반복 필터링 (Iterative Loop) ===
    for i in range(max_iterations):
        print(f"   🔄 [Round {i+1}] 음악 몽타주 업데이트 및 필터링 중... (현재 표본 수: {len(music_embeddings_pool)}개)")
        
        # 1. 현재 풀(Pool)에 있는 모든 음악 임베딩의 평균(Centroid) 계산
        #    Round가 거듭될수록 1차에서 걸러진 '애매한 음악'들의 특징이 반영됩니다.
        music_centroid = torch.mean(torch.stack(music_embeddings_pool), dim=0)

        tagged_in_this_round = 0
        
        # 2. VAD 세그먼트 전수 조사
        for seg in seg_list:
            # 이미 노이즈/음악으로 판명난 건 건너뛰되, 임베딩 풀에는 기여했음
            if seg.get('audio_type') in ['noise_or_music', 'noise_short', 'noise_music']:
                continue
            
            start = seg['start']
            end = seg['end']
            duration = end - start
            
            # 너무 짧은건 패스 (0.1초 미만)
            if duration < 0.1: continue

            s_sample = int(start * sample_rate)
            e_sample = int(end * sample_rate)
            if e_sample > total_len: e_sample = total_len
            
            try:
                # 현재 검사할 구간의 임베딩
                curr_emb = verification_model.encode_batch(waveform[:, s_sample:e_sample]).flatten()
                
                # 업데이트된 몽타주와 비교
                score = F.cosine_similarity(music_centroid, curr_emb, dim=0).item()
                
                if score >= threshold:
                    # 음악으로 판명!
                    seg['audio_type'] = 'noise_music'
                    seg['music_sim'] = f"{score:.2f}"
                    
                    # 🔥 [핵심] 잡아낸 이 녀석의 임베딩을 다음 라운드 기준점에 추가!
                    music_embeddings_pool.append(curr_emb) 
                    tagged_in_this_round += 1
                else:
                    # 아직은 speech로 유지 (다음 라운드에서 다시 검사 당할 수 있음)
                    if 'audio_type' not in seg:
                        seg['audio_type'] = 'speech'
                        
            except Exception:
                pass
        
        print(f"     👉 Round {i+1} 결과: {tagged_in_this_round}개의 숨겨진 음악 구간 추가 검거.")
        
        # 이번 라운드에서 새로 잡은 게 없으면 더 돌릴 필요 없음
        if tagged_in_this_round == 0:
            print("     ✅ 더 이상 새로운 음악 구간이 발견되지 않아 조기 종료합니다.")
            break

    total_music_count = sum(1 for s in seg_list if s.get('audio_type') == 'noise_music')
    print(f"✅ 최종 필터링 완료. (총 음악 분류: {total_music_count}개)")
    return seg_list




def apply_sandwich_smoothing(segments, max_duration=1.0):
    """
    1초 이하의 짧은 구간이 양옆과 다른 타입일 경우, 양옆의 타입(Context)에 맞춰 변경합니다.
    - Music (Speech) Music -> Music (Speech를 Music으로 변경)
    - Speech (Music) Speech -> Speech (Music을 Speech로 변경)
    """
    print(f"\n🥪 [Smoothing] 샌드위치 규칙 적용 중 (기준: {max_duration}초 이하)...")
    
    if len(segments) < 3:
        return segments

    changed_count = 0
    
    # 리스트의 두 번째부터 뒤에서 두 번째까지 순회 (양옆을 봐야 하니까요)
    for i in range(1, len(segments) - 1):
        prev_seg = segments[i-1]
        curr_seg = segments[i]
        next_seg = segments[i+1]
        
        # 현재 구간의 길이 계산
        duration = curr_seg['end'] - curr_seg['start']
        
        # 1초 초과면 패스
        if duration > max_duration:
            continue

        # 각 구간의 타입 가져오기 (없으면 'speech'로 간주)
        prev_type = prev_seg.get('audio_type', 'speech')
        curr_type = curr_seg.get('audio_type', 'speech')
        next_type = next_seg.get('audio_type', 'speech')

        # Case 1: [음악] - (짧은 말) - [음악] => 말 -> 음악으로 변경
        if curr_type == 'speech' and prev_type == 'noise_music' and next_type == 'noise_music':
            curr_seg['audio_type'] = 'noise_music'
            curr_seg['change_log'] = 'Sandwich Correction (Speech->Music)'
            changed_count += 1
            # print(f"   👉 {curr_seg['start']:.1f}s: 짧은 음성({duration:.2f}s)을 음악 사이에 맞춰 음악으로 변경")

        # Case 2: [말] - (짧은 음악) - [말] => 음악 -> 말로 변경
        elif curr_type == 'noise_music' and prev_type == 'speech' and next_type == 'speech':
            curr_seg['audio_type'] = 'speech'
            curr_seg['change_log'] = 'Sandwich Correction (Music->Speech)'
            changed_count += 1
            # print(f"   👉 {curr_seg['start']:.1f}s: 짧은 음악({duration:.2f}s)을 음성 사이에 맞춰 음성으로 변경")

    print(f"✅ 샌드위치 보정 완료. (총 {changed_count}구간 수정됨)")
    return segments




def select_sub_language(audio_file, lang_map, sorted_list, segment_with_lang):
    """오디오 파일명을 토대로 보조언어를 설정합니다."""
    
    filename = re.sub(r'\s+|_', "", Path(audio_file).stem) # 오디오 파일명에서 띄어쓰기와 _를 제거합니다.
    filename = re.sub(r'0(\d)주차', r'\1주차', filename) # 오디오 파일명의 주차숫자를 정규화합니다.
    
    sub_lang = None
    for prefix in sorted_list:
        if filename.startswith(prefix): # 길이 순서대로 정렬한 리스트를 차례로 순환하며 오디오 파일명이 해당 항목으로 시작하는지를 확인합니다.
            sub_lang = lang_map[prefix] # 찾아내면 생성돼 있는 딕셔너리를 참고해 보조언어를 설정합니다.
            print(f'보조언어를 {sub_lang}으로 설정합니다.')
            break
    
    if sub_lang == None: # 만약 리스트 안에 강의명이 없다면 보조언어를 찾아냅니다.
        print(f'엑셀파일에 해당 강의명이 존재하지 않습니다. 주언어 다음으로 많이 등장한 언어로 보조언어를 설정합니다.')
        lang_list = [seg['lang'] for seg in segment_with_lang if seg['lang'] not in [MAIN_LANGUAGE, 'unknown']] # 세그먼트와 언어정보가 포함된 딕셔너리에서 주언어가 아닌 언어들만 뽑아냅니다.
        if lang_list:
            lang_counts = Counter(lang_list) # 주언어가 아닌 언어들과 그 언어들이 나온 횟수를 튜플 형식으로 반환합니다.
            sub_lang = lang_counts.most_common(1)[0][0] # 주언어가 아니면서 가장 많이 등장한 언어로 보조언어를 설정합니다.            
            print(f'보조언어를 {sub_lang}으로 설정합니다.')
    
    return sub_lang




def define_third_language(segment_with_lang, target_languages):
    """제 3언어를 설정하고 unknown VAD를 이전 VAD에 흡수시킵니다."""
    print('\n 제 3언어를 설정하기 위해 VAD들을 분석합니다.')
    
    set_to_remove = ['unknown']
    set_to_remove = set(set_to_remove) | set(target_languages)
    allowed_langs = set(ALLOWED_LANGS) - set_to_remove
    
    operation_segment_list = segment_with_lang.copy()  # 세그먼트 목록을 혹시 발생할지 모를 변경에서 온전히 보존하기 위해 복사합니다. 
    lang_durations = {}  # 타겟언어와 unknown을 제외한 다른 언어들의 duration을 딕셔너리 형식으로 저장할 빈 딕셔너리를 생성합니다.
    for segment in operation_segment_list:
        if segment['lang'] in allowed_langs:
            lang = segment['lang']
            duration = segment['end'] - segment['start']
            if lang not in lang_durations:
                lang_durations[lang] = duration
            else:
                duration = lang_durations[lang] + duration
                lang_durations[lang] = duration
        else:
            continue
    
    # lang_duration 데이터 형태 : {'en': 9.0, 'ja': 3.0, ...}
                
        
    if lang_durations:
        third_lang = max(lang_durations, key=lang_durations.get)
        if lang_durations[third_lang] < 30: # 제 3언어 길이의 합이 총 30초가 되지 않는다면 제 3언어를 지정하지 않습니다.
            third_lang = None
            print('총합 길이가 가장 긴 제 3언어의 길이가 30초 미만입니다.')
        print(f'제 3언어를 {third_lang}으로 지정합니다.')
    else:
        third_lang = None  # 타겟언어와 unknown을 제외한 다른 언어가 없다면 제 3언어를 지정하지 않습니다.
        print('제 3언어가 존재하지 않습니다. 제 3언어를 지정하지 않습니다.')
        
    return third_lang




def convert_to_unknown(third_lang, segment_with_lang, target_languages):
    """지정된 제 3언어를 받아와 타겟언어와 unknown,  제 3언어를 제외한 언어를 모두 unknown으로 바꿉니다."""

    set_to_remove = ['unknown']
    set_to_remove = set(set_to_remove) | set(target_languages)


    if third_lang == None or third_lang == 'no_sub':
        print('제 3언어가 지정되지 않아 타겟언어와 unknown을 제외한 언어를 모두 unknown으로 바꿉니다.')
        for segment in segment_with_lang:
            if segment['lang'] not in set_to_remove:
                segment['lang'] ='unknown'
    else:
        set_to_remove = set_to_remove | {third_lang}
        print('타겟언어와 unknown, 제 3언어를 제외한 언어를 모두 unknown으로 바꿉니다.')
        for segment in segment_with_lang:
            if segment['lang'] not in set_to_remove:
                segment['lang'] ='unknown'  
    

    return segment_with_lang




def merge_unknown(segment_with_lang):
    ## unknown VAD 흡수작업
    print('unknown VAD를 바로 직전 VAD에 흡수시킵니다.')
    segment_with_lang_unknown_merged = []
    for segment in segment_with_lang:
        if segment == segment_with_lang[0]: # 맨 첫번째 
            segment_with_lang_unknown_merged.append(segment)
        else:
            if segment['lang'] == 'unknown':
                segment_with_lang_unknown_merged[-1]['end'] = segment['end']
            else:
                segment_with_lang_unknown_merged.append(segment)
                
    # segment_with_lang_unknown_merged = [seg for seg in segment_with_lang_unknown_merged if seg['lang'] != 'unknown']
        
        
    return segment_with_lang_unknown_merged




def duration_up_and_down(segment, MAX_DURATION):
    """duration값을 기준값을 기준으로 up and down으로 변환합니다. up은 이상, down은 미만입니다."""
    duration = segment['end'] - segment['start']
    if duration < MAX_DURATION:
        return "down"
    else:
        return "up"




def gap_up_and_down(previous, segment, MAX_GAP):
    """gap값을 기준값을 기준으로 up and down으로 변환합니다. up은 이상, down은 미만입니다."""
    gap = segment['start'] - previous['end']
    if gap < MAX_GAP:
        return "down"
    else:
        return "up"
   



def merge_vad(merge_list, first, case, MAX_DURATION, MAX_MERGED_DURATION):
    """병합하기로 판단된 세그먼트들을 알맞은 형태로 병합하고 최종리스트와 임시리스트를 반환합니다."""

    work_list = []
    temp_list = []
    final_list = []
    if len(merge_list) != 1: # case에 final key가 없는 경우에는
        for i, seg in enumerate(merge_list):
            work_list.append(seg)
            merged_duration = seg['end'] - work_list[0]['start']
            if merged_duration > MAX_MERGED_DURATION: # 이 시간을 넘는 순간
                if i == len(merge_list) - 2 and duration_up_and_down(merge_list[i+1], MAX_DURATION) == 'down': 
                    # 그 순간의 세그먼트가 병합 리스트의 맨 마지막 항목 바로 이전의 항목이면서 마지막 항목의 길이가 최대시간 미만일 경우
                    chunk_1 = {'start':work_list[0]['start'], 'end':merge_list[i+1]['end'], 'lang':case['chunk_1_lang']} 
                    # 굳이 분리하지 않고 그냥 합칩니다.
                    temp_list.append(chunk_1)
                    break

                elif len(work_list) == 1: # 1개 항목 그 자체만으로도 최대 길이를 넘는다면
                    final_list.append(seg) # 그 항목은 바로 최종 리스트로 보냅니다.
                    work_list = [] # 다음 작업을 위해 작업 리스트를 비웁니다.

                elif i == len(merge_list) - 1: # 전부 병합했을 때에만 최대 길이를 넘는다면
                    if duration_up_and_down(merge_list[i], MAX_DURATION) == 'down': # 맨 마지막 항목의 길이가 미만이라면
                        chunk_1 = {'start':work_list[0]['start'], 'end':work_list[-1]['end'], 'lang':case['chunk_1_lang']}
                        temp_list.append(chunk_1) # 전부 합쳐서 임시 리스트에 넣습니다. 어차피 다음 작업에서 맨 처음에 걸러질 겁니다.
                    else: # 맨 마지막 항목의 길이가 이상이라면
                        chunk_1 = {'start':work_list[0]['start'], 'end':work_list[-2]['end'], 'lang':case['chunk_1_lang']}
                        final_list.append(chunk_1) # 그 바로 앞까지만 자른 걸 최종리스트에 보내고
                        temp_list.append(seg) # 길이가 긴 맨 마지막 항목은 임시리스트로 보냅니다.

                else: # 그 외의 경우에는 해당 항목을 작업 리스트에 추가하기 직전까지만 병합합니다.
                    chunk_1 = {'start':work_list[0]['start'], 'end':work_list[-2]['end'], 'lang':case['chunk_1_lang']}
                    final_list.append(chunk_1)
                    work_list = [seg] # 다음 작업을 위해 작업 리스트에 해당 항목만 남겨놓습니다.

            else: # 이 시간을 넘지 않으면
                if i == len(merge_list) - 1: # 전부 병합했을 때에도 최대 길이를 넘지 않는다면
                    chunk_1 = {'start':work_list[0]['start'], 'end':work_list[-1]['end'], 'lang':case['chunk_1_lang']}
                    temp_list.append(chunk_1) # 전부 합쳐서 임시 리스트에 넣습니다.

                else: # 그 외에는 다음 항목을 추가로 받아와 duration을 확인하기 위해 그대로 진행합니다.
                    continue

   
    else: # case에 final key가 있다면
        final_list = [{'start':merge_list[0]['start'], 'end':merge_list[0]['end'], 'lang':case['chunk_1_lang']}]
        temp_list = [{'start':first['start'], 'end':first['end'], 'lang':case['chunk_2_lang']}]

    return final_list, temp_list




def final_merge_VAD_by_lang(segment_with_lang, sub, third, MAX_DURATION, MAX_GAP, MAX_MERGED_DURATION):
    """정리된 세그먼트 목록을 받아 정해진 규칙에 따라 병합합니다."""
    
    if not segment_with_lang:
        return []
    

    case_1 = {'chunk_1_seg':['temp', 'first'], 'chunk_1_lang':'ko', 'temp':'chunk_1'}
    case_2 = {'chunk_1_seg':['temp', 'first'], 'chunk_1_lang':sub, 'temp':'chunk_1'}
    case_3 = {'chunk_1_seg':['temp', 'first'], 'chunk_1_lang':third, 'temp':'chunk_1'}
    case_4 = {'chunk_1_seg':['temp', 'first'], 'chunk_1_lang':'ko', 'temp':'chunk_1'}
    case_5 = {'chunk_1_seg':['temp'], 'chunk_1_lang':'ko', 'chunk_2_seg':['first'], 'chunk_2_lang':'ko', 'final':'chunk_1', 'temp':'chunk_2'}
    case_6 = {'chunk_1_seg':['temp'], 'chunk_1_lang':'ko', 'chunk_2_seg':['first'], 'chunk_2_lang':third, 'final':'chunk_1', 'temp':'chunk_2'}
    case_7 = {'chunk_1_seg':['temp'], 'chunk_1_lang':'ko', 'chunk_2_seg':['first'], 'chunk_2_lang':sub, 'final':'chunk_1', 'temp':'chunk_2'}
    case_8 = {'chunk_1_seg':['temp'], 'chunk_1_lang':sub, 'chunk_2_seg':['first'], 'chunk_2_lang':'ko', 'final':'chunk_1', 'temp':'chunk_2'}
    case_9 = {'chunk_1_seg':['temp'], 'chunk_1_lang':sub, 'chunk_2_seg':['first'], 'chunk_2_lang':third, 'final':'chunk_1', 'temp':'chunk_2'}
    case_10 = {'chunk_1_seg':['temp'], 'chunk_1_lang':sub, 'chunk_2_seg':['first'], 'chunk_2_lang':sub, 'final':'chunk_1', 'temp':'chunk_2'}
    case_11 = {'chunk_1_seg':['temp'], 'chunk_1_lang':third, 'chunk_2_seg':['first'], 'chunk_2_lang':'ko', 'final':'chunk_1', 'temp':'chunk_2'}
    case_12 = {'chunk_1_seg':['temp'], 'chunk_1_lang':third, 'chunk_2_seg':['first'], 'chunk_2_lang':third, 'final':'chunk_1', 'temp':'chunk_2'}
    case_13 = {'chunk_1_seg':['temp'], 'chunk_1_lang':third, 'chunk_2_seg':['first'], 'chunk_2_lang':sub, 'final':'chunk_1', 'temp':'chunk_2'}
    case_14 = {'chunk_1_seg':['temp', 'first', 'second'], 'chunk_1_lang':'ko', 'temp':'chunk_1'}
    case_15 = {'chunk_1_seg':['temp', 'first', 'second'], 'chunk_1_lang':third, 'temp':'chunk_1'}
    case_16 = {'chunk_1_seg':['temp', 'first', 'second'], 'chunk_1_lang':sub, 'temp':'chunk_1'}
    case_17 = {'chunk_1_seg':['temp', 'first', 'second'], 'chunk_1_lang':'ko', 'temp':'chunk_1'}
    case_18 = {'chunk_1_seg':['temp', 'first', 'second', 'third'], 'chunk_1_lang':'ko', 'temp':'chunk_1'}
    case_19 = {'chunk_1_seg':['temp', 'first', 'second', 'third'], 'chunk_1_lang':third, 'temp':'chunk_1'}
    case_20 = {'chunk_1_seg':['temp', 'first', 'second', 'third'], 'chunk_1_lang':'ko', 'temp':'chunk_1'}


    all_case = [
        {'temp_lang':'ko', 'first_gap':'down', 'first_lang':'ko', 'case':case_1}, # 1
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'case':case_1}, # 2
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'up', 'second_lang':'ko', 'case':case_1}, # 3
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'up', 'second_lang':'ko', 'case':case_1}, # 4
        {'temp_lang':'ko', 'temp_dur':'up', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'case':case_1}, # 5
        {'temp_lang':'ko', 'temp_dur':'up', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'up', 'second_lang':'ko', 'case':case_1}, # 6
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'case':case_1}, # 7
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'up', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':'ko', 'case':case_1}, # 8
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'up', 'second_gap':'down', 'case':case_1}, # 9
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'up', 'second_gap':'up', 'case':case_1}, # 10
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'case':case_1}, # 11
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'case':case_1}, # 12
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'up', 'second_lang':'ko', 'case':case_1}, # 13
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'up', 'second_gap':'down', 'case':case_1}, # 14
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'up', 'second_gap':'up', 'case':case_1}, # 15
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'second_dur':'down', 'third_lang':sub, 'case':case_2}, # 16
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'second_dur':'up', 'case':case_2}, # 17
        {'temp_lang':sub, 'first_gap':'down', 'first_lang':sub, 'case':case_2}, # 18
        {'temp_lang':sub, 'temp_dur':'up', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'case':case_2}, # 19
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'third_lang':sub, 'case':case_2}, # 20
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':sub, 'case':case_2}, # 21
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'up', 'second_lang':sub, 'case':case_2}, # 22
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'up', 'second_gap':'down', 'case':case_2}, # 23
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'up', 'second_gap':'up', 'case':case_2}, # 24
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'up', 'second_lang':third, 'case':case_3}, # 25
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'up', 'case':case_3}, # 26
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'up', 'second_lang':third, 'case':case_3}, # 27
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'up', 'case':case_3}, # 28
        {'temp_lang':third, 'first_gap':'down', 'first_lang':third, 'case':case_3}, # 29
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':third, 'case':case_3}, # 30
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'up', 'second_lang':third, 'case':case_3}, # 31
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':third, 'second_dur':'up', 'case':case_4}, # 32
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'up', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':sub, 'case':case_4}, # 33
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'up', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':third, 'case':case_4}, # 34
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'up', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'up', 'third_lang':sub, 'case':case_4}, # 35
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'up', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'up', 'third_lang':third, 'case':case_4}, # 36
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'up', 'second_lang':sub, 'case':case_4}, # 37
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'up', 'second_lang':third, 'case':case_4}, # 38
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'case':case_4}, # 39
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'up', 'second_lang':sub, 'case':case_4}, # 40
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':third, 'case':case_4}, # 41
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'up', 'case':case_4}, # 42
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':'ko', 'case':case_4}, # 43
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'up', 'case':case_4}, # 44
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'up', 'second_lang':'ko', 'case':case_4}, # 45
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'up', 'second_lang':sub, 'case':case_4}, # 46
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'up', 'case':case_4}, # 47
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':'ko', 'case':case_4}, # 48
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':third, 'case':case_4}, # 49
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'up', 'third_lang':'ko', 'case':case_4}, # 50
        {'temp_lang':'ko', 'first_gap':'up', 'first_lang':'ko', 'case':case_5}, #51
        {'temp_lang':'ko', 'first_gap':'up', 'first_lang':third, 'case':case_6}, #52
        {'temp_lang':'ko', 'temp_dur':'up', 'first_gap':'down', 'first_lang':third, 'case':case_6}, #53
        {'temp_lang':'ko', 'first_gap':'up', 'first_lang':sub, 'case':case_7}, #54
        {'temp_lang':'ko', 'first_gap':'down', 'first_lang':sub, 'first_dur':'up', 'case':case_7}, #55
        {'temp_lang':'ko', 'temp_dur':'up', 'first_gap':'down', 'first_lang':sub, 'case':case_7}, #56
        {'temp_lang':sub,  'first_gap':'up', 'first_lang':'ko', 'case':case_8}, #57
        {'temp_lang':sub, 'temp_dur':'up', 'first_gap':'down', 'first_lang':'ko', 'case':case_8}, #58
        {'temp_lang':sub, 'first_gap':'up', 'first_lang':third, 'case':case_9}, #59
        {'temp_lang':sub, 'temp_dur':'up', 'first_gap':'down', 'first_lang':third, 'case':case_9}, #60
        {'temp_lang':sub, 'first_gap':'up', 'first_lang':sub, 'case':case_10}, #61
        {'temp_lang':third, 'first_gap':'up', 'first_lang':'ko', 'case':case_11}, #62
        {'temp_lang':third, 'temp_dur':'up', 'first_gap':'down', 'first_lang':'ko', 'case':case_11}, #63
        {'temp_lang':third, 'first_gap':'up', 'first_lang':third, 'case':case_12}, #64
        {'temp_lang':third, 'first_gap':'up', 'first_lang':sub, 'case':case_13}, #65
        {'temp_lang':third, 'temp_dur':'up', 'first_gap':'down', 'first_lang':sub, 'case':case_13}, #66
        {'temp_lang':'ko', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'case':case_14}, #67
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':third, 'case':case_15}, #68
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':third, 'case':case_15}, #69
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':third, 'case':case_15}, #70
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'case':case_16}, #71
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'second_dur':'down', 'third_gap':'up', 'third_lang':third, 'case':case_16}, #72
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'case':case_16}, #73
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'case':case_16}, #74
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':sub, 'third_dur':'up', 'case':case_17}, #75
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':third, 'second_dur':'down', 'case':case_17}, #76
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'second_dur':'down', 'third_gap':'down', 'third_lang':'ko', 'third_dur':'up', 'case':case_17}, #77
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'second_dur':'down', 'third_gap':'down', 'third_lang':'ko', 'third_dur':'up', 'case':case_17}, #78
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'second_dur':'down', 'third_gap':'down', 'third_lang':third, 'case':case_17}, #79
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'second_dur':'down', 'third_gap':'up', 'third_lang':'ko', 'case':case_17}, #80
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':sub, 'third_dur':'up', 'case':case_17}, #81
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'up', 'case':case_17}, #82
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'up', 'third_lang':sub, 'case':case_17}, #83
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'up', 'third_lang':third, 'case':case_17}, #84
        {'temp_lang':third, 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'up', 'case':case_17}, #85
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':'ko', 'case':case_18}, #86
        {'temp_lang':'ko', 'temp_dur':'up', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':'ko', 'case':case_18}, #87
        {'temp_lang':'ko', 'temp_dur':'up', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'up', 'third_gap':'down', 'third_lang':'ko', 'case':case_18}, #88
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':third, 'case':case_19}, #89
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':sub, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':sub, 'third_dur':'down', 'case':case_20}, #90
        {'temp_lang':'ko', 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'second_dur':'down', 'third_gap':'down', 'third_lang':'ko', 'third_dur':'down', 'case':case_20}, #91
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':'ko', 'first_dur':'down', 'second_gap':'down', 'second_lang':sub, 'second_dur':'down', 'third_gap':'down', 'third_lang':'ko', 'third_dur':'down', 'case':case_20}, #92
        {'temp_lang':sub, 'temp_dur':'down', 'first_gap':'down', 'first_lang':third, 'first_dur':'down', 'second_gap':'down', 'second_lang':'ko', 'second_dur':'down', 'third_gap':'down', 'third_lang':sub, 'third_dur':'down', 'case':case_20} #93
    ]
   


    sorted_all_case = sorted(all_case, key=len, reverse=True)

    final_segment = [] # 병합을 완료한 결과물들을 모아놓을 빈 리스트를 생성합니다.
    temp_list = [] # 병합작업을 실행할 공간을 생성합니다.

   
    temp_list.append(segment_with_lang[0]) # 첫 세그먼트를 임시리스트에 바로 넣습니다.
    work_list = segment_with_lang[1:5] # case 판단을 위한 작업리스트를 만듭니다.
    next_list = segment_with_lang[5:] # 병합작업을 진행하며 항목이 빠져나가는 작업리스트에 빠져나간만큼 다음 항목을 추가할 세그먼트 리스트를 만듭니다.

    while work_list: # 순환종료 기준은 작업리스트입니다.
        # ⚡ [보스의 의문을 해결하는 코드]
        # merge_vad가 temp를 비워서 보냈다는 건, 이전 덩어리가 완결났다는 뜻입니다.
        # 그러므로 대기열(work_list)의 첫 번째 타자를 새로운 기준점(temp)으로 세워야 합니다.
        if not temp_list:
            if work_list:
                # 대기열에서 하나 꺼내서 temp로 승격
                temp_list.append(work_list.pop(0))
                
                # work_list가 하나 줄었으니 next_list에서 하나 충전
                if next_list:
                    work_list.append(next_list.pop(0))
                
                # 기준점이 새로 생겼으니 다시 루프 시작 (Case 판단)
                continue
            
        if len(work_list) >= 4: # 작업리스트의 기본적인 항목 수는 4개입니다.
            temp = temp_list[0] # 임시리스트에 있는 항목을 가져옵니다.
            first = work_list[0]
            second = work_list[1]
            third = work_list[2] # 작업리스트에 있는 항목을 순서대로 가져옵니다.

            sequence = {'temp':temp, 'first':first, 'second':second, 'third':third}
            # 인덱싱을 위한 각 세그먼트 딕셔너리를 만듭니다.
          

            temp_lang = temp['lang']
            temp_dur = duration_up_and_down(temp, MAX_DURATION)
            first_gap = gap_up_and_down(temp, first, MAX_GAP)

            first_lang = first['lang']
            first_dur = duration_up_and_down(first, MAX_DURATION)
            second_gap = gap_up_and_down(first, second, MAX_GAP)

            second_lang = second['lang']
            second_dur = duration_up_and_down(second, MAX_DURATION)
            third_gap = gap_up_and_down(second, third, MAX_GAP)

            third_lang = third['lang']
            third_dur = duration_up_and_down(third, MAX_DURATION)


            temp_dict = {'temp_lang':temp_lang, 'temp_dur':temp_dur, 'first_gap':first_gap, 'first_lang':first_lang, 'first_dur':first_dur, 'second_gap':second_gap, 'second_lang':second_lang, 'second_dur':second_dur, 'third_gap':third_gap, 'third_lang':third_lang, 'third_dur':third_dur}
            # 참고할 모든 값들을 저장할 임시 딕셔너리를 생성합니다.
       

            work_case = {} # 최종적으로 all_case 중 하나와 일치하는 key를 가진 case를 만들기 위해 빈 딕셔너리를 생성합니다.
            selected_case = None # 최종적으로 선택된 case를 저장하기 위한 빈 변수를 생성합니다.
            for case in sorted_all_case: # 93가지의 case를 가장 key를 많이 가진 case부터 하나씩 살펴봅니다.
                key_list = list(case.keys()) # 해당 case가 가진 key들만 뽑아 리스트를 만듭니다.
                key_list.remove('case') # work_case에는 'case' key가 없기 때문에 'case' key를 제거합니다.
                for key in key_list: # 여기에 존재하는 key 명칭을 하나씩 가져와
                    work_case[key] = temp_dict[key] # 참고할 모든 값들을 저장한 임시 딕셔너리 내에서 해당 key, value를 추가합니다.
                temp_case = case.copy() # work_case에는 'case' key가 없기 때문에 본 case를 복사하여 임시 case를 만듭니다.
                del temp_case['case'] # work_case와 일치 여부를 확인하기 위해 'case' key를 제거합니다.
                if work_case == temp_case: # 이 둘이 일치하면
                    selected_case = case['case'] # 본 case의 'case' key 값을 가져와 selected_case에 할당합니다.
                    break
                else : 
                    work_case = {}

            if selected_case == None:
                print(temp_dict)
                print(key_list)
                print(temp_case)
                print(work_case)
       

            merge_list = [sequence[name] for name in selected_case['chunk_1_seg']] # 판단된 case에 해당하는 병합 세그먼트 리스트를 불러옵니다.
            long_segments, temp_segment = merge_vad(merge_list, first, selected_case, MAX_DURATION, MAX_MERGED_DURATION) # 병합 세그먼트 리스트를 최대 길이 이하가 되도록 병합하여 최종리스트와 임시리스트로 반환합니다.
            final_segment += long_segments # 병합 세그먼트 리스트 병합 작업 중 나온 최대 길이 세그먼트를 미리 최종리스트에 추가합니다.
            temp_list = temp_segment # 임시리스트를 병합 세그먼트 리스트 병합 작업 중 나온 임시 세그먼트로 최신화합니다.
            del_count = len(selected_case['chunk_1_seg']) - 1 # 병합작업 후 작업리스트에서 빠져나가는 항목 개수입니다.
            if del_count == 0: # 'chunk_1_seg'가 1개인 경우는 'chunk_2_seg'가 1개인 경우밖에 없기 때문에 0이 되는 순간 1로 바꿔줍니다.
                del_count = 1
            work_list.extend(next_list[:del_count]) # 작업리스트의 뒤에 다음 항목들을 추가합니다.
            del work_list[:del_count]
            del next_list[:del_count] # 작업리스트와 다음리스트에서 추가된 만큼의 항목 수를 제거합니다.


        else: # 병합작업을 진행하다 최후에 작업리스트가 4개 미만이 된다면
            final_segment = final_segment + temp_list + work_list # 임시리스트와 작업리스트를 병합하지 않고 전부 최종리스트에 추가합니다.
            work_list = [] # 작업이 완료되었으니 while 문을 빠져나가기 위한 조건을 만들어 줍니다.

    return final_segment




def redetect_language_for_merged_segments(merged_segments, waveform, sample_rate, lang_id_model, sub_lang, third_lang):
    """
    병합된 세그먼트를 대상으로 언어 감지를 다시 수행합니다.
    단, 전체 허용 목록이 아니라 [한국어 + 보조언어 + 제3언어] 내에서만 결정합니다.
    """
    print("\n🔍 [Re-detection] 병합된 구간 언어 재감지 시작 (타겟 언어 한정)...")
    
    label_encoder = lang_id_model.hparams.label_encoder
    changed_count = 0

    # 1. 재감지 후보군(Target Languages) 설정
    # 무조건 한국어는 포함
    target_langs = {'ko'} 
    if sub_lang:
        target_langs.add(sub_lang)
    if third_lang != None:
        target_langs.add(third_lang)
        
    print(f"   🎯 재감지 후보 언어: {target_langs}")

    # SpeechBrain용 Tensor 변환
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    
    if DEVICE == "cuda":
        waveform = waveform.to(DEVICE)

    for i, seg in enumerate(merged_segments):
        start = seg['start']
        end = seg['end']
        old_lang = seg['lang']
        
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        
        segment_waveform = waveform[:, start_sample:end_sample]
        
        if segment_waveform.shape[1] < sample_rate * 0.1: 
            continue

        try:
            prediction = lang_id_model.classify_batch(segment_waveform)
        except Exception as e:
            continue

        # ----------------------------------------------------------
        # 🎯 타겟 언어 내 필터링 로직 (수정됨)
        # ----------------------------------------------------------
        top_full_label = prediction[3][0]
        top_lang_code = top_full_label.split(':')[0].strip().lower()

        final_new_lang = old_lang 

        # 1순위가 우리 타겟 목록에 있으면 바로 채택
        if top_lang_code in target_langs:
            final_new_lang = top_lang_code
        else:
            # 1순위가 엉뚱한 언어라면, 타겟 목록 중에서 확률이 제일 높은 놈 찾기
            probabilities = prediction[0].squeeze()
            allowed_probs = {}
            
            num_check = min(len(probabilities), len(label_encoder.ind2lab))
            for idx in range(num_check):
                if idx not in label_encoder.ind2lab: continue
                label_str = label_encoder.ind2lab[idx]
                lang_code = label_str.split(':')[0].strip().lower()
                
                # ★ 여기가 핵심: 전체 허용 목록이 아니라, '타겟 목록'에 있는 것만 검사
                if lang_code in target_langs:
                    allowed_probs[lang_code] = probabilities[idx].item()
            
            if allowed_probs:
                final_new_lang = max(allowed_probs, key=allowed_probs.get)
            else:
                # 타겟 언어 확률이 전부 너무 낮으면 그냥 원래 언어 유지
                final_new_lang = old_lang

        # ----------------------------------------------------------
        # 결과 반영
        # ----------------------------------------------------------
        if final_new_lang != old_lang:
            # print(f"   👉 [{i}] 언어 교정: {old_lang} -> {final_new_lang} ({start:.1f}s~{end:.1f}s)")
            seg['lang'] = final_new_lang
            seg['change_log'] = seg.get('change_log', '') + f" | Re-detected ({old_lang}->{final_new_lang})"
            changed_count += 1

    print(f"✅ 재감지 완료. 총 {len(merged_segments)}개 중 {changed_count}개 구간 수정됨.")
    return merged_segments




def merge_subtitle_objects(subs):
    """단어 단위 자막(srt.Subtitle 객체 리스트)을 2단계에 걸쳐 병합합니다."""
    if not subs:
        return []

    # ====================================================
    # 1단계: 문장 및 길이 기반 병합
    # ====================================================
    print("  - [자막 병합] 1차 병합: 문장 및 길이 규칙에 따라 병합 중...")
    pass1_subs = []
    current_sub = subs[0]

    for next_sub in subs[1:]:
        gap = (next_sub.start - current_sub.end).total_seconds()
        current_ends_sentence = current_sub.content.endswith(('.', '?', '!'))
        combined_text = current_sub.content + " " + next_sub.content
        
        # 🔥 [수정 위치 1] 마커인지 확인 (내용이 '###'으로 시작하면 마커임)
        # .strip()을 써서 혹시 모를 공백을 제거하고 확인하는 게 안전합니다.
        is_marker = current_sub.content.strip().startswith('###') or next_sub.content.strip().startswith('###')
        
        should_merge = (
            gap <= MERGE_THRESHOLD_SECONDS and
            not current_ends_sentence and
            len(combined_text) <= MAX_CHARS_PER_LINE and
            not is_marker # 🔥 마커가 포함되어 있으면 절대 합치지 않음
        )

        if should_merge:
            current_sub.end = next_sub.end
            current_sub.content = combined_text
        else:
            pass1_subs.append(current_sub)
            current_sub = next_sub
            
    pass1_subs.append(current_sub)

    # ====================================================
    # 2단계: 지나치게 짧은 세그먼트 정리 (누락 방지 + 마커 보호)
    # ====================================================
    print("  - [자막 병합] 2차 병합: 짧은 세그먼트 정리 중 (누락 방지 적용)...")
    if len(pass1_subs) < 2:
        return pass1_subs

    final_subs = [pass1_subs[0]]
    
    for i in range(1, len(pass1_subs)):
        previous_sub = final_subs[-1]
        current_sub_to_check = pass1_subs[i]
        
        duration = (current_sub_to_check.end - current_sub_to_check.start).total_seconds()
        gap = (current_sub_to_check.start - previous_sub.end).total_seconds()
        
        # 🔥 [수정 위치 2] 여기서도 마커인지 확인해야 합니다.
        # 안 그러면 짧은 '### 미인식 ###' 자막이 앞 문장에 흡수될 수 있습니다.
        is_marker = previous_sub.content.strip().startswith('###') or current_sub_to_check.content.strip().startswith('###')

        # 조건: (짧음 AND 가까움) AND (마커가 아님)
        if duration < MIN_DURATION_SECONDS and gap < 1.0 and not is_marker:
            new_text = previous_sub.content + " " + current_sub_to_check.content
            
            if len(new_text) <= MAX_CHARS_PER_LINE * 1.5:
                previous_sub.end = current_sub_to_check.end
                previous_sub.content = new_text
            else:
                final_subs.append(current_sub_to_check)
        else:
            final_subs.append(current_sub_to_check)
            
    return final_subs




def refine_segments_with_speaker_analysis(segments, waveform, sample_rate, verification_model, sub_lang):
    print(f"\n👑 [Global] 화자 분석 및 역할(Role) 태깅 시작 (보조언어: {sub_lang})")
    
    # === 튜닝 파라미터 ===
    CLUSTERING_THRESHOLD = 0.25      # 화자 구분 유사도 기준
    FOREIGNER_RATIO_LIMIT = 0.6      # 전체 발화 중 60% 이상이 외국어면 '원어민'으로 간주
    INSTRUCTOR_SHORT_LIMIT = 10.0    # 강사의 발화 중 이보다 짧은 보조언어는 한국어로 변경
    
    total_len = waveform.shape[1]
    
    # --- 1단계: 화자 클러스터링 ---
    clusters = [] 
    
    print("   📊 목소리 데이터 수집 및 그룹화 중...")
    
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        lang = seg['lang']
        duration = end - start
        
        is_valid_sample = duration >= 1.0
        
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        if end_sample > total_len: end_sample = total_len
        if start_sample >= end_sample: continue

        seg_waveform = waveform[:, start_sample:end_sample]
        
        if seg_waveform.shape[1] < sample_rate * 0.1:
            current_emb = None
        else:
            current_emb = verification_model.encode_batch(seg_waveform).flatten()
            
        if current_emb is None: continue

        matched_idx = -1
        best_score = -1.0
        
        for c_idx, cluster in enumerate(clusters):
            score = F.cosine_similarity(cluster['centroid'], current_emb, dim=0).item()
            if score > CLUSTERING_THRESHOLD and score > best_score:
                best_score = score
                matched_idx = c_idx
        
        if matched_idx != -1:
            c = clusters[matched_idx]
            n = len(c['ids'])
            c['centroid'] = (c['centroid'] * n + current_emb) / (n + 1)
            c['total_dur'] += duration
            if lang == sub_lang:
                c['sub_lang_dur'] += duration
            c['ids'].append(i)
            
        elif is_valid_sample:
            clusters.append({
                'centroid': current_emb,
                'total_dur': duration,
                'sub_lang_dur': duration if lang == sub_lang else 0.0,
                'ids': [i],
                'role': 'Unassigned'
            })
            
    if not clusters:
        print("⚠️ 화자 분석 실패. 태깅 없이 원본 반환.")
        return segments

    # --- 2단계: 역할(Role) 부여 ---
    clusters.sort(key=lambda x: x['total_dur'], reverse=True)
    
    clusters[0]['role'] = 'Instructor'
    print(f"   🥇 강사(Instructor) 확정: 총 {clusters[0]['total_dur']:.1f}초 발화")

    for c in clusters[1:]:
        ratio = c['sub_lang_dur'] / c['total_dur'] if c['total_dur'] > 0 else 0
        if ratio > FOREIGNER_RATIO_LIMIT:
            c['role'] = 'Native_Speaker'
            print(f"   👽 원어민(Native) 감지: {sub_lang} 비율 {ratio*100:.1f}%")
        else:
            c['role'] = 'Third_Party'
            print(f"   👥 제3자(Third_Party) 분류: 발화량 {c['total_dur']:.1f}초")

    # --- 3단계: 세그먼트에 태깅 및 강사 교정 ---
    changed_count = 0
    
    for c in clusters:
        role = c['role']
        
        for seg_idx in c['ids']:
            seg = segments[seg_idx]
            
            # 1. 화자 역할 태깅
            seg['speaker_role'] = role
            
            # 2. 강사 교정 로직
            if role == 'Instructor':
                lang = seg['lang']
                duration = seg['end'] - seg['start']
                
                # 강사가 쓴 짧은 외국어 -> 한국어로 변경
                if lang == sub_lang and duration < INSTRUCTOR_SHORT_LIMIT:
                    old_lang = lang
                    new_lang = 'ko'
                    
                    print(f"    👉 [교정] {seg['start']:.1f}초: 강사의 짧은 외국어 -> 'ko'로 변경")
                    
                    seg['original_lang'] = old_lang
                    # ★ 변경된 부분: change_log에 구체적인 변경 내역 기록
                    seg['change_log'] = f"Instructor Correction ({old_lang} -> {new_lang})"
                    seg['lang'] = new_lang
                    
                    changed_count += 1
            
            # 원어민/제3자는 건드리지 않음

    print(f"✅ 화자 태깅 및 교정 완료. (총 {changed_count}구간 수정됨)")
    return segments




def run_stt_and_save_srt(waveform, sample_rate, audio_path, segments, output_folder, instructor_prompt, done_path):
    """
    [Human-in-the-loop 모드]
    1. 파일을 물리적으로 저장하여 STT 안정성 확보.
    2. 인식이 안 된 구간(Gap)을 계산하여 '### 미인식 ###' 자막 자동 생성.
    """
    print("\n🚀 4. STT 및 자막 병합 시작 (Gap Detection Mode)...")
    
    if stt_model is None:
        print("❌ Whisper 모델이 로드되지 않아 STT를 진행할 수 없습니다.")
        return

    try:
        all_word_subs = []
        
        # 1. 전체 오디오를 CPU NumPy로 변환
        if isinstance(waveform, torch.Tensor):
            full_audio_np = waveform.squeeze().cpu().numpy()
        else:
            full_audio_np = np.array(waveform).squeeze()
            
        sr = sample_rate
        total_samples = len(full_audio_np)

        # ⚙️ 설정값
        PAD_SECONDS = 0.2        # 앞뒤 여유 (Whisper 인식률 향상용)
        GAP_THRESHOLD = 2.0      # 이 시간(초) 이상 비면 '누락'으로 간주
        
        pad_samples = int(PAD_SECONDS * sr)
        
        # 임시 파일 경로
        temp_wav_path = os.path.join(output_folder, "temp_gap_process.wav")

        for i, seg in enumerate(segments, 1):
            start_time, end_time = seg['start'], seg['end']
            lang = seg['lang'] if seg['lang'] != 'unknown' else None

            print(f"  - [{i}/{len(segments)}] 처리 중: {start_time:.2f}s ~ {end_time:.2f}s ({lang})")

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # 1. 오디오 자르기 (패딩 적용)
            padded_start_sample = max(0, start_sample - pad_samples)
            padded_end_sample = min(total_samples, end_sample + pad_samples)
            
            segment_audio = full_audio_np[padded_start_sample:padded_end_sample]
            
            # 2. 파일 저장 (안전성 확보)
            sf.write(temp_wav_path, segment_audio, sr)
            
            # 3. STT 실행
            result = stt_model.transcribe(
                temp_wav_path,
                language=lang,
                initial_prompt=instructor_prompt if (lang == MAIN_LANGUAGE) else None,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
                fp16=True,
                task='transcribe',
                no_speech_threshold=0.95 # 파일 모드라 기본값에 가깝게 둠 (너무 낮추면 환각 발생)
            )

            # 4. 결과 분석 및 Gap 채우기
            
            # 기준 시간 설정 (패딩된 오디오의 시작점)
            real_start_seconds = padded_start_sample / sr
            offset = timedelta(seconds=real_start_seconds)
            
            # VAD 기준 오디오 길이 (초)
            audio_duration = len(segment_audio) / sr

            # 단어 추출
            words = []
            for s in result.segments:
                for w in s.words:
                    if w.word.strip():
                        words.append(w)
            
            # ==========================================================
            # 🚨 CASE 1: 완전 실패 (Total Fail)
            # ==========================================================
            if not words:
                print(f"    ⚠️ 텍스트 미검출 -> '판독 불가' 마커 생성 ({audio_duration:.2f}s)")
                # VAD 구간 전체를 빈칸 자막으로 생성
                start_ts = offset
                end_ts = offset + timedelta(seconds=audio_duration)
                
                # 패딩 때문에 겹칠 수 있으니 살짝 보정
                content = f"### 판독 불가 구간 ({audio_duration:.1f}s) ###"
                all_word_subs.append(srt.Subtitle(index=0, start=start_ts, end=end_ts, content=content))
                
                continue # 다음 세그먼트로

            # ==========================================================
            # 🚨 CASE 2: 부분 누락 (Partial Gap)
            # ==========================================================
            first_word_start = words[0].start
            last_word_end = words[-1].end
            
            # (A) 앞부분 누락 (Head Gap)
            if first_word_start > GAP_THRESHOLD:
                gap_duration = first_word_start
                print(f"    ⚠️ 앞부분 누락 감지: {gap_duration:.2f}s")
                
                g_start = offset
                g_end = offset + timedelta(seconds=first_word_start)
                all_word_subs.append(srt.Subtitle(index=0, start=g_start, end=g_end, 
                                                  content=f"### 앞부분 미인식 ({gap_duration:.1f}s) ###"))

            # (B) 정상 텍스트 추가
            for w in words:
                start_ts = timedelta(seconds=w.start) + offset
                end_ts = timedelta(seconds=w.end) + offset
                all_word_subs.append(srt.Subtitle(index=0, start=start_ts, end=end_ts, content=w.word.strip()))

            # (C) 뒷부분 누락 (Tail Gap)
            if (audio_duration - last_word_end) > GAP_THRESHOLD:
                gap_duration = audio_duration - last_word_end
                print(f"    ⚠️ 뒷부분 누락 감지: {gap_duration:.2f}s")
                
                g_start = offset + timedelta(seconds=last_word_end)
                g_end = offset + timedelta(seconds=audio_duration)
                all_word_subs.append(srt.Subtitle(index=0, start=g_start, end=g_end, 
                                                  content=f"### 뒷부분 미인식 ({gap_duration:.1f}s) ###"))

        # 5. 임시 파일 삭제
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

        if not all_word_subs:
            print("  - [경고] 생성된 자막이 없습니다.")
            return

        # STT로 생성된 단어 자막들을 병합
        merged_subs = merge_subtitle_objects(all_word_subs)
        
        for idx, sub in enumerate(merged_subs, 1):
            sub.index = idx

        srt_content = srt.compose(merged_subs)
        base_filename = Path(audio_path).stem
        srt_filename = f"{base_filename}{FILENAME_SUFFIX}.srt"
        srt_filepath = Path(output_folder) / srt_filename

        with open(srt_filepath, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        print(f"✅ 최종 병합 SRT 파일 저장 완료: {srt_filepath}")

    except Exception as e:
        print(f"❌ STT/병합 처리 중 치명적인 오류 발생: {e}")
        traceback.print_exc()
    finally:
        # 청소
        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
            try: os.remove(temp_wav_path)
            except: pass

        if done_path:
            print("  - 작업 완료 후 파일 이동을 시도합니다.")
            try:
                shutil.move(audio_path, done_path)
                print(f"✅ 원본 WAV 파일 이동 완료: {Path(done_path) / Path(audio_path).name}")
            except Exception as e:
                 print(f"❌ 파일 이동 중 오류 발생: {e}")
        else:
            print("✅ 작업 완료. 파일은 원본 위치에 유지됩니다.")
            
            
            
            
def run_stt_and_save_srt_no_file(waveform, sample_rate, audio_path, segments, output_folder, instructor_prompt, done_path):
    """
    [Human-in-the-loop 모드 - 메모리 가속 버전]
    1. 파일을 생성하지 않고 메모리(NumPy)에서 직접 처리하여 속도를 높입니다.
    2. 'int16 정규화' 트릭을 사용하여 파일 저장 방식과 동일한 인식률을 확보합니다.
    3. 인식이 안 된 구간(Gap)을 계산하여 '### 미인식 ###' 자막을 자동 생성합니다.
    """
    print("\n🚀 4. STT 및 자막 병합 시작 (Memory Gap Detection Mode)...")
    
    if stt_model is None:
        print("❌ Whisper 모델이 로드되지 않아 STT를 진행할 수 없습니다.")
        return

    try:
        all_word_subs = []
        
        # 1. 전체 오디오를 CPU NumPy로 변환 (한 번만 수행)
        if isinstance(waveform, torch.Tensor):
            full_audio_np = waveform.squeeze().cpu().numpy()
        else:
            full_audio_np = np.array(waveform).squeeze()
            
        sr = sample_rate
        total_samples = len(full_audio_np)

        # ⚙️ 설정값
        PAD_SECONDS = 0.2        # 앞뒤 여유
        GAP_THRESHOLD = 2.0      # 누락 판단 기준 (초)
        
        pad_samples = int(PAD_SECONDS * sr)
        
        # (파일 경로 생성 로직 삭제됨)

        for i, seg in enumerate(segments, 1):
            start_time, end_time = seg['start'], seg['end']
            lang = seg['lang'] if seg['lang'] != 'unknown' else None

            print(f"  - [{i}/{len(segments)}] 처리 중: {start_time:.2f}s ~ {end_time:.2f}s ({lang})")

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # 1. 오디오 자르기 (패딩 적용)
            padded_start_sample = max(0, start_sample - pad_samples)
            padded_end_sample = min(total_samples, end_sample + pad_samples)
            
            segment_audio = full_audio_np[padded_start_sample:padded_end_sample]
            
            # 혹시 모를 차원 축소
            if segment_audio.ndim > 1:
                segment_audio = segment_audio.flatten()

            # =======================================================
            # 🔥 [핵심] "가상 파일 저장" 효과 (Int16 Quantization)
            # 파일을 직접 저장하지 않고도, 저장한 것과 똑같은 음질 상태로 만듭니다.
            # =======================================================
            # 1. -1.0 ~ 1.0 클리핑
            segment_audio = np.clip(segment_audio, -1.0, 1.0)
            # 2. int16 변환 (파일 저장 효과)
            segment_audio_int16 = (segment_audio * 32767).astype(np.int16)
            # 3. float32 복구 (Whisper 입력용)
            segment_audio_clean = segment_audio_int16.astype(np.float32) / 32767.0
            # =======================================================
            
            # 3. STT 실행 (NumPy 배열 직접 입력)
            result = stt_model.transcribe(
                segment_audio_clean,  # <- 파일 경로 대신 배열 입력
                language=lang,
                initial_prompt=instructor_prompt if (lang == MAIN_LANGUAGE) else None,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
                fp16=True,
                task='transcribe',
                no_speech_threshold=0.95 
            )

            # 4. 결과 분석 및 Gap 채우기 (로직 동일)
            
            # 기준 시간 설정
            real_start_seconds = padded_start_sample / sr
            offset = timedelta(seconds=real_start_seconds)
            
            # 오디오 길이 (초)
            audio_duration = len(segment_audio_clean) / sr

            # 단어 추출
            words = []
            for s in result.segments:
                for w in s.words:
                    if w.word.strip():
                        words.append(w)
            
            # ----------------------------------------------------------
            # 🚨 CASE 1: 완전 실패 (Total Fail)
            # ----------------------------------------------------------
            if not words:
                print(f"    ⚠️ 텍스트 미검출 -> '판독 불가' 마커 생성 ({audio_duration:.2f}s)")
                start_ts = offset
                end_ts = offset + timedelta(seconds=audio_duration)
                content = f"### 판독 불가 구간 ({audio_duration:.1f}s) ###"
                all_word_subs.append(srt.Subtitle(index=0, start=start_ts, end=end_ts, content=content))
                continue 

            # ----------------------------------------------------------
            # 🚨 CASE 2: 부분 누락 (Partial Gap)
            # ----------------------------------------------------------
            first_word_start = words[0].start
            last_word_end = words[-1].end
            
            # (A) 앞부분 누락 (Head Gap)
            if first_word_start > GAP_THRESHOLD:
                gap_duration = first_word_start
                print(f"    ⚠️ 앞부분 누락 감지: {gap_duration:.2f}s")
                g_start = offset
                g_end = offset + timedelta(seconds=first_word_start)
                all_word_subs.append(srt.Subtitle(index=0, start=g_start, end=g_end, 
                                                  content=f"### 앞부분 미인식 ({gap_duration:.1f}s) ###"))

            # (B) 정상 텍스트 추가
            for w in words:
                start_ts = timedelta(seconds=w.start) + offset
                end_ts = timedelta(seconds=w.end) + offset
                all_word_subs.append(srt.Subtitle(index=0, start=start_ts, end=end_ts, content=w.word.strip()))

            # (C) 뒷부분 누락 (Tail Gap)
            if (audio_duration - last_word_end) > GAP_THRESHOLD:
                gap_duration = audio_duration - last_word_end
                print(f"    ⚠️ 뒷부분 누락 감지: {gap_duration:.2f}s")
                g_start = offset + timedelta(seconds=last_word_end)
                g_end = offset + timedelta(seconds=audio_duration)
                all_word_subs.append(srt.Subtitle(index=0, start=g_start, end=g_end, 
                                                  content=f"### 뒷부분 미인식 ({gap_duration:.1f}s) ###"))

        # (파일 삭제 로직 제거됨)

        if not all_word_subs:
            print("  - [경고] 생성된 자막이 없습니다.")
            return

        # STT로 생성된 단어 자막들을 병합
        merged_subs = merge_subtitle_objects(all_word_subs)
        
        for idx, sub in enumerate(merged_subs, 1):
            sub.index = idx

        srt_content = srt.compose(merged_subs)
        base_filename = Path(audio_path).stem
        srt_filename = f"{base_filename}{FILENAME_SUFFIX}.srt"
        srt_filepath = Path(output_folder) / srt_filename

        with open(srt_filepath, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        print(f"✅ 최종 병합 SRT 파일 저장 완료: {srt_filepath}")

    except Exception as e:
        print(f"❌ STT/병합 처리 중 치명적인 오류 발생: {e}")
        traceback.print_exc()
    finally:
        # (파일 이동 로직은 동일)
        if done_path:
            print("  - 작업 완료 후 파일 이동을 시도합니다.")
            try:
                shutil.move(audio_path, done_path)
                print(f"✅ 원본 WAV 파일 이동 완료: {Path(done_path) / Path(audio_path).name}")
            except Exception as e:
                 print(f"❌ 파일 이동 중 오류 발생: {e}")
        else:
            print("✅ 작업 완료. 파일은 원본 위치에 유지됩니다.")
            



def run_stt_and_save_srt_no_merge(waveform, sample_rate, audio_path, segments, output_folder, instructor_prompt, done_path):
    """STT 수행 후, 병합 없이 단어 단위(Word-level) 자막을 그대로 저장합니다."""
    print("\n🚀 4. STT 및 단어 단위 자막 생성 시작...")
    
    if stt_model is None:
        print("❌ Whisper 모델이 로드되지 않아 STT를 진행할 수 없습니다.")
        return

    try:
        all_word_subs = []
        audio_waveform, sr = waveform, sample_rate

        for i, seg in enumerate(segments, 1):
            start_time, end_time = seg['start'], seg['end']
            lang = seg['lang'] if seg['lang'] != 'unknown' else None

            print(f"   - [{i}/{len(segments)}] STT 구간 처리 중: {start_time:.2f}s ~ {end_time:.2f}s (언어: {lang or '자동 감지'})")

            start_sample, end_sample = int(start_time * sr), int(end_time * sr)
            # segment_audio = audio_waveform[:, start_sample:end_sample][0]
            segment_tensor = audio_waveform[:, start_sample:end_sample][0]
            
            # 1. 텐서가 GPU에 있다면 CPU로 내림
            if isinstance(segment_tensor, torch.Tensor):
                segment_audio = segment_tensor.cpu().numpy()
            else:
                segment_audio = np.array(segment_tensor)
            
            # 2. 데이터 타입을 float32로 강제 변환 (Whisper가 가장 좋아하는 포맷)
            segment_audio = segment_audio.astype(np.float32)

            result = stt_model.transcribe(
                segment_audio,
                language=lang,
                initial_prompt=instructor_prompt if (lang == MAIN_LANGUAGE) else None,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
                fp16=True,
                task='transcribe'
            )

            offset = timedelta(seconds=start_time)
            
            # 단어 단위 정보를 리스트에 모두 담습니다.
            for segment in result.segments:
                for word in segment.words:
                    start_ts = timedelta(seconds=word.start) + offset
                    end_ts = timedelta(seconds=word.end) + offset
                    content = word.word.strip()
                    
                    if content:
                        # index는 나중에 일괄적으로 매깁니다 (0으로 임시 저장)
                        all_word_subs.append(srt.Subtitle(index=0, start=start_ts, end=end_ts, content=content))
        
        if not all_word_subs:
            print("   - [경고] STT 결과 텍스트가 없어 SRT 파일을 생성하지 않습니다.")
            return

        # ▼▼▼ [수정된 부분] 자막 병합 로직 제거 ▼▼▼
        # merged_subs = merge_subtitle_objects(all_word_subs) <--- 이 줄을 삭제/주석 처리했습니다.
        
        print(f"   - 자막 병합을 건너뜁니다. 총 {len(all_word_subs)}개의 단어가 저장됩니다.")
        
        # 단어 리스트의 인덱스(순번)를 1부터 차례대로 매깁니다.
        for idx, sub in enumerate(all_word_subs, 1):
            sub.index = idx

        # 병합된 자막 대신 단어 자막(all_word_subs)을 바로 사용합니다.
        srt_content = srt.compose(all_word_subs)
        
        base_filename = Path(audio_path).stem
        # 파일명에 _WORDS 등을 붙여서 구분을 짓고 싶으시면 아래 줄을 수정하세요.
        srt_filename = f"{base_filename}{FILENAME_SUFFIX}.srt" 
        srt_filepath = Path(output_folder) / srt_filename

        with open(srt_filepath, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        print(f"✅ 최종 단어 단위 SRT 파일 저장 완료: {srt_filepath}")

    except Exception as e:
        print(f"❌ STT 처리 중 치명적인 오류 발생: {e}")
        traceback.print_exc()
    finally:
        if done_path:
            print("   - 작업 완료 후 파일 이동을 시도합니다.")
            try:
                shutil.move(audio_path, done_path)
                print(f"✅ 원본 WAV 파일 이동 완료: {Path(done_path) / Path(audio_path).name}")
            except shutil.Error as move_e:
                print(f"❌ 파일 이동 중 오류 발생: {move_e}")
            except Exception as e:
                 print(f"❌ 파일 이동 중 예기치 않은 오류 발생: {e}")
        else:
            print("✅ 작업 완료. 파일은 원본 위치에 유지됩니다.")
            
            
            

def vad_annotation_to_srt_empty_with_lang(final_merged, output_folder, audio_path, file_suffix):
    print("\n📄 4. SRT 파일 생성 시작...")

    subs = []
    idx = 1
    for seg in final_merged:
        start, end = float(seg['start']), float(seg['end'])
        
        start_td, end_td = timedelta(seconds=start), timedelta(seconds=end)
        if end_td <= start_td: end_td = start_td + timedelta(milliseconds=10)
        
        content_items = []
        for key, value in seg.items():
            if key not in ['start', 'end']:
                content_items.append(f"{key}: {value}")
                
        if not content_items:
            content_str = " "
        else:
            content_str = "|".join(content_items)
                
        subs.append(srt.Subtitle(index=idx, start=start_td, end=end_td, content=content_str))
        idx += 1
        
    print(f"   - 최종 SRT에 포함될 구간 수: {len(subs)}개")
    srt_text = srt.compose(subs)
    
    base_filename = Path(audio_path).stem # 원본 파일명 (확장자 제외)
    srt_filename = f"{base_filename}{FILENAME_SUFFIX}_{file_suffix}.srt" # 새 파일명
    srt_filepath = Path(output_folder) / srt_filename # 폴더 경로와 파일명 조합
    Path(srt_filepath).write_text(srt_text, encoding="utf-8")
    if not subs: print("\n[⚠️ 경고] 최종 SRT 파일에 포함된 구간이 없습니다. 파일이 비어있을 수 있습니다.")
    print(f"✅ SRT 저장 완료: {srt_filepath} (총 {len(subs)}구간)")




# ========== 메인 실행 로직 ==========
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    # 🔥 [수정 1] 변수 초기화 (안전장치)
    # 엑셀을 선택하지 않더라도 변수가 존재해야 나중에 에러가 안 납니다.
    lang_map = {}
    sorted_list = []

    print("\n📊 강의 정보가 담긴 엑셀 파일이 있다면 선택하세요. (취소 시 자동 감지 모드로 동작)")
    xlsx_file_path = filedialog.askopenfilename(title="xlsx 파일 선택 (선택 안 함: 취소)", filetypes=[("xlsx File", "*.xlsx")])
    
    if xlsx_file_path:
        try:
            lang_map, sorted_list = read_xlsx_and_create_dict(xlsx_file_path)
            print(f"✅ 엑셀 파일 로드 완료: {len(lang_map)}개의 강의 정보")
        except Exception as e:
            print(f"⚠️ 엑셀 파일 읽기 실패 (자동 감지 모드로 전환): {e}")
    else:
        print("⚠️ 엑셀 파일이 선택되지 않았습니다. 보조언어는 오디오 분석을 통해 자동 감지됩니다.")

    if lang_id_model is not None and stt_model is not None:
        print("\n🎵 분석할 WAV 파일을 선택하세요 (16kHz, mono 권장, 다중 선택 가능)")
        audio_files = filedialog.askopenfilenames(
            title="WAV 파일 선택 (다중 선택 가능)",
            filetypes=[("WAV Files", "*.wav")]
        )

        if not audio_files:
            print("❌ 파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
        else:
            print(f"\n총 {len(audio_files)}개의 파일을 선택했습니다.")
            print("\n💾 결과 SRT 파일을 저장할 폴더를 선택하세요.")
            output_path = filedialog.askdirectory(title="SRT 파일을 저장할 폴더 선택")
            
            print("\n📂 완료된 WAV 파일을 이동시킬 폴더를 선택하세요 (취소 시 이동 안 함).")
            done_path = filedialog.askdirectory(title="완료된 WAV 파일을 이동시킬 폴더 선택")

            if not output_path:
                print("❌ 결과 저장 폴더가 선택되지 않았습니다. 프로그램을 종료합니다.")
            else:
                if not done_path:
                    print("\n⚠️ '완료' 폴더가 선택되지 않았습니다. 처리된 파일은 원본 위치에 그대로 남습니다.")

                for i, audio_file in enumerate(list(audio_files), 1):
                    if not os.path.exists(audio_file):
                        print(f"⚠️ [{i}/{len(audio_files)}] 파일 '{os.path.basename(audio_file)}'가 이미 이동되었거나 존재하지 않아 건너뜁니다.")
                        continue

                    print(f"\n{'='*60}")
                    print(f"▶️  [{i}/{len(audio_files)}] 파일 처리 시작: {os.path.basename(audio_file)}")
                    print(f"{'='*60}")

                    non_silent_segments = get_non_silent_segments_ffmpeg(audio_file)
                    # ffmpeg를 이용해 먼저 침묵구간을 제외합니다.
                    
                    # ---------- NEW: 음악 구간 감지 + 병합 ----------
                    ina_segments = detect_music_segments(audio_file)          # (label, start, end) 리스트
                    music_blocks = build_music_blocks(ina_segments, short_speech_max=1.0)  # 1초 이하 speech는 음악으로 흡수

                    # ffmpeg 결과가 "full_audio"인 경우(침묵 못 찾은 경우)
                    if non_silent_segments == "full_audio":
                        if music_blocks:
                            # 전체 길이에서 음악 블록만 빼고 나머지만 사용
                            try:
                                audio_for_duration = AudioSegment.from_file(str(audio_file))
                                total_dur = len(audio_for_duration) / 1000.0  # ms → sec
                            except Exception as e:
                                print(f"   [경고] 오디오 길이 계산 실패: {e}")
                                total_dur = None

                            if total_dur is not None:
                                base_segments = [{"start": 0.0, "end": total_dur}]
                                non_silent_segments = remove_music_from_non_silent(base_segments, music_blocks)
                            else:
                                # 길이 계산 실패하면 그냥 full_audio 유지
                                pass
                    else:
                        # 평소에는 ffmpeg 유성 구간에서 음악 블록 제거
                        non_silent_segments = remove_music_from_non_silent(non_silent_segments, music_blocks)

                    # 음악 제거 후 남은 유성 구간이 없으면 이 파일은 STT를 스킵
                    if not non_silent_segments:
                        print("   [정보] 음악 구간을 제거하고 나니 남는 구간이 없습니다. 이 파일 STT 건너뜀.")
                        continue

                    if non_silent_segments: # FFmpeg 분석이 성공적으로 끝나면
                        print("\n🎵 오디오 파일 로딩 중...")
                        try:
                            audio_loader = Audio(sample_rate=16000, mono=True)
                            waveform, sample_rate = audio_loader(audio_file)
                            print("✅ 오디오 파일 로딩 완료.")
                        except Exception as e:
                            print(f"[치명적 오류] 오디오 파일 로딩에 실패했습니다: {e}")
                            exit()
                    
                    vad_annotation = extract_segments_2stage(waveform, sample_rate, non_silent_segments) 
                    # VAD 기반 세그먼트를 추출합니다.
                    segment_with_lang = detect_language_for_vad_segments(vad_annotation, waveform, sample_rate, lang_id_model) 
                    # pyannote VAD Annotation 결과와 이미 로드된 waveform을 사용해 각 세그먼트의 언어를 감지합니다.
                    segment_with_lang_and_music = tag_noise_by_music_blacklist_iterative(segment_with_lang, ina_segments, waveform, sample_rate, verification_model, threshold=0.4, max_iterations=2)
                    
                    segment_with_lang_and_music2 = apply_sandwich_smoothing(segment_with_lang_and_music, max_duration=1.0)
                    print(f"🧹 음악 제거 전: {len(segment_with_lang)}개")
                    
                    
                    segment_with_lang = [seg for seg in segment_with_lang_and_music2 if seg.get('audio_type') != 'noise_music']
                    print(f"🧹 음악 제거 후: {len(segment_with_lang)}개")
                    
                    
                    # sub_language = None if (SUB_LANGUAGE==None) else SUB_LANGUAGE
                    instructor_prompt = None
                    target_languages = [MAIN_LANGUAGE]
                    if SUB_LANGUAGE == None:
                        sub_language = select_sub_language(audio_file, lang_map, sorted_list, segment_with_lang) 
                        # 오디오 파일명을 토대로 보조언어를 설정합니다.
                    elif SUB_LANGUAGE == "no_sub":
                        sub_language = None
                    else:
                        sub_language = SUB_LANGUAGE

                    if sub_language != None:
                        target_languages.append(sub_language)
                        
                    instructor_prompt = INSTRUCTOR_PROMPT_DICT.get(sub_language)
                    # 설정된 보조언어를 기준으로 적용할 프롬프트를 선정합니다.

                    if THIRD_LANG == 'no_third':
                        third_lang = None
                    else:
                        third_lang = define_third_language(segment_with_lang, target_languages)
                    # 제 3언어를 설정합니다.

                    segment_with_lang_still_unknown_in = convert_to_unknown(third_lang, segment_with_lang, target_languages)
                    # 제 3언어와 타겟언어, unknown을 제외한 언어는 모두 unknown으로 바꿉니다.
                    segment_with_lang_without_unknown = merge_unknown(segment_with_lang_still_unknown_in)
                    # 언어가 unknown인 세그먼트는 직전 세그먼트에 흡수시킵니다.

     
                    
                    if segment_with_lang:
                        # final_segment = refine_segments_with_speaker_analysis(segment_with_lang_without_unknown, waveform, sample_rate, verification_model, sub_language)
                        final_segment_2 = final_merge_VAD_by_lang(segment_with_lang_without_unknown, sub_language, third_lang, MAX_DURATION, MAX_GAP, MAX_MERGED_DURATION)
                        final_segment_3 = redetect_language_for_merged_segments(final_segment_2, waveform, sample_rate, lang_id_model, sub_language, third_lang)
                        run_stt_and_save_srt_no_file(waveform, sample_rate, audio_file, final_segment_3, output_path, instructor_prompt, done_path)
                        # run_stt_and_save_srt_no_merge(waveform, sample_rate, audio_file, final_segment_2, output_path, instructor_prompt, done_path)
                        # 자막병합 전 단어단위 srt 확인 코드
                        # vad_annotation_to_srt_empty_with_lang(segment_with_lang, output_path, audio_file, file_suffix="vad_lang")
                        # vad_annotation_to_srt_empty_with_lang(segment_with_lang_and_music2, output_path, audio_file, file_suffix="vad_lang_with_music")
                        # vad_annotation_to_srt_empty_with_lang(segment_with_lang_still_unknown_in, output_path, audio_file, file_suffix="vad_lang_with_unknown")
                        # vad_annotation_to_srt_empty_with_lang(segment_with_lang_without_unknown, output_path, audio_file, file_suffix="vad_lang_without_unknown")
                        # vad_annotation_to_srt_empty_with_lang(final_segment, output_path, audio_file, file_suffix="merged_vad")
                        # vad_annotation_to_srt_empty_with_lang(final_segment_2, output_path, audio_file, file_suffix="merged_vad_with_speaker")
                        # vad_annotation_to_srt_empty_with_lang(final_segment_3, output_path, audio_file, file_suffix="merged_vad_with_speaker_2")
                        # STT 직전의 세그먼트를 확인할 때 사용할 코드
                    
                    else:
                        print("❌ STT를 진행할 유효한 구간이 없습니다.")
                        if done_path:
                            print("  - 원본 파일을 '완료' 폴더로 이동합니다.")
                            try:
                                shutil.move(audio_file, done_path)
                                print(f"  - 파일 이동 완료: {Path(done_path) / Path(audio_file).name}")
                            except Exception as e:
                                print(f"  - 파일 이동 중 오류 발생: {e}")
                        else:
                            print("✅ 작업 완료. 파일은 원본 위치에 유지됩니다.")


                print("\n\n🎉 모든 파일 처리가 완료되었습니다.")
    else:
        print("\n❌ 필수 모델 중 일부가 로드되지 않아 프로그램을 종료합니다.")