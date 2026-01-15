import os
import re
import math
import shutil
import subprocess
import traceback
from pathlib import Path
from datetime import timedelta
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import tkinter as tk
from tkinter import filedialog

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment

import srt
from pyannote.audio import Pipeline, Audio
from pyannote.core import Annotation, Segment
from speechbrain.inference import EncoderClassifier, SpeakerRecognition
from inaSpeechSegmenter import Segmenter

from openai import OpenAI


# ============================================================
# 0) 사용자 설정 파라미터
# ============================================================

PUNCT_ATTACH_TO_PREV = r""".,!?;:，。！？；：…"""
CLOSERS_ATTACH_TO_PREV = r""")\]\}〉》」』】"”’'"""

# --- (A) 토큰/키 ---
HF_TOKEN = os.getenv("HF_TOKEN")  # 예: hf_xxx
# OpenAI API Key는 환경변수 OPENAI_API_KEY 사용 (OpenAI SDK가 자동 인식)

# --- (B) 디바이스 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- (C) 외부 도구 ---
FFMPEG_PATH = r"C:\Users\cufs\Desktop\업무\subtitle\ffmpeg\ffmpeg.exe"  # ffmpeg.exe 전체 경로

# --- (D) 언어/파일명 ---
FILENAME_SUFFIX = ""     # 최종 SRT 파일명 접미사
MAIN_LANGUAGE = "ko"
SUB_LANGUAGE = None      # None: 자동 / "no_sub": 보조언어 없음 / 'en','vi' 등: 고정
THIRD_LANG = "no_third"  # "no_third": 제3언어 없음 / 그 외: 로직 수행
ALLOWED_LANGS = ['ko', 'en', 'vi', 'es', 'zh', 'ja', 'id']

# --- (E) FFmpeg 침묵 탐지 파라미터 ---
SILENCE_THRESH_DB = -50
MIN_SILENCE_DURATION_S = 0.05

# --- (F) VAD 파라미터 ---
VAD_PARAMS = {
    "min_duration_off": 0.01,
    "min_duration_on": 0.05,
    "onset": 0.01,
    "offset": 0.01
}

MAX_DURATION = 1
MAX_GAP = 0.5
MAX_MERGED_DURATION = 600

# 최종 SRT 생성 및 병합 파라미터
MIN_SEGMENT_DURATION = 0.1
MERGE_MAX_SECONDS = 15.0

# --- (G) 음악/노이즈 필터 ---
MUSIC_SIM_THRESHOLD = 0.4
MUSIC_MAX_ITER = 2
SANDWICH_MAX_DURATION = 1.0

# --- (H) OpenAI diarize STT 설정 (여기만 건드리면 됨) ---
OPENAI_MODEL = "gpt-4o-transcribe-diarize"
SUBTITLE_MAX_CHARS = 35          # ✅ 최대 글자수 (기본 35)
SUBTITLE_ONE_LINE = True         # ✅ 항상 1줄
SUBTITLE_MIN_CUE_DUR = 0.25      # 너무 짧은 cue 최소 길이 (초)
OPENAI_PAD_SECONDS = 0.0        # 세그먼트 앞뒤 여유
OPENAI_MAX_CHUNK_SECONDS = 600.0  # 세그먼트가 너무 길면 내부에서 이 길이로 잘라서 여러 번 호출

INCLUDE_SPEAKER_PREFIX = False   # 원하면 [SPEAKER_00] 같은 prefix 붙이기

# --- 프롬프트 딕셔너리(유지: diarize 모델은 prompt 안 받지만, 네 로직 구조상 남겨둠) ---
en = "Today, we will discuss the importance of renewable energy. The quick brown fox jumps over the lazy dog."
ja = "테형 뒤에 이루를 붙이면 진행형이 돼. 시테 이루는 '하고 있다'라는 뜻이야. 오스스메 메뉴가 뭐예요? 나마비루 두 잔 주세요. 사이후를 잃어버려서 케이사츠에 신고했어."
vi = "씬짜오 깜언 또이 드억조이 퍼 반미 응온 아잉 엠 자오비엔 바오니에우"
es = "이것은 한국어와 스페인어를 사용하는 스페인어 문법 강의입니다. Me llamo Juan. ¿Dónde está la biblioteca? Te quiero mucho."
idn = "슬라맛 빠기 뜨리마 까시 아빠 까바르 쁘르기 싸야 팅갈 디 서울 뜨리마 까시 바냑 삼빠이 줌빠 라기"
zh = "오늘은 把字句랑 被字句를 비교할 거야. 把字句는 처분 강조, 被字句는 피동.三号出口에서 만나. 택시는 打车, 갈아타기는 换乘."
ko = ""
INSTRUCTOR_PROMPT_DICT = {'vi': vi, 'es': es, 'id': idn, 'zh': zh, 'en': en, 'ja': ja, 'ko': ko}


# ============================================================
# 1) Symlink 우회 Patch (SpeechBrain Windows 호환)
# ============================================================

def force_copy(src, dst):
    if src is None or dst is None:
        return None
    src_path, dst_path = Path(src), Path(dst)
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
        return dst
    except Exception as e:
        print(f"   [경고] 파일 복사 중 오류 발생: {e}")
        return None

import speechbrain.utils.fetching as sb_fetch
sb_fetch.link_with_strategy = lambda src, dst, strategy: force_copy(src, dst)


# ============================================================
# 2) 모델 로딩 (시작 시 1회)
# ============================================================

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN 환경변수가 비어있음. HF_TOKEN을 환경변수로 설정해줘.")

print("🔄 모델 로딩 중... (VAD, Language ID, Speaker Verification, Music Segmenter)")
# 1) VAD
vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=HF_TOKEN)
vad_pipeline.to(torch.device(DEVICE))
vad_pipeline.instantiate(VAD_PARAMS)
print("✅ VAD 모델 로딩 완료.")

# 2) Language ID
lang_id_model = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp_lang_id"
)
print("✅ Language ID 모델 로딩 완료.")

# 3) Speaker Verification (음악/노이즈 검출용 임베딩)
verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_speaker_verification",
    run_opts={"device": DEVICE}
)
print("✅ Speaker Verification 모델 로딩 완료.")

# 4) Music/Speech Segmenter
music_segmenter = Segmenter(vad_engine="smn", detect_gender=False)
print("✅ Music/Speech 세그먼터 로딩 완료.")

# 5) OpenAI Client
client = OpenAI()
print("✅ OpenAI 클라이언트 준비 완료.")


# ============================================================
# 3) 유틸 함수들 (엑셀/FFmpeg/음악/VAD/언어감지/병합)
# ============================================================

def read_xlsx_and_create_dict(xlsx_file_path):
    """강의명과 보조언어 매칭 엑셀 -> dict 생성"""
    df = pd.read_excel(io=xlsx_file_path, header=3, usecols="C:D")
    df = df.dropna(subset=['보조언어'])
    lang_map = df.set_index('강의명')['보조언어'].to_dict()
    keys_view = list(lang_map.keys())
    sorted_list = sorted(keys_view, key=len, reverse=True)
    return lang_map, sorted_list


def get_non_silent_segments_ffmpeg(audio_path):
    print("\n🔊 0. FFmpeg로 침묵 구간 분석 시작...")
    command = [
        FFMPEG_PATH, '-i', str(audio_path),
        '-af', f'silencedetect=noise={SILENCE_THRESH_DB}dB:d={MIN_SILENCE_DURATION_S}',
        '-f', 'null', '-'
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        ffmpeg_output = result.stderr
    except FileNotFoundError:
        print(f"\n[치명적 오류] ffmpeg를 찾을 수 없음. FFMPEG_PATH 확인: {FFMPEG_PATH}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"\n[오류] FFmpeg 실행 중 오류: {e.stderr}")
        return None

    silence_starts = [float(t) for t in re.findall(r'silence_start: (\d+\.?\d*)', ffmpeg_output)]
    silence_ends = [float(t) for t in re.findall(r'silence_end: (\d+\.?\d*)', ffmpeg_output)]

    if not silence_starts:
        print("   [정보] FFmpeg가 침묵 구간을 찾지 못함. 전체 파일 분석.")
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
        print(f"   [경고] 전체 길이 확인 실패: {e}")

    print(f"✅ FFmpeg 분석 완료. {len(non_silent_segments)}개의 유성 구간.")
    return non_silent_segments


def detect_music_segments(audio_path):
    print("\n🎼 0-1. inaSpeechSegmenter로 음악/음성 구간 분석...")
    try:
        raw_segments = music_segmenter(str(audio_path))
        dict_segments = []
        for label, start, end in raw_segments:
            dict_segments.append({'label': label, 'start': float(start), 'end': float(end)})
        return dict_segments
    except Exception as e:
        print(f"   [경고] 음악 구간 분석 실패: {e}")
        return []


def build_music_blocks(ina_segments, short_speech_max=1.0):
    if not ina_segments:
        return []

    blocks = []
    cur_start, cur_end = None, None

    for item in ina_segments:
        label = item['label']
        start = item['start']
        end = item['end']
        dur = end - start

        if label == "music":
            if cur_start is None:
                cur_start = start
            cur_end = end
        else:
            if cur_start is not None and label in ("speech", "noise") and dur <= short_speech_max:
                cur_end = end
            else:
                if cur_start is not None:
                    blocks.append((cur_start, cur_end))
                    cur_start, cur_end = None, None

    if cur_start is not None:
        blocks.append((cur_start, cur_end))

    if not blocks:
        return []

    blocks = sorted(blocks)
    merged = []
    for start, end in blocks:
        if not merged:
            merged.append([start, end])
        else:
            ls, le = merged[-1]
            if start <= le + 0.2:
                merged[-1][1] = max(le, end)
            else:
                merged.append([start, end])

    music_blocks = [(s, e) for s, e in merged]
    print(f"   - 병합된 음악 블록: {len(music_blocks)}개")
    return music_blocks


def remove_music_from_non_silent(non_silent_segments, music_blocks, min_len=0.05):
    if not music_blocks:
        return non_silent_segments

    if not non_silent_segments or non_silent_segments == "full_audio":
        return non_silent_segments

    cleaned = []
    for seg in non_silent_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        parts = [(seg_start, seg_end)]

        for m_start, m_end in music_blocks:
            new_parts = []
            for p_start, p_end in parts:
                if p_end <= m_start or p_start >= m_end:
                    new_parts.append((p_start, p_end))
                    continue
                if p_start < m_start:
                    new_parts.append((p_start, m_start))
                if p_end > m_end:
                    new_parts.append((m_end, p_end))
            parts = new_parts
            if not parts:
                break

        for p_start, p_end in parts:
            if p_end - p_start >= min_len:
                cleaned.append({"start": p_start, "end": p_end})

    print(f"   - 음악 제거 전 유성: {len(non_silent_segments)}개 → 제거 후: {len(cleaned)}개")
    return cleaned


def extract_segments_2stage(waveform, sample_rate, non_silent_segments):
    print("\n🚀 1. 2단계 VAD 기반 세그먼트 추출...")
    final_vad_annotation = Annotation()

    if non_silent_segments == "full_audio":
        print("   - 전체 오디오 VAD 실행")
        return vad_pipeline({"waveform": waveform, "sample_rate": sample_rate})

    total_speech_chunks_found = 0
    skipped_chunks = 0
    MIN_CHUNK_SAMPLES = int(sample_rate * 0.06)

    for segment in non_silent_segments:
        start, end = segment['start'], segment['end']
        start_frame, end_frame = int(start * sample_rate), int(end * sample_rate)
        if end_frame > waveform.shape[1]:
            end_frame = waveform.shape[1]
        chunk_waveform = waveform[:, start_frame:end_frame]

        if chunk_waveform.shape[1] < MIN_CHUNK_SAMPLES:
            skipped_chunks += 1
            continue

        file_chunk = {"waveform": chunk_waveform, "sample_rate": sample_rate}
        try:
            vad_result_chunk = vad_pipeline(file_chunk)
            for speech_turn, _, _ in vad_result_chunk.itertracks(yield_label=True):
                offset_speech_turn = Segment(speech_turn.start + start, speech_turn.end + start)
                final_vad_annotation[offset_speech_turn] = "speech"
                total_speech_chunks_found += 1
        except Exception as e:
            print(f"   [경고] VAD 처리 스킵 ({start:.2f}~{end:.2f}s): {e}")
            continue

    merged_annotation = Annotation()
    for segment in final_vad_annotation.support().itersegments():
        merged_annotation[segment] = "speech"

    print(f"✅ VAD 완료. 음성 조각 {total_speech_chunks_found}개 (너무 짧아 생략 {skipped_chunks}개)")
    return merged_annotation


def detect_language_for_vad_segments(vad_annotation, waveform, sample_rate, lang_id_model):
    print("\n🚀 VAD 구간별 언어 감지 시작 (0.1초 미만 사전 제거)...")
    label_encoder = lang_id_model.hparams.label_encoder

    segments_with_lang = []
    skipped_short_count = 0

    for segment in vad_annotation.itersegments():
        duration = segment.end - segment.start
        if duration < 0.1:
            skipped_short_count += 1
            continue
        segments_with_lang.append({'start': segment.start, 'end': segment.end})

    print(f"   - ✂️ 0.1초 미만 제거: {skipped_short_count}개")

    for seg in segments_with_lang:
        start_sample = int(seg['start'] * sample_rate)
        end_sample = int(seg['end'] * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]

        if segment_waveform.shape[1] < sample_rate * 0.5:
            seg['lang'] = 'ko'
            continue

        prediction = lang_id_model.classify_batch(segment_waveform)
        top_full_label = prediction[3][0]
        top_lang_code = top_full_label.split(':')[0].strip().lower()

        if top_lang_code in ALLOWED_LANGS:
            seg['lang'] = top_lang_code
        else:
            if (len(prediction) < 1 or not isinstance(prediction[0], torch.Tensor) or prediction[0].numel() == 0):
                seg['lang'] = 'ko'
                continue

            probabilities = prediction[0]
            allowed_probs = {}
            num_langs_to_check = min(len(probabilities), len(label_encoder.ind2lab))
            for i in range(num_langs_to_check):
                if i not in label_encoder.ind2lab:
                    continue
                label_str = label_encoder.ind2lab[i]
                lang_code = label_str.split(':')[0].strip().lower()
                if lang_code in ALLOWED_LANGS:
                    if i < len(probabilities):
                        allowed_probs[lang_code] = probabilities[i].item()

            seg['lang'] = max(allowed_probs, key=allowed_probs.get) if allowed_probs else 'ko'

    print("✅ 언어 감지 완료")
    return segments_with_lang


def tag_noise_by_music_blacklist_iterative(vad_segments, ina_segments, waveform, sample_rate,
                                          verification_model, threshold=0.4, max_iterations=2):
    print(f"\n🎼 [Iterative Blacklist] 음악 제거 시작 (최대 {max_iterations}회)...")

    if not vad_segments:
        return []

    seg_list = vad_segments if isinstance(vad_segments, list) else [{'start': s.start, 'end': s.end} for s in vad_segments.itersegments()]
    total_len = waveform.shape[1]

    music_embeddings_pool = []
    for item in ina_segments:
        if item['label'] != 'music':
            continue
        start = item['start']
        end = item['end']
        curr = start
        while curr < end:
            chunk_end = min(curr + 5.0, end)
            if chunk_end - curr < 1.0:
                break
            s_sample = int(curr * sample_rate)
            e_sample = int(chunk_end * sample_rate)
            try:
                emb = verification_model.encode_batch(waveform[:, s_sample:e_sample]).flatten()
                music_embeddings_pool.append(emb)
            except:
                pass
            curr += 5.0

    if not music_embeddings_pool:
        print("   ⚠️ 초기 음악 구간 없음. 필터링 중단.")
        return seg_list

    for it in range(max_iterations):
        print(f"   🔄 Round {it+1} (표본 {len(music_embeddings_pool)}개)")
        music_centroid = torch.mean(torch.stack(music_embeddings_pool), dim=0)
        tagged = 0

        for seg in seg_list:
            if seg.get('audio_type') in ['noise_or_music', 'noise_short', 'noise_music']:
                continue

            start = seg['start']
            end = seg['end']
            duration = end - start
            if duration < 0.1:
                continue

            s_sample = int(start * sample_rate)
            e_sample = int(end * sample_rate)
            if e_sample > total_len:
                e_sample = total_len

            try:
                curr_emb = verification_model.encode_batch(waveform[:, s_sample:e_sample]).flatten()
                score = F.cosine_similarity(music_centroid, curr_emb, dim=0).item()
                if score >= threshold:
                    seg['audio_type'] = 'noise_music'
                    seg['music_sim'] = f"{score:.2f}"
                    music_embeddings_pool.append(curr_emb)
                    tagged += 1
                else:
                    if 'audio_type' not in seg:
                        seg['audio_type'] = 'speech'
            except:
                pass

        print(f"     👉 Round {it+1}: 추가 음악 {tagged}개")
        if tagged == 0:
            print("     ✅ 더 이상 새 음악 없음. 종료.")
            break

    total_music = sum(1 for s in seg_list if s.get('audio_type') == 'noise_music')
    print(f"✅ 음악 분류 완료. 총 음악 {total_music}개")
    return seg_list


def apply_sandwich_smoothing(segments, max_duration=1.0):
    print(f"\n🥪 샌드위치 규칙 적용 (기준 {max_duration}s 이하)...")
    if len(segments) < 3:
        return segments

    changed = 0
    for i in range(1, len(segments) - 1):
        prev_seg = segments[i - 1]
        curr_seg = segments[i]
        next_seg = segments[i + 1]

        dur = curr_seg['end'] - curr_seg['start']
        if dur > max_duration:
            continue

        prev_type = prev_seg.get('audio_type', 'speech')
        curr_type = curr_seg.get('audio_type', 'speech')
        next_type = next_seg.get('audio_type', 'speech')

        if curr_type == 'speech' and prev_type == 'noise_music' and next_type == 'noise_music':
            curr_seg['audio_type'] = 'noise_music'
            curr_seg['change_log'] = 'Sandwich Correction (Speech->Music)'
            changed += 1
        elif curr_type == 'noise_music' and prev_type == 'speech' and next_type == 'speech':
            curr_seg['audio_type'] = 'speech'
            curr_seg['change_log'] = 'Sandwich Correction (Music->Speech)'
            changed += 1

    print(f"✅ 샌드위치 보정 완료. 수정 {changed}개")
    return segments


def select_sub_language(audio_file, lang_map, sorted_list, segment_with_lang):
    filename = re.sub(r'\s+|_', "", Path(audio_file).stem)
    filename = re.sub(r'0(\d)주차', r'\1주차', filename)

    sub_lang = None
    for prefix in sorted_list:
        if filename.startswith(prefix):
            sub_lang = lang_map[prefix]
            print(f'보조언어를 {sub_lang}으로 설정.')
            break

    if sub_lang is None:
        print("엑셀에 강의명 없음 → 주언어 다음으로 많이 등장한 언어로 보조언어 자동 설정.")
        lang_list = [seg['lang'] for seg in segment_with_lang if seg['lang'] not in [MAIN_LANGUAGE, 'unknown']]
        if lang_list:
            sub_lang = Counter(lang_list).most_common(1)[0][0]
            print(f'보조언어를 {sub_lang}으로 설정.')

    return sub_lang


def define_third_language(segment_with_lang, target_languages):
    print('\n제3언어 설정을 위해 VAD 분석...')
    set_to_remove = set(['unknown']) | set(target_languages)
    allowed_langs = set(ALLOWED_LANGS) - set_to_remove

    lang_durations = {}
    for segment in segment_with_lang:
        if segment['lang'] in allowed_langs:
            lang = segment['lang']
            dur = segment['end'] - segment['start']
            lang_durations[lang] = lang_durations.get(lang, 0.0) + dur

    if lang_durations:
        third_lang = max(lang_durations, key=lang_durations.get)
        if lang_durations[third_lang] < 60:
            third_lang = None
            print('제3언어 후보가 60초 미만 → 제3언어 없음')
        print(f'제3언어: {third_lang}')
    else:
        third_lang = None
        print('제3언어 없음')

    return third_lang


def convert_to_unknown(third_lang, segment_with_lang, target_languages):
    set_to_remove = set(['unknown']) | set(target_languages)

    if third_lang is None or third_lang == 'no_sub':
        for segment in segment_with_lang:
            if segment['lang'] not in set_to_remove:
                segment['lang'] = 'unknown'
    else:
        set_to_remove = set_to_remove | {third_lang}
        for segment in segment_with_lang:
            if segment['lang'] not in set_to_remove:
                segment['lang'] = 'unknown'

    return segment_with_lang


def merge_unknown(segment_with_lang):
    print('unknown VAD를 바로 직전 VAD에 흡수...')
    
    if not segment_with_lang:
        return []
    
    if segment_with_lang[0]['lang'] == 'unknown':
        segment_with_lang[0]['lang'] = 'ko'
    
    merged = []
    merged.append(segment_with_lang[0])
    for seg in segment_with_lang[1:]:
        if seg['lang'] == 'unknown':
            merged[-1]['end'] = seg['end']
        else:
            merged.append(seg)
    return merged


def duration_up_and_down(segment, MAX_DURATION):
    duration = segment['end'] - segment['start']
    return "down" if duration < MAX_DURATION else "up"


def gap_up_and_down(previous, segment, MAX_GAP):
    gap = segment['start'] - previous['end']
    return "down" if gap < MAX_GAP else "up"


def merge_vad(merge_list, first, case, MAX_DURATION, MAX_MERGED_DURATION):
    work_list = []
    temp_list = []
    final_list = []

    if len(merge_list) != 1:
        for i, seg in enumerate(merge_list):
            work_list.append(seg)
            merged_duration = seg['end'] - work_list[0]['start']

            if merged_duration > MAX_MERGED_DURATION:
                if i == len(merge_list) - 2 and duration_up_and_down(merge_list[i+1], MAX_DURATION) == 'down':
                    chunk_1 = {'start': work_list[0]['start'], 'end': merge_list[i+1]['end'], 'lang': case['chunk_1_lang']}
                    temp_list.append(chunk_1)
                    break

                elif len(work_list) == 1:
                    final_list.append(seg)
                    work_list = []

                elif i == len(merge_list) - 1:
                    if duration_up_and_down(merge_list[i], MAX_DURATION) == 'down':
                        chunk_1 = {'start': work_list[0]['start'], 'end': work_list[-1]['end'], 'lang': case['chunk_1_lang']}
                        temp_list.append(chunk_1)
                    else:
                        chunk_1 = {'start': work_list[0]['start'], 'end': work_list[-2]['end'], 'lang': case['chunk_1_lang']}
                        final_list.append(chunk_1)
                        temp_list.append(seg)

                else:
                    chunk_1 = {'start': work_list[0]['start'], 'end': work_list[-2]['end'], 'lang': case['chunk_1_lang']}
                    final_list.append(chunk_1)
                    work_list = [seg]

            else:
                if i == len(merge_list) - 1:
                    chunk_1 = {'start': work_list[0]['start'], 'end': work_list[-1]['end'], 'lang': case['chunk_1_lang']}
                    temp_list.append(chunk_1)
                else:
                    continue

    else:
        final_list = [{'start': merge_list[0]['start'], 'end': merge_list[0]['end'], 'lang': case['chunk_1_lang']}]
        temp_list = [{'start': first['start'], 'end': first['end'], 'lang': case['chunk_2_lang']}]

    return final_list, temp_list


# ============================================================
# 4) final_merge_VAD_by_lang (✅ 네 코드 그대로. 건드리면 final_segment_3 의미 깨짐)
# ============================================================
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
    print("\n🔍 [Re-detection] 병합된 구간 언어 재감지 (타겟 언어 한정)...")
    label_encoder = lang_id_model.hparams.label_encoder
    changed_count = 0

    target_langs = {'ko'}
    if sub_lang:
        target_langs.add(sub_lang)
    if third_lang is not None:
        target_langs.add(third_lang)

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    if DEVICE == "cuda":
        waveform = waveform.to(DEVICE)

    for seg in merged_segments:
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
        except:
            continue

        top_full_label = prediction[3][0]
        top_lang_code = top_full_label.split(':')[0].strip().lower()

        final_new_lang = old_lang
        if top_lang_code in target_langs:
            final_new_lang = top_lang_code
        else:
            probabilities = prediction[0].squeeze()
            allowed_probs = {}
            num_check = min(len(probabilities), len(label_encoder.ind2lab))
            for idx in range(num_check):
                if idx not in label_encoder.ind2lab:
                    continue
                label_str = label_encoder.ind2lab[idx]
                lang_code = label_str.split(':')[0].strip().lower()
                if lang_code in target_langs:
                    allowed_probs[lang_code] = probabilities[idx].item()

            if allowed_probs:
                final_new_lang = max(allowed_probs, key=allowed_probs.get)

        if final_new_lang != old_lang:
            seg['lang'] = final_new_lang
            seg['change_log'] = seg.get('change_log', '') + f" | Re-detected ({old_lang}->{final_new_lang})"
            changed_count += 1

    print(f"✅ 재감지 완료. 수정 {changed_count}개")
    return merged_segments


# ============================================================
# 5) OpenAI diarize STT + 35자 1줄 SRT 생성 (✅ 새로 추가된 핵심)
# ============================================================

@dataclass
class Cue:
    start: float
    end: float
    text: str

def srt_ts(t: float) -> str:
    """초(float)를 SRT 시간 포맷(00:00:00,000)으로 변환"""
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000.0))
    hh = ms // 3_600_000
    ms -= hh * 3_600_000
    mm = ms // 60_000
    ms -= mm * 60_000
    ss = ms // 1000
    ms -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def normalize_one_line(text: str) -> str:
    """텍스트 정규화: 줄바꿈 제거, 공백 정리, 구두점 앞 공백 제거"""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 구두점/닫는기호 앞 공백 제거: "word ." -> "word."
    text = re.sub(rf"\s+([{re.escape(PUNCT_ATTACH_TO_PREV)}])", r"\1", text)
    text = re.sub(rf"\s+([{re.escape(CLOSERS_ATTACH_TO_PREV)}])", r"\1", text)
    return text

# -----------------------------------------------------------------------------
# 3. 핵심 로직: 텍스트 균형 자르기 (보스 아이디어 적용)
# -----------------------------------------------------------------------------

def split_text_max_chars(text: str, max_chars: int) -> List[str]:
    """
    [업그레이드 버전] 
    텍스트를 단순히 앞에서부터 자르는 게 아니라, 
    전체 길이를 고려해 '균형 있게(N등분)' 나눕니다.
    """
    text = normalize_one_line(text)
    if not text:
        return []

    chunks: List[str] = []
    s = text

    # 다음 chunk가 문장부호로 시작하면 앞줄로 당기는 정규식
    leading_attach_re = re.compile(
        rf"^[{re.escape(PUNCT_ATTACH_TO_PREV + CLOSERS_ATTACH_TO_PREV)}]+"
    )
    # 자를 후보(공백, 구두점) 찾는 정규식
    split_re = re.compile(rf"[ \t]+|[{re.escape(PUNCT_ATTACH_TO_PREV)}]")

    while len(s) > max_chars:
        # 1. 앞으로 몇 줄이 필요한지 계산 (올림)
        lines_needed = math.ceil(len(s) / max_chars)
        
        # 2. 이번 줄의 '목표 길이' 설정 (균형점 찾기)
        target_len = int(len(s) / lines_needed)

        # 탐색 범위: 최대 길이(max_chars)를 넘을 순 없음
        window = s[: max_chars + 1]

        best_cut = None
        min_diff = float('inf') 

        # 자를 후보들 중 target_len에 가장 가까운 곳 선택
        for m in split_re.finditer(window):
            if m.group(0).isspace():
                cand = m.start()
            else:
                cand = m.end()  # 구두점은 포함

            diff = abs(cand - target_len)
            
            # 더 가깝거나, 거리가 같다면 최대한 뒤쪽을 선택
            if diff < min_diff:
                min_diff = diff
                best_cut = cand
            elif diff == min_diff:
                best_cut = max(best_cut if best_cut else 0, cand)

        # 자를 곳을 못 찾았거나, 너무 앞쪽이면 강제로 max_chars에서 자름
        if best_cut is None or best_cut <= 0:
            best_cut = max_chars

        part = s[:best_cut].rstrip()
        rest = s[best_cut:].lstrip()

        # 남은 뒷부분이 문장부호로 시작하면 앞줄로 당겨오기
        while rest:
            mm = leading_attach_re.match(rest)
            if not mm:
                break
            part += mm.group(0)
            rest = rest[len(mm.group(0)) :].lstrip()

        if part:
            chunks.append(part)
        s = rest
        if not s:
            break

    if s:
        chunks.append(s)

    return chunks

def openai_diarize_segments(wav_path: str, language: Optional[str] = None) -> List[dict]:
    # (실제 client 객체는 외부에 있다고 가정)
    # from your_module import client 
    
    with open(wav_path, "rb") as f:
        kwargs = dict(
            model=OPENAI_MODEL,
            file=f,
            response_format="diarized_json", # 화자 분리 포맷 요청
            chunking_strategy="auto",
        )
        if language:
            kwargs["language"] = language

        # client 호출 부분 (환경에 맞게 수정 필요)
        try:
            out = client.audio.transcriptions.create(**kwargs)
        except Exception as e:
            print(f"❌ API 호출 중 에러 발생: {e}")
            return []

    # 결과 정규화 로직 (segments 추출)
    if hasattr(out, "segments") and out.segments is not None:
        segs = list(out.segments)
    elif isinstance(out, dict) and "segments" in out:
        segs = out["segments"]
    elif hasattr(out, "model_dump"):
        d = out.model_dump()
        segs = d.get("segments", [])
    else:
        segs = []

    norm: List[dict] = []
    for s in segs:
        if isinstance(s, dict):
            norm.append(s)
            continue
        if hasattr(s, "model_dump"):
            norm.append(s.model_dump())
            continue
        norm.append({
            "start": getattr(s, "start", 0.0),
            "end": getattr(s, "end", getattr(s, "start", 0.0)),
            "speaker": getattr(s, "speaker", None),
            "text": getattr(s, "text", "") or "",
        })
    return norm

def diarize_segments_to_cues(
    diar_segs: List[dict],
    offset_sec: float,
    max_chars: int = 35,
    min_cue_dur: float = 0.25,
    include_speaker_prefix: bool = True, # ✅ True 기본값 (화자 표시)
) -> List[Cue]:
    """
    OpenAI 결과를 받아서 자막(Cue) 객체로 변환.
    - 텍스트는 균형 있게 자름
    - 시간 배분은 '화자 태그 제외' 순수 텍스트 기준 (싱크 정확도)
    - 화자 태그는 첫 덩어리에만 부착
    """
    cues: List[Cue] = []
    
    for seg in diar_segs:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        speaker = seg.get("speaker")
        text = seg.get("text", "") or ""
        text = normalize_one_line(text)

        if end <= start or not text:
            continue

        # 1. '순수 텍스트' 기준으로 균형 있게 자르기
        chunks = split_text_max_chars(text, max_chars=max_chars)
        if not chunks:
            continue

        dur = end - start
        
        # 2. 시간 배분 (화자 태그 없이 순수 글자 수로 계산 -> 싱크 정확)
        lengths = [max(1, len(c)) for c in chunks]
        total = sum(lengths)

        cum = 0
        for i, (c, ln) in enumerate(zip(chunks, lengths)):
            c0 = cum / total
            cum += ln
            c1 = cum / total

            cs = start + dur * c0
            ce = start + dur * c1
            
            if ce - cs < min_cue_dur:
                ce = min(end, cs + min_cue_dur)
            
            # 마지막 조각은 세그먼트 끝시간에 딱 맞춤
            if i == len(chunks) - 1:
                ce = end

            # 3. 자막 텍스트 완성 (화자 태그 붙이기)
            final_text = c
            
            # ✅ 화자가 바뀌는 '첫 번째 덩어리'에만 이름표 붙이기
            if include_speaker_prefix and speaker is not None and i == 0:
                final_text = f"[{speaker}] {c}"
            
            cues.append(Cue(start=cs + offset_sec, end=ce + offset_sec, text=final_text))

    # 유효성 검사
    cues = [c for c in cues if c.end > c.start and c.text.strip()]
    return cues

def cues_to_srt(cues: List[Cue]) -> str:
    lines = []
    for idx, c in enumerate(cues, 1):
        lines.append(str(idx))
        lines.append(f"{srt_ts(c.start)} --> {srt_ts(c.end)}")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines)

def extract_temp_wav_from_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    pad_sec: float,
    temp_dir: Path,
    temp_name: str,
) -> Tuple[str, float, float]:
    temp_dir.mkdir(parents=True, exist_ok=True)

    start_sec2 = max(0.0, start_sec - pad_sec)
    end_sec2 = max(start_sec2, end_sec + pad_sec)

    start_sample = int(start_sec2 * sample_rate)
    end_sample = int(end_sec2 * sample_rate)
    if end_sample > waveform.shape[1]:
        end_sample = waveform.shape[1]

    seg = waveform[:, start_sample:end_sample].squeeze().detach().cpu().numpy()
    if seg.ndim > 1:
        seg = seg.flatten()

    # 양자화 및 저장
    seg = np.clip(seg, -1.0, 1.0)
    seg_i16 = (seg * 32767).astype(np.int16)
    seg_clean = seg_i16.astype(np.float32) / 32767.0

    wav_path = str(temp_dir / temp_name)
    sf.write(wav_path, seg_clean, sample_rate)
    return wav_path, start_sec2, end_sec2

def run_openai_diarize_and_save_srt(
    waveform: torch.Tensor,
    sample_rate: int,
    audio_path: str,
    final_segments: List[dict],
    output_folder: str,
    done_path: str,
):
    print("\n🚀 OpenAI diarize STT 시작 (35자/균형분할/화자표시)...")

    base_filename = Path(audio_path).stem
    srt_filename = f"{base_filename}{FILENAME_SUFFIX}.srt"
    output_srt_path = str(Path(output_folder) / srt_filename)

    temp_dir = Path(output_folder) / "_tmp_openai_segments"
    all_cues: List[Cue] = []

    try:
        for i, seg in enumerate(final_segments, 1):
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
            seg_lang = seg.get("lang", None)

            if seg_end <= seg_start:
                continue

            seg_dur = seg_end - seg_start
            print(f"  - [{i}/{len(final_segments)}] {seg_start:.2f}s~{seg_end:.2f}s ({seg_lang}), dur={seg_dur:.2f}s")

            # 세그먼트가 너무 길면 chunking
            cursor = seg_start
            chunk_idx = 0
            while cursor < seg_end - 0.01:
                chunk_idx += 1
                chunk_start = cursor
                chunk_end = min(seg_end, cursor + OPENAI_MAX_CHUNK_SECONDS)
                cursor = chunk_end

                temp_wav_path, chunk_offset, _ = extract_temp_wav_from_waveform(
                    waveform=waveform,
                    sample_rate=sample_rate,
                    start_sec=chunk_start,
                    end_sec=chunk_end,
                    pad_sec=OPENAI_PAD_SECONDS,
                    temp_dir=temp_dir,
                    temp_name=f"{base_filename}_seg{i:04d}_chunk{chunk_idx:02d}.wav",
                )

                # API 호출 (여기서 화자 분리된 정보 획득)
                diar_segs = openai_diarize_segments(temp_wav_path, language=seg_lang)

                # 자막 변환 (✅ INCLUDE_SPEAKER_PREFIX=True 반영됨)
                cues = diarize_segments_to_cues(
                    diar_segs=diar_segs,
                    offset_sec=chunk_offset,
                    max_chars=SUBTITLE_MAX_CHARS,
                    min_cue_dur=SUBTITLE_MIN_CUE_DUR,
                    include_speaker_prefix=INCLUDE_SPEAKER_PREFIX, # 화자 표시 켜기
                )

                # 패딩 범위 밖 클리핑
                clipped = []
                for c in cues:
                    cs = max(c.start, chunk_start)
                    ce = min(c.end, chunk_end)
                    if ce > cs:
                        text = normalize_one_line(c.text) if SUBTITLE_ONE_LINE else c.text
                        clipped.append(Cue(start=cs, end=ce, text=text))
                all_cues.extend(clipped)

        # ---------------------------------------------------------------------
        # 자막 정렬 및 겹침 처리 (화자 바뀔 때 자막 끊기 로직 포함)
        # ---------------------------------------------------------------------
        all_cues.sort(key=lambda x: (x.start, x.end))
        cleaned: List[Cue] = []
        for c in all_cues:
            if not cleaned:
                cleaned.append(c)
                continue
            prev = cleaned[-1]
            
            # ✅ 겹치면(뒷사람이 말 시작하면) 앞사람 자막을 끊어버림
            if c.start < prev.end:
                prev_end_new = max(prev.start + 0.05, c.start)
                cleaned[-1] = Cue(start=prev.start, end=prev_end_new, text=prev.text)
                if c.end <= c.start:
                    continue
            cleaned.append(c)

        srt_text = cues_to_srt(cleaned)
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
        print(f"✅ SRT 저장 완료: {output_srt_path} (cue {len(cleaned)}개)")

    except Exception as e:
        print(f"❌ OpenAI diarize STT 중 오류: {e}")
        traceback.print_exc()
        return

    finally:
        # 임시 파일 정리
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

        # 원본 이동
        if done_path:
            try:
                shutil.move(audio_path, done_path)
                print(f"✅ 원본 WAV 이동 완료: {Path(done_path) / Path(audio_path).name}")
            except Exception as e:
                print(f"❌ 원본 이동 실패: {e}")
        else:
            print("✅ 작업 완료. 원본 WAV는 그대로 유지.")



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
    
    

# ============================================================
# 6) 메인 실행
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    lang_map = {}
    sorted_list = []

    print("\n📊 강의 정보 엑셀(xlsx)이 있으면 선택 (취소 시 자동 감지)")
    xlsx_file_path = filedialog.askopenfilename(title="xlsx 파일 선택 (선택 안 함: 취소)", filetypes=[("xlsx File", "*.xlsx")])

    if xlsx_file_path:
        try:
            lang_map, sorted_list = read_xlsx_and_create_dict(xlsx_file_path)
            print(f"✅ 엑셀 로드 완료: {len(lang_map)}개 강의")
        except Exception as e:
            print(f"⚠️ 엑셀 읽기 실패 → 자동 감지로 진행: {e}")
    else:
        print("⚠️ 엑셀 미선택 → 보조언어 자동 감지")

    print("\n🎵 분석할 WAV 파일 선택 (다중 선택 가능)")
    audio_files = filedialog.askopenfilenames(title="WAV 파일 선택", filetypes=[("WAV Files", "*.wav")])
    if not audio_files:
        print("❌ 파일 미선택 → 종료")
        raise SystemExit(0)

    print("\n💾 결과 SRT 저장 폴더 선택")
    output_path = filedialog.askdirectory(title="SRT 저장 폴더")
    if not output_path:
        print("❌ 저장 폴더 미선택 → 종료")
        raise SystemExit(0)

    print("\n📂 완료된 WAV 이동 폴더 선택 (취소 시 이동 안 함)")
    done_path = filedialog.askdirectory(title="완료 WAV 이동 폴더")
    if not done_path:
        done_path = ""
        print("⚠️ 완료 폴더 미선택 → 원본 유지")

    for i, audio_file in enumerate(list(audio_files), 1):
        if not os.path.exists(audio_file):
            print(f"⚠️ [{i}/{len(audio_files)}] 파일 없음: {audio_file} → 스킵")
            continue

        print(f"\n{'='*60}")
        print(f"▶️  [{i}/{len(audio_files)}] 처리 시작: {os.path.basename(audio_file)}")
        print(f"{'='*60}")

        non_silent_segments = get_non_silent_segments_ffmpeg(audio_file)
        if non_silent_segments is None:
            print("❌ FFmpeg 분석 실패 → 스킵")
            continue

        # 음악 감지 + 병합
        #ina_segments = detect_music_segments(audio_file)
        #music_blocks = build_music_blocks(ina_segments, short_speech_max=1.0)

        # full_audio 처리
        #if non_silent_segments == "full_audio":
        #    if music_blocks:
        #        try:
        #            audio_for_duration = AudioSegment.from_file(str(audio_file))
        #            total_dur = len(audio_for_duration) / 1000.0
        #            base_segments = [{"start": 0.0, "end": total_dur}]
        #            non_silent_segments = remove_music_from_non_silent(base_segments, music_blocks)
        #        except Exception as e:
        #            print(f"   [경고] 오디오 길이 실패: {e}")
        #else:
        #    non_silent_segments = remove_music_from_non_silent(non_silent_segments, music_blocks)
#
        #if not non_silent_segments:
        #    print("   [정보] 음악 제거 후 남는 구간 없음 → 스킵")
        #    continue

        # 오디오 로딩
        print("\n🎵 오디오 로딩...")
        try:
            audio_loader = Audio(sample_rate=16000, mono=True)
            waveform, sample_rate = audio_loader(audio_file)
            print("✅ 로딩 완료.")
        except Exception as e:
            print(f"[치명] 로딩 실패: {e}")
            continue

        # VAD + 언어 감지
        vad_annotation = extract_segments_2stage(waveform, sample_rate, non_silent_segments)
        segment_with_lang = detect_language_for_vad_segments(vad_annotation, waveform, sample_rate, lang_id_model)

        # 음악 제거(임베딩 기반)
        #segment_with_lang_and_music = tag_noise_by_music_blacklist_iterative(
        #    segment_with_lang,
        #    ina_segments,
        #    waveform,
        #    sample_rate,
        #    verification_model,
        #    threshold=MUSIC_SIM_THRESHOLD,
        #    max_iterations=MUSIC_MAX_ITER
        #)
        #segment_with_lang_and_music2 = apply_sandwich_smoothing(segment_with_lang_and_music, max_duration=SANDWICH_MAX_DURATION)
#
        #print(f"🧹 음악 제거 전: {len(segment_with_lang)}개")
        #segment_with_lang_2 = [seg for seg in segment_with_lang_and_music2 if seg.get('audio_type') != 'noise_music']
        #print(f"🧹 음악 제거 후: {len(segment_with_lang)}개")

        if not segment_with_lang:
            print("❌ 유효 구간 없음 → 스킵/이동")
            if done_path:
                try:
                    shutil.move(audio_file, done_path)
                    print(f"✅ 이동 완료: {Path(done_path) / Path(audio_file).name}")
                except Exception as e:
                    print(f"❌ 이동 실패: {e}")
            continue

        # 보조언어 결정
        target_languages = [MAIN_LANGUAGE]
        if SUB_LANGUAGE is None:
            sub_language = select_sub_language(audio_file, lang_map, sorted_list, segment_with_lang)
        elif SUB_LANGUAGE == "no_sub":
            sub_language = None
        else:
            sub_language = SUB_LANGUAGE

        if sub_language is not None:
            target_languages.append(sub_language)

        # diarize 모델은 prompt를 안 받지만, 기존 구조 유지용
        instructor_prompt = INSTRUCTOR_PROMPT_DICT.get(sub_language)

        # 제3언어
        if THIRD_LANG == 'no_third':
            third_lang = None
        else:
            third_lang = define_third_language(segment_with_lang, target_languages)

        # unknown 처리
        segment_with_lang_still_unknown_in = convert_to_unknown(third_lang, segment_with_lang, target_languages)
        segment_with_lang_without_unknown = merge_unknown(segment_with_lang_still_unknown_in)

        # ✅ 여기부터가 네가 말한 "final segment 3" 만드는 구간 (그대로 유지)
        final_segment_2 = final_merge_VAD_by_lang(
            segment_with_lang_without_unknown,
            sub_language,
            third_lang,
            MAX_DURATION,
            MAX_GAP,
            MAX_MERGED_DURATION
        )
        final_segment_3 = redetect_language_for_merged_segments(
            final_segment_2,
            waveform,
            sample_rate,
            lang_id_model,
            sub_language,
            third_lang
        )
        
        
        # VAD 관찰용 코드
        # vad_annotation_to_srt_empty_with_lang(final_segment_3, output_path, audio_file, file_suffix="vad_view")
        final_segment_4 = final_merge_VAD_by_lang(final_segment_3, sub_language, third_lang, MAX_DURATION, MAX_GAP, MAX_MERGED_DURATION)
        # vad_annotation_to_srt_empty_with_lang(segment_with_lang, output_path, audio_file, file_suffix="segment_with_lang(2)")

        # ✅ 여기서부터 STT만 OpenAI diarize로 수행 (all_cues 루프 포함)
        run_openai_diarize_and_save_srt(waveform=waveform, 
                                        sample_rate=sample_rate, 
                                        audio_path=audio_file, 
                                        final_segments=final_segment_4, 
                                        output_folder=output_path, 
                                        done_path=done_path
                                        )

    print("\n\n🎉 모든 파일 처리가 완료되었습니다.")
