import srt
from datetime import timedelta
import tkinter as tk
from tkinter import filedialog
import shutil
from pathlib import Path
import torch
import torchaudio
import torch.nn.functional as F
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


# ========== 사용자 설정 파라미터 ==========
HF_TOKEN = "" # 허깅페이스 토큰
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FFMPEG_PATH = r"C:\Users\cufs\Desktop\업무\subtitle\ffmpeg\ffmpeg.exe" # 📌 FFmpeg 실행 파일의 '전체 경로'를 정확하게 입력하세요.
STT_MODEL_SIZE = "large-v3" # STT 모델 크기 ('tiny', 'base', 'small', 'medium', 'large-v3')
FILENAME_SUFFIX = "" # 최종 파일명에 추가될 접미사
MAIN_LANGUAGE = "ko"
SUB_LANGUAGE = None # 보조언어를 강제로 고정. 기본값은 None


# 1. FFmpeg 볼륨 전처리 파라미터
SILENCE_THRESH_DB = -40  # FFmpeg이 '침묵'으로 판단할 소리의 크기 기준입니다. -40dB보다 작은 소리는 침묵으로 간주합니다.
MIN_SILENCE_DURATION_S = 0.3  # 최소 0.3초 이상 지속되는 침묵 구간만 찾아내도록 설정합니다.


# 2. VAD 모델 세부 파라미터
VAD_PARAMS = {
    "min_duration_off": 0.05,  # 음성이 없는 구간(침묵)이 최소 0.05초는 되어야 침묵으로 인정합니다.
    "min_duration_on": 0.01,  # 음성이 있는 구간이 최소 0.01초는 되어야 음성으로 인정합니다.
    "onset": 0.4,  # 음성 시작이라고 판단할 확률의 임계값입니다. (0~1 사이, 높을수록 보수적)
    "offset": 0.6  # 음성 종료라고 판단할 확률의 임계값입니다. (0~1 사이, 높을수록 보수적)
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
# ja = "오늘은 て形랑 辞書形를 볼 거야. て形는 연결·부탁(〜てください), 辞書形는 기본형. ５番出口에서 만나. 엘리베이터는 エレベーター, 계단은 階段."
ja = "오늘은 테형랑 지쇼형를 볼 거야. 테형는 연결·부탁(~테 쿠다사이), 지쇼형는 기본형. 고반 데구치에서 만나. 엘리베이터는 에레베-타-, 계단은 카이단."
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


# ========== 핵심 기능 함수 ==========

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


def extract_segments_2stage(waveform, sample_rate, non_silent_segments):
    print("\n🚀 1. 2단계 VAD 기반 세그먼트 추출 시작...")
    final_vad_annotation = Annotation()
    
    if non_silent_segments == "full_audio":
        print("   - 전체 오디오에 대해 VAD를 실행합니다.")
        return vad_pipeline({"waveform": waveform, "sample_rate": sample_rate})

    total_speech_chunks_found = 0
    for i, segment in enumerate(non_silent_segments):
        start, end = segment['start'], segment['end']
        start_frame, end_frame = int(start * sample_rate), int(end * sample_rate)
        chunk_waveform = waveform[:, start_frame:end_frame]
        file_chunk = {"waveform": chunk_waveform, "sample_rate": sample_rate}
        vad_result_chunk = vad_pipeline(file_chunk)
        
        for speech_turn, _, _ in vad_result_chunk.itertracks(yield_label=True):
            offset_speech_turn = Segment(speech_turn.start + start, speech_turn.end + start)
            final_vad_annotation[offset_speech_turn] = "speech"
            total_speech_chunks_found += 1
            
    print(f"✅ 2단계 VAD 분석 완료. 총 {total_speech_chunks_found}개의 음성 조각 발견.")
    return final_vad_annotation


def detect_language_for_vad_segments(vad_annotation, waveform, sample_rate, lang_id_model):
    """
    pyannote VAD Annotation 결과와 이미 로드된 waveform을 사용해 각 세그먼트의 언어를 감지합니다.
    """
    print("\n🚀 VAD 구간별 언어 감지 시작...")
    
    allowed_langs = {'ko', 'en', 'vi', 'es', 'zh', 'ja', 'id', 'unknown'}
    label_encoder = lang_id_model.hparams.label_encoder
    
    # 1. Annotation 객체를 처리하기 쉬운 딕셔너리 리스트로 변환합니다.
    segments_with_lang = []
    for segment in vad_annotation.itersegments():
        segments_with_lang.append({'start': segment.start, 'end': segment.end})

    # 2. 각 세그먼트를 순회하며 언어를 감지합니다.
    for seg in segments_with_lang:
        # 오디오 파일을 다시 읽는 대신, 메모리에 있는 waveform에서 바로 잘라냅니다.
        start_sample = int(seg['start'] * sample_rate)
        end_sample = int(seg['end'] * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]

        # 세그먼트가 너무 짧으면(0.5초 미만) 'unknown'으로 처리합니다.
        if segment_waveform.shape[1] < sample_rate * 0.5:
            seg['lang'] = 'unknown'
            continue
        
        # 잘라낸 오디오 조각으로 언어를 예측합니다.
        prediction = lang_id_model.classify_batch(segment_waveform)
        
        # 1. 일단 Top 1 언어를 확인합니다.
        top_full_label = prediction[3][0]
        top_lang_code = top_full_label.split(':')[0].strip().lower()

        if top_lang_code in allowed_langs:
            # 2. Top 1이 허용 목록에 있으면, 그대로 사용 (가장 빠름)
            seg['lang'] = top_lang_code
        else:
            # 3. Top 1이 허용 목록에 없으면, 전체 확률을 뒤져봅니다.
            print(f"    - [언어 재조정] Top 1 '{top_lang_code}'(이)가 허용 목록에 없음. '{list(allowed_langs)}' 내에서 재검색...")

            if (len(prediction) < 1 or
                    not isinstance(prediction[0], torch.Tensor) or
                    prediction[0].numel() == 0):
                print(f"    - [경고] 확률 텐서 없음/비었음 ({seg['start']:.2f}s~{seg['end']:.2f}s). 'unknown' 처리.")
                seg['lang'] = 'unknown'
                continue

            probabilities = prediction[0]

            allowed_probs = {}
            num_langs_to_check = min(len(probabilities), len(label_encoder.ind2lab))
            for i in range(num_langs_to_check):
                if i not in label_encoder.ind2lab: continue
                label_str = label_encoder.ind2lab[i]
                lang_code = label_str.split(':')[0].strip().lower()

                if lang_code in allowed_langs:
                    if i < len(probabilities):
                         prob = probabilities[i].item()
                         allowed_probs[lang_code] = prob

            if allowed_probs:
                final_lang = max(allowed_probs, key=allowed_probs.get)
                seg['lang'] = final_lang
            else:
                seg['lang'] = 'unknown'

    print("✅ 언어 감지 완료")
    return segments_with_lang


def merge_segments_by_language(segments, target_languages, max_gap=2.5, short_threshold=2.5):
    """모든 세그먼트를 순회하며 지능적으로 병합합니다."""
    print("\n🚀 3. 언어별 세그먼트 병합 시작...")
    if not segments:
        return []

    processed_segments = segments.copy()
    
    #allowed_langs = {'ko', 'en', 'vi', 'es', 'zh', 'ja', 'id', 'unknown'}
    #for seg in processed_segments:
    #    if seg.get('lang') not in allowed_langs:
    #        print(f"  - [언어 코드 정리] {seg['start']:.2f}s 구간의 언어 '{seg['lang']}'를 'ko'로 변경합니다.")
    #        seg['lang'] = 'ko'
            
    if len(processed_segments) >= 3:
        for i in range(1, len(processed_segments) - 1):
            prev_seg, current_seg, next_seg = processed_segments[i-1], processed_segments[i], processed_segments[i+1]
            
            is_sandwiched = (prev_seg['lang'] == next_seg['lang']) and (current_seg['lang'] != prev_seg['lang'])
            is_short = (current_seg['end'] - current_seg['start']) <= short_threshold
            is_target_sandwich = prev_seg['lang'] in target_languages

            if is_sandwiched and is_short and is_target_sandwich:
                print(f"  - [병합 사전 처리] {current_seg['start']:.2f}s 구간({current_seg['lang']})을 앞뒤 언어({prev_seg['lang']})와 동일하게 처리합니다.")
                current_seg['lang'] = prev_seg['lang']
    
    merged = []
    if not processed_segments:
        print("✅ 병합할 대상 언어 구간이 없습니다.")
        return []
        
    current_seg = processed_segments[0].copy()

    for next_seg in processed_segments[1:]:
        gap = next_seg['start'] - current_seg['end']
        
        is_same_language = (next_seg['lang'] == current_seg['lang'])
        is_close_enough = gap <= max_gap
        is_current_target = current_seg['lang'] in target_languages
        is_next_absorbable = (next_seg['lang'] not in target_languages)

        if (is_same_language and is_close_enough) or (is_current_target and is_next_absorbable and is_close_enough): # 1. 앞뒤 언어가 같으면서 두 세그먼트의 갭이 2.5초 미만이거나 2. 현재 세그먼트가 타겟언어이면서 다음 세그먼트가 타겟언어가 아니고 갭이 2.5초 미만일 경우
            if is_current_target and is_next_absorbable: # 2의 경우라면 병합
                print(f"  - [병합] {next_seg['start']:.2f}s 구간({next_seg['lang']})을 이전 구간({current_seg['lang']})에 흡수합니다.")
            current_seg['end'] = next_seg['end']
        else: # 1의 경우라면 현재 세그먼트를 merged 리스트에 추가
            merged.append(current_seg)
            current_seg = next_seg.copy()

    merged.append(current_seg) # 맨 마지막 세그먼트도 merged 리스트에 추가

    # final_merged = [seg for seg in merged if seg['lang'] in target_languages]

    print(f"✅ 병합 완료: 총 {len(merged)}개 구간")
    for i, seg in enumerate(merged, 1):
        print(f"  - [{i:03}] {seg['start']:.2f}s ~ {seg['end']:.2f}s (언어: {seg['lang']})")
    return merged


def merge_subtitle_objects(subs):
    """단어 단위 자막(srt.Subtitle 객체 리스트)을 2단계에 걸쳐 병합합니다."""
    if not subs:
        return []

    # --- 1차 병합: 문장 및 길이 기반 ---
    print("  - [자막 병합] 1차 병합: 문장 및 길이 규칙에 따라 병합 중...")
    pass1_subs = []
    current_sub = subs[0]

    for next_sub in subs[1:]:
        gap = (next_sub.start - current_sub.end).total_seconds()
        current_ends_sentence = current_sub.content.endswith('.') or current_sub.content.endswith('?')
        combined_text = current_sub.content + " " + next_sub.content
        
        should_merge = (
            gap <= MERGE_THRESHOLD_SECONDS and
            not current_ends_sentence and
            len(combined_text) <= MAX_CHARS_PER_LINE
        )

        if should_merge:
            current_sub.end = next_sub.end
            current_sub.content = combined_text
        else:
            pass1_subs.append(current_sub)
            current_sub = next_sub
            
    pass1_subs.append(current_sub)

    # --- 2차 병합: 지나치게 짧은 세그먼트 강제 병합 ---
    print("  - [자막 병합] 2차 병합: 짧은 세그먼트 정리 중...")
    if len(pass1_subs) < 2:
        return pass1_subs

    final_subs = [pass1_subs[0]]
    
    for i in range(1, len(pass1_subs)):
        current_sub_to_check = pass1_subs[i]
        duration = (current_sub_to_check.end - current_sub_to_check.start).total_seconds()
        
        previous_sub = final_subs[-1]
        
        if duration < MIN_DURATION_SECONDS:
            new_text = previous_sub.content + " " + current_sub_to_check.content
            
            if len(new_text) <= MAX_CHARS_PER_LINE * 1.5:
                previous_sub.end = current_sub_to_check.end
                previous_sub.content = new_text
                print(f"    - 짧은 자막 병합: \"{current_sub_to_check.content}\"")
            else:
                final_subs.append(current_sub_to_check)
        else:
            final_subs.append(current_sub_to_check)
            
    return final_subs


def run_stt_and_save_srt(waveform, sample_rate, audio_path, segments, output_folder, instructor_prompt, done_path):
    """STT 수행 후, 단어 자막을 생성하고 이를 다시 문장으로 병합하여 최종 SRT를 저장합니다."""
    print("\n🚀 4. STT 및 자막 병합 시작...")
    if stt_model is None:
        print("❌ Whisper 모델이 로드되지 않아 STT를 진행할 수 없습니다.")
        return

    try:
        all_word_subs = []
        audio_waveform, sr = waveform, sample_rate

        for i, seg in enumerate(segments, 1):
            start_time, end_time = seg['start'], seg['end']
            lang = seg['lang'] if seg['lang'] != 'unknown' else None

            print(f"  - [{i}/{len(segments)}] STT 구간 처리 중: {start_time:.2f}s ~ {end_time:.2f}s (언어: {lang or '자동 감지'})")

            start_sample, end_sample = int(start_time * sr), int(end_time * sr)
            segment_audio = audio_waveform[:, start_sample:end_sample][0]

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
            
            for segment in result.segments:
                for word in segment.words:
                    start_ts = timedelta(seconds=word.start) + offset
                    end_ts = timedelta(seconds=word.end) + offset
                    content = word.word.strip()
                    if content:
                        all_word_subs.append(srt.Subtitle(index=0, start=start_ts, end=end_ts, content=content))
        
        if not all_word_subs:
            print("  - [경고] STT 결과 텍스트가 없어 SRT 파일을 생성하지 않습니다.")
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
        if done_path:
            print("  - 작업 완료 후 파일 이동을 시도합니다.")
            try:
                shutil.move(audio_path, done_path)
                print(f"✅ 원본 WAV 파일 이동 완료: {Path(done_path) / Path(audio_path).name}")
            except shutil.Error as move_e:
                print(f"❌ 파일 이동 중 오류 발생: {move_e}")
            except Exception as e:
                 print(f"❌ 파일 이동 중 예기치 않은 오류 발생: {e}")
        else:
            print("✅ 작업 완료. 파일은 원본 위치에 유지됩니다.")



# ========== 메인 실행 로직 ==========
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

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
                    segment_with_lang = detect_language_for_vad_segments(vad_annotation, waveform, sample_rate, lang_id_model)

                    sub_language = None if (SUB_LANGUAGE==None) else SUB_LANGUAGE
                    instructor_prompt = None
                    target_languages = [MAIN_LANGUAGE]
                    
                    lang_list = [seg['lang'] for seg in segment_with_lang if seg['lang'] not in [MAIN_LANGUAGE, 'unknown']]
                    
                    if sub_language == None:
                        if lang_list:
                            lang_counts = Counter(lang_list)
                            for lang, count in lang_counts.most_common():
                                if lang in INSTRUCTOR_PROMPT_DICT:
                                    sub_language = lang
                                    instructor_prompt = INSTRUCTOR_PROMPT_DICT.get(sub_language)
                                    target_languages.append(sub_language)
                                    break
                        if sub_language:
                            print(f"\n✅ 보조 언어 설정: {sub_language.upper()}")
                        else:
                            print(f"\nℹ️ 처리 가능한 보조 언어가 감지되지 않았습니다. 주 언어({MAIN_LANGUAGE.upper()})만 처리합니다.")
                    else:
                        target_languages.append(sub_language)
                        instructor_prompt = INSTRUCTOR_PROMPT_DICT.get(sub_language)
                        print(f"\n✅ 보조 언어 강제 설정: {sub_language.upper()}")
                    
                    merged_segments = merge_segments_by_language(segment_with_lang, target_languages)

                    if merged_segments:
                        run_stt_and_save_srt(waveform, sample_rate, audio_file, merged_segments, output_path, instructor_prompt, done_path)
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