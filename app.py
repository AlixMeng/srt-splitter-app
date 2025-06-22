import os
import re
import uuid
import zipfile
import io
import shutil
import sys
import logging
from datetime import timedelta, datetime
import json
import string

# 配置日志级别为 DEBUG，以便看到详细信息
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 强制标准输出不带缓冲，帮助调试
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from flask import Flask, request, render_template, send_from_directory, url_for, redirect, flash, jsonify, Response, send_file
from werkzeug.utils import secure_filename

# 初始化 Flask 应用
app = Flask(__name__)
# 配置上传文件的文件夹
app.config['UPLOAD_FOLDER'] = 'uploads'
# 配置处理后文件的文件夹
app.config['PROCESSED_FOLDER'] = 'processed'
# 限制文件大小为16MB (16 * 1024 * 1024 字节)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# 设置 Flask 应用的 secret key，用于会话管理和安全。生产环境中建议从环境变量加载。
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# --- 文件夹初始化和清理函数 ---
def setup_and_clean_folders():
    """
    在应用启动时设置并清理上传和处理文件所需的文件夹。
    对于 'uploads' 文件夹，如果存在则删除其内容并重新创建，确保每次启动都是干净的环境。
    对于 'processed' 文件夹，只确保其存在，不对其内容进行清理，因为它通常用于持久化数据。
    改进了错误处理，以应对 'Device or resource busy' 错误。
    """
    app.logger.info("Initializing and cleaning folders...")

    # 清理 uploads 文件夹（用于临时文件），确保每次启动时为空
    uploads_folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(uploads_folder):
        app.logger.debug(f"Deleting existing folder: {uploads_folder}")
        try:
            shutil.rmtree(uploads_folder) # 尝试删除整个目录
            app.logger.info(f"Successfully deleted: {uploads_folder}")
        except OSError as e:
            # 如果删除失败（例如，目录被占用），则尝试清空目录内容
            app.logger.warning(f"Warning: Could not remove '{uploads_folder}': {e}. Attempting to empty instead.", exc_info=True)
            for filename in os.listdir(uploads_folder):
                file_path = os.path.join(uploads_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path) # 删除文件或链接
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path) # 递归删除子目录
                except Exception as inner_e:
                    app.logger.error(f"Failed to remove {file_path} during cleanup: {inner_e}", exc_info=True)
    os.makedirs(uploads_folder, exist_ok=True) # 重新创建或确保存在
    app.logger.info(f"Initialized folder: {uploads_folder}")

    # 只确保 processed 文件夹存在，不删除其内容
    processed_folder = app.config['PROCESSED_FOLDER']
    os.makedirs(processed_folder, exist_ok=True) # 确保存在
    app.logger.info(f"Initialized folder: {processed_folder} (contents not cleared for persistence).")

setup_and_clean_folders()

# --- HanLP 模型加载部分 ---
HanLP_tokenizer = None
HANLP_CONTAINER_ROOT = '/app/hanlp_data'
MODEL_SUBPATH = '.hanlp/model/tok/open_tok_pos_ner_srl_dep_sdp_con_electra_base_20201223_201906'
MODEL_FULL_PATH = os.path.join(HANLP_CONTAINER_ROOT, MODEL_SUBPATH)
os.environ['HANLP_HOME'] = HANLP_CONTAINER_ROOT

def load_hanlp_model():
    global HanLP_tokenizer
    app.logger.info(f"设置 HANLP_HOME 环境变量为: {os.environ.get('HANLP_HOME')}")
    app.logger.info(f"HanLP 模型期望加载路径: {MODEL_FULL_PATH}")

    config_json_path = os.path.join(MODEL_FULL_PATH, 'config.json')
    if not os.path.exists(config_json_path):
        app.logger.error(f"HanLP 模型文件 '{config_json_path}' 不存在。请确保模型已正确放置在 Docker 镜像的 '{MODEL_FULL_PATH}' 路径下。")
        HanLP_tokenizer = None
        return

    try:
        import hanlp
        app.logger.info(f"尝试从绝对路径加载 HanLP 模型: {MODEL_FULL_PATH}")
        HanLP_tokenizer = hanlp.load(MODEL_FULL_PATH)
        app.logger.info("HanLP 模型加载成功！")
    except Exception as e:
        app.logger.error(f"HanLP 模型加载失败: {e}", exc_info=True)
        app.logger.error(f"请检查以下路径是否正确且模型文件存在于容器内部：{MODEL_FULL_PATH}")
        HanLP_tokenizer = None

load_hanlp_model()

# --- 自定义 JSON 编码器，用于处理 timedelta 对象 ---
class TimedeltaEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, timedelta):
            # 将 timedelta 对象转换为其字符串表示形式，例如 "0:00:01.234000"
            return str(obj)
        # 让基类处理其他类型
        return json.JSONEncoder.default(self, obj)

# --- 辅助函数：将调试信息写入文件 ---
def write_debug_output(stage_name, data, original_filename_prefix):
    debug_dir = app.config['PROCESSED_FOLDER']
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    debug_filename = f"{original_filename_prefix}_{stage_name}_{timestamp}.log"
    debug_filepath = os.path.join(debug_dir, debug_filename)
    
    with open(debug_filepath, 'w', encoding='utf-8') as f:
        if isinstance(data, list) and all(isinstance(item, dict) for item in data): # 针对字典列表 (例如 merged_logic_blocks)
            try:
                # 尝试使用自定义编码器进行 JSON 序列化
                json_output = json.dumps(data, ensure_ascii=False, indent=2, cls=TimedeltaEncoder)
                f.write(json_output)
            except TypeError as e:
                # 如果序列化失败，记录错误并写入一个可读的字符串表示
                app.logger.error(f"在阶段 '{stage_name}' JSON 序列化时发生 TypeError: {e}", exc_info=True)
                # 写入一个简单的表示，确保文件内容不会导致后续问题
                f.write(f"错误: 无法将数据序列化为 JSON。原始错误: {e}\n")
                f.write(repr(data)) # 写入原始数据的字符串表示
            except Exception as e:
                # 捕获其他可能的异常
                app.logger.error(f"在阶段 '{stage_name}' JSON 序列化时发生意外错误: {e}", exc_info=True)
                f.write(f"错误: 在 JSON 序列化时发生意外错误。原始错误: {e}\n")
                f.write(repr(data))
        else: # 针对字符串列表或单个字符串
            if isinstance(data, list):
                f.write("\n---\n".join(str(item) for item in data)) # 确保项目是字符串以便连接
            else:
                f.write(str(data)) # 确保数据是字符串
    app.logger.debug(f"Debug output for '{stage_name}' written to: {debug_filepath}")

# --- SRT 时间处理辅助函数 ---
def parse_srt_time(time_str):
    parts = time_str.replace(',', '.').split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_ms = float(parts[2])
    seconds = int(seconds_ms)
    milliseconds = int(round((seconds_ms - seconds) * 1000))
    if milliseconds < 0: milliseconds = 0
    elif milliseconds > 999: milliseconds = 999
    return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

def format_srt_time(td):
    total_seconds = int(td.total_seconds())
    milliseconds = td.microseconds // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# --- 字符长度计算辅助函数 ---
def get_char_length(text):
    length = 0
    for char in text:
        if ('\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4DBF' or '\uF900' <= char <= '\uFAFF' or
            '\u3000' <= char <= '\u303F' or '\uFF00' <= char <= '\uFFEF' or '\uAC00' <= char <= '\uD7AF' or
            '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF'):
            length += 2
        else:
            length += 1
    return length

# --- 标点转换和断句辅助函数 ---
FULL_TO_HALF_MAP = str.maketrans({
    '。': '.', '，': ',', '！': '!', '？': '?', '；': ';', '：': ':',
    '“': '"', '”': '"', '‘': "'", '’': "'", '（': '(', '）': ')',
    '【': '[', '】': ']', '《': '<', '》': '>', '—': '-', '…': '...',
    '·': '.', '、': ',', '　': ' '
})
HARD_BREAK_PUNCTUATIONS_FOR_SPLITTING = {'.', ',', '!', '?', ';', ':'}

def clean_segment_for_output(text: str, remove_ending_punc: bool) -> str:
    app.logger.debug(f"DEBUG PRE-FINAL CLEAN (Specific End Punc Removal & Space Removal): Original='{repr(text)}', RemoveEndingPunc={remove_ending_punc}")
    processed_text = text.translate(FULL_TO_HALF_MAP)
    app.logger.debug(f"DEBUG DURING FINAL CLEAN (Full to Half): '{repr(text)}' -> '{repr(processed_text)}'")
    text_no_spaces = re.sub(r'\s+', '', processed_text)
    app.logger.debug(f"DEBUG DURING FINAL CLEAN (All Spaces Removed): -> '{repr(text_no_spaces)}'")
    if remove_ending_punc:
        if text_no_spaces and text_no_spaces.endswith(('.', ',')):
            text_no_spaces = text_no_spaces[:-1]
            app.logger.debug(f"DEBUG DURING FINAL CLEAN (Specific Ending '.' or ',' Removed): -> '{repr(text_no_spaces)}'")
        if text_no_spaces.endswith('...'):
            text_no_spaces = text_no_spaces[:-3]
            app.logger.debug(f"DEBUG DURING FINAL CLEAN (Ellipsis Removed): -> '{repr(text_no_spaces)}'")
    final_text = text_no_spaces.strip()
    app.logger.debug(f"DEBUG POST-FINAL CLEAN: Final='{repr(final_text)}'")
    return final_text

def normalize_text_for_comparison(text: str) -> str:
    processed_text = text.translate(FULL_TO_HALF_MAP)
    text_no_spaces = re.sub(r'\s+', '', processed_text)
    return text_no_spaces.strip()

def merge_adjacent_duplicate_blocks(raw_blocks_str_list, original_filename_prefix):
    app.logger.debug(f"Entering merge_adjacent_duplicate_blocks with {len(raw_blocks_str_list)} raw blocks.")
    merged_blocks_data = []
    current_merged_block = None
    time_proximity_threshold = timedelta(milliseconds=200)

    for idx, block_str in enumerate(raw_blocks_str_list):
        lines = block_str.strip().split('\n')
        if len(lines) < 3:
            app.logger.debug(f"Skipping malformed raw block {idx}: {block_str[:50]}...")
            continue

        timestamps_str = lines[1].strip()
        start_time_str, end_time_str = timestamps_str.split(' --> ')
        current_start_td = parse_srt_time(start_time_str)
        current_end_td = parse_srt_time(end_time_str)
        current_raw_text = ''.join(line.strip() for line in lines[2:] if line.strip())

        if not current_raw_text:
            app.logger.debug(f"Skipping raw block {idx} with empty text content.")
            continue

        current_normalized_text = normalize_text_for_comparison(current_raw_text)

        if current_merged_block and \
           current_merged_block['normalized_text_content'] == current_normalized_text and \
           (current_start_td - current_merged_block['end_td']) <= time_proximity_threshold:

            current_merged_block['end_td'] = current_end_td
            app.logger.debug(f"MERGE: Extending block ending {format_srt_time(current_merged_block['start_td'])} for text '{current_normalized_text[:20]}...' to {format_srt_time(current_end_td)}")
        else:
            if current_merged_block:
                merged_blocks_data.append(current_merged_block)
                app.logger.debug(f"ADD: Appended merged block with text '{current_merged_block['normalized_text_content'][:20]}...' starting {format_srt_time(current_merged_block['start_td'])}")

            current_merged_block = {
                'start_td': current_start_td,
                'end_td': current_end_td,
                'text_content': current_raw_text,
                'normalized_text_content': current_normalized_text
            }
            app.logger.debug(f"START NEW: New merged block with text '{current_normalized_text[:20]}...' starting {format_srt_time(current_start_td)}")

    if current_merged_block:
        merged_blocks_data.append(current_merged_block)
        app.logger.debug(f"FINAL ADD: Appended last merged block with text '{current_merged_block['normalized_text_content'][:20]}...' starting {format_srt_time(current_merged_block['start_td'])}")

    app.logger.debug(f"Finished merge_adjacent_duplicate_blocks. Total merged blocks: {len(merged_blocks_data)}")
    return merged_blocks_data

def split_long_srt_lines_logic(srt_content, max_chars_per_line=30, min_chars_per_line=10, short_sentence_threshold=None, remove_punctuation=True, original_filename_prefix="debug_file"):
    app.logger.debug(f"进入 split_long_srt_lines_logic，max_chars_per_line={max_chars_per_line}, min_chars_per_line={min_chars_per_line}, remove_punctuation={remove_punctuation}")
    OVERLAP_ALLOWANCE = 4
    global HanLP_tokenizer
    raw_srt_blocks_list = re.split(r'\r?\n\s*\r?\n', srt_content.strip())
    write_debug_output("raw_blocks", raw_srt_blocks_list, original_filename_prefix)
    merged_logic_blocks = merge_adjacent_duplicate_blocks(raw_srt_blocks_list, original_filename_prefix)
    write_debug_output("merged_blocks", merged_logic_blocks, original_filename_prefix)

    if HanLP_tokenizer is None:
        app.logger.warning("HanLP 模型未加载或加载失败，无法执行智能断句。将退回默认规则。")
        return _fallback_split_long_srt_lines_logic(merged_logic_blocks, max_chars_per_line, min_chars_per_line, short_sentence_threshold, remove_punctuation, original_filename_prefix)

    new_srt_blocks = []
    current_block_index = 1

    for merged_block_data in merged_logic_blocks:
        original_start_td = merged_block_data['start_td']
        original_end_td = merged_block_data['end_td']
        text_content_from_raw_merged_block = merged_block_data['text_content']
        original_duration = original_end_td - original_start_td

        if original_duration.total_seconds() <= 0:
            app.logger.warning(f"合并块的时长为0或负数，无法分配时间戳。跳过。文本: '{text_content_from_raw_merged_block[:20]}...'")
            continue

        app.logger.debug(f"处理合并后的块 (时间: {format_srt_time(original_start_td)} --> {format_srt_time(original_end_td)})，文本：'{repr(text_content_from_raw_merged_block)}'")
        text_content_for_splitting = text_content_from_raw_merged_block.translate(FULL_TO_HALF_MAP)

        pre_split_segments = []
        current_pre_segment_chars = []
        for char_idx, char in enumerate(text_content_for_splitting):
            current_pre_segment_chars.append(char)
            if char in HARD_BREAK_PUNCTUATIONS_FOR_SPLITTING:
                pre_split_segments.append("".join(current_pre_segment_chars).strip())
                current_pre_segment_chars = []
        if current_pre_segment_chars:
            pre_split_segments.append("".join(current_pre_segment_chars).strip())

        if not pre_split_segments:
            pre_split_segments = [text_content_from_raw_merged_block]

        app.logger.debug(f"强标点预分割结果: {repr(pre_split_segments)}")

        all_initial_text_lines_for_block = []

        for pre_segment_text in pre_split_segments:
            if not pre_segment_text: continue

            hanlp_result_doc = HanLP_tokenizer(pre_segment_text)
            doc_json = json.loads(hanlp_result_doc.to_json())
            tokens = doc_json.get('tok', [])

            if not tokens:
                app.logger.warning(f"HanLP 未能为预分割段 '{repr(pre_segment_text[:20])}' 生成有效词语列表。退回字符级分割。")
                tokens = list(pre_segment_text)

            if not tokens: continue

            current_line_tokens = []

            for token in tokens:
                token_cleaned = token.strip()
                if not token_cleaned: continue

                token_char_len = get_char_length(token_cleaned)
                current_line_text_so_far = "".join(current_line_tokens)
                current_line_char_len = get_char_length(current_line_text_so_far)
                potential_new_line_text = current_line_text_so_far + token_cleaned
                potential_new_line_char_len = get_char_length(potential_new_line_text)

                app.logger.debug(f"TOKEN: '{repr(token_cleaned)}' (len:{token_char_len}), CURRENT_LINE: '{repr(current_line_text_so_far)}' (len:{current_line_char_len}), POTENTIAL_LINE: '{repr(potential_new_line_text)}' (len:{potential_new_line_char_len}), MaxChars:{max_chars_per_line}")

                should_break_before_token = False

                is_two_char_chinese_word = (token_char_len == 4 and len(token_cleaned) == 2 and
                                            '\u4e00' <= token_cleaned[0] <= '\u9fff' and
                                            '\u4e00' <= token_cleaned[1] <= '\u9fff')

                if token_cleaned == '，' and current_line_char_len >= min_chars_per_line:
                    should_break_before_token = True
                    app.logger.debug(f"DEBUG: Triggered comma break for '{repr(token_cleaned)}'.")
                elif potential_new_line_char_len > max_chars_per_line:
                    if is_two_char_chinese_word and potential_new_line_char_len <= (max_chars_per_line + OVERLAP_ALLOWANCE):
                        app.logger.debug(f"DEBUG: Allowing 2-char word '{repr(token_cleaned)}' to slightly exceed max_chars_per_line ({potential_new_line_char_len} vs {max_chars_per_line}).")
                        should_break_before_token = False
                    elif current_line_char_len >= min_chars_per_line:
                        should_break_before_token = True
                        app.logger.debug(f"DEBUG: Breaking before token '{repr(token_cleaned)}' due to length and current line meets min length.")
                    else:
                        should_break_before_token = True
                        app.logger.debug(f"DEBUG: Forced break because current line is too short and next token makes it too long (or general overflow). Token: '{repr(token_cleaned)}'")

                if token_char_len > max_chars_per_line:
                    app.logger.warning(f"单个词语 '{repr(token_cleaned)}' (长度 {token_char_len}) 超过每行最大字符数 {max_chars_per_line}，进行字符级强制截断。")
                    if current_line_tokens:
                        all_initial_text_lines_for_block.append("".join(current_line_tokens).strip())
                        app.logger.debug(f" 　超长词前添加剩余部分：'{repr(all_initial_text_lines_for_block[-1])}'")

                    temp_remaining_token = token_cleaned
                    while temp_remaining_token:
                        split_point_for_char_cut = 0
                        current_segment_len_for_cut = 0
                        for char_i, char in enumerate(temp_remaining_token):
                            current_segment_len_for_cut += get_char_length(char)
                            if current_segment_len_for_cut > max_chars_per_line:
                                split_point_for_char_cut = char_i
                                break
                            
                        if split_point_for_char_cut == 0 and len(temp_remaining_token) > 0: split_point_for_char_cut = 1
                        part = temp_remaining_token[:split_point_for_char_cut].strip()
                        temp_remaining_token = temp_remaining_token[split_point_for_char_cut:].strip()
                        if part: all_initial_text_lines_for_block.append(part)
                        if not part and temp_remaining_token: break # 防止死循环
                        if not temp_remaining_token and part: break # 处理完毕
                    current_line_tokens = [] # 重置，因为超长 token 已处理
                    continue # 跳过当前 token 的常规处理，直接进入下一个 token

                if should_break_before_token and current_line_tokens: # 确保有内容才断开成新行
                    all_initial_text_lines_for_block.append(current_line_text_so_far.strip())
                    app.logger.debug(f" 　条件性断句：添加 '{repr(all_initial_text_lines_for_block[-1])}' (len:{get_char_length(all_initial_text_lines_for_block[-1])})")
                    current_line_tokens = [token_cleaned] # 以当前 token 开始新行
                else: # 不断句，将当前 token 追加到当前行
                    current_line_tokens.append(token_cleaned)

            if current_line_tokens:
                all_initial_text_lines_for_block.append("".join(current_line_tokens).strip())
                app.logger.debug(f" 　最终添加剩余词语：'{repr(all_initial_text_lines_for_block[-1])}' (len:{get_char_length(all_initial_text_lines_for_block[-1])})")

        # --- 第三步：对所有生成的文本行进行初步清理：移除所有空格，并有条件地移除行末句号或逗号 ---
        cleaned_raw_segments = [] # 临时存储经过初步清理的段落

        for segment_text in all_initial_text_lines_for_block:
            # 调用新的统一的清理函数
            processed_segment = clean_segment_for_output(segment_text, remove_punctuation)

            if processed_segment: # 仅当处理后非空时才添加
                cleaned_raw_segments.append(processed_segment)

        if not cleaned_raw_segments:
            app.logger.warning(f"块处理后未生成任何有效文本段，将跳过此块。")
            continue

        app.logger.debug(f"初步清理后的文本段落 ({len(cleaned_raw_segments)} 段): {repr(cleaned_raw_segments)}")

        # --- 第三点五步：后处理以强制最小行长度，通过两阶段合并短行 ---
        final_text_segments_for_time_allocation = []
        # 确保至少有一个初始段落
        if cleaned_raw_segments:
            # 第一阶段：尝试将短行与前一行合并
            temp_merged_segments_phase1 = []
            if cleaned_raw_segments:
                temp_merged_segments_phase1.append(cleaned_raw_segments[0])
                for i in range(1, len(cleaned_raw_segments)):
                    current_segment = cleaned_raw_segments[i]
                    last_temp_segment = temp_merged_segments_phase1[-1]

                    if (get_char_length(current_segment) < min_chars_per_line) and \
                       (get_char_length(last_temp_segment + current_segment) <= (max_chars_per_line + OVERLAP_ALLOWANCE)):

                        temp_merged_segments_phase1[-1] += current_segment
                        app.logger.debug(f"DEBUG MERGE (Phase 1: Previous): Merged '{repr(current_segment)}' with previous. New line: '{repr(temp_merged_segments_phase1[-1])}'")
                    else:
                        temp_merged_segments_phase1.append(current_segment)

            # 第二阶段：处理第一阶段后可能仍然存在的短行，尝试与下一行合并
            i = 0
            while i < len(temp_merged_segments_phase1):
                current_seg = temp_merged_segments_phase1[i]
                current_seg_len = get_char_length(current_seg)

                if current_seg_len < min_chars_per_line:
                    if (i + 1) < len(temp_merged_segments_phase1):
                        next_seg = temp_merged_segments_phase1[i+1]
                        potential_merge_text_with_next = current_seg + next_seg

                        if get_char_length(potential_merge_text_with_next) <= (max_chars_per_line + OVERLAP_ALLOWANCE):
                            final_text_segments_for_time_allocation.append(potential_merge_text_with_next)
                            app.logger.debug(f"DEBUG MERGE (Phase 2: Next): Merged '{repr(current_seg)}' with next '{repr(next_seg)}'. New line: '{repr(potential_merge_text_with_next)}'")
                            i += 1 # 跳过下一个段落因为它已被合并
                        else:
                            final_text_segments_for_time_allocation.append(current_seg)
                            app.logger.warning(f"WARNING: Short line '{repr(current_seg)}' (len:{current_seg_len}) could not be merged in phase 2, appending as is.")
                    else:
                        final_text_segments_for_time_allocation.append(current_seg)
                        app.logger.warning(f"WARNING: Last segment '{repr(current_seg)}' (len:{current_seg_len}) is short and cannot be merged.")
                else:
                    final_text_segments_for_time_allocation.append(current_seg)
                i += 1

            if not final_text_segments_for_time_allocation: # 再次检查是否为空
                app.logger.warning(f"合并后的文本段落为空，将跳过此块。")
                continue

        else: # 如果 cleaned_raw_segments 为空，则 final_cleaned_segments 也为空
            final_text_segments_for_time_allocation = []

        write_debug_output("final_segments", final_text_segments_for_time_allocation, original_filename_prefix)

        # --- 第四步：时间戳重新分配并生成新的字幕块 ---
        num_new_lines = len(final_text_segments_for_time_allocation)
        if num_new_lines == 0:
            app.logger.warning(f"原始块处理后未生成任何有效文本行（二次检查）。")
            continue

        if original_duration.total_seconds() <= 0:
            app.logger.warning(f"原始块的时长为0或负数，无法分配时间戳。跳过。")
            continue

        # 计算总的原始文本显示宽度，用于时间分配（使用处理后的文本长度计算）
        total_cleaned_display_length_for_time_calc = sum(get_char_length(s) for s in final_text_segments_for_time_allocation)

        if total_cleaned_display_length_for_time_calc == 0:
            # 如果所有内容都被移除，分配一个默认的显示宽度来避免除以零
            # 这种情况通常不会发生，因为我们已经过滤掉了空行
            total_cleaned_display_length_for_time_calc = 1
            app.logger.warning(f"原始块文本处理后显示宽度为0，使用默认值1进行时间分配。")

        # 计算每显示宽度对应的毫秒数 (浮点数)
        ms_per_display_unit = (original_duration.total_seconds() * 1000) / total_cleaned_display_length_for_time_calc

        app.logger.debug(f"原始块：总时长 {original_duration.total_seconds() * 1000}ms，总处理后显示宽度 {total_cleaned_display_length_for_time_calc}，每显示宽度 {ms_per_display_unit:.2f}ms。")

        current_segment_start_td = original_start_td
        for i, segment_text_final in enumerate(final_text_segments_for_time_allocation):
            # segment_text_final 已经是经过处理和清洁的最终文本

            segment_display_length = get_char_length(segment_text_final)

            estimated_segment_ms = segment_display_length * ms_per_display_unit
            estimated_segment_td = timedelta(milliseconds=estimated_segment_ms)

            segment_end_td = current_segment_start_td + estimated_segment_td

            # 最后一个片段的结束时间强制为原始块的结束时间，以消除累积误差
            if i == num_new_lines - 1:
                segment_end_td = original_end_td

            # 确保结束时间不会早于或等于开始时间，且至少有一个最小持续时间
            min_duration_td = timedelta(milliseconds=100) # 最小100毫秒
            if segment_end_td <= current_segment_start_td + min_duration_td:
                segment_end_td = current_segment_start_td + min_duration_td

            # 确保结束时间不会超过原始块的结束时间
            if segment_end_td > original_end_td:
                segment_end_td = original_end_td

            # 确保开始时间不小于原始开始时间 (主要用于防止第一句被计算成负值)
            if current_segment_start_td < original_start_td:
                current_segment_start_td = original_start_td

            new_srt_blocks.append(
                f"{current_block_index}\n"
                f"{format_srt_time(current_segment_start_td)} --> {format_srt_time(segment_end_td)}\n"
                f"{segment_text_final}\n"
            )
            app.logger.debug(f"生成新块 {current_block_index}：'{repr(segment_text_final)}'，时间: {format_srt_time(current_segment_start_td)} --> {format_srt_time(segment_end_td)} (显示宽度: {segment_display_length})")
            current_block_index += 1
            current_segment_start_td = segment_end_td # 下一个片段的开始时间是当前片段的结束时间

    app.logger.debug("退出 split_long_srt_lines_logic。")
    return '\n'.join(new_srt_blocks).strip() + '\n'

# --- 回退函数：当 HanLP 模型未加载时使用原有的分割逻辑 (已更新以匹配新要求) ---
def _fallback_split_long_srt_lines_logic(merged_logic_blocks, max_chars_per_line=30, min_chars_per_line=10, short_sentence_threshold=None, remove_punctuation=True, original_filename_prefix="debug_file_fallback"):
    """
    当 HanLP 模型未加载时，回退到原有的基于标点符号和长度的分割逻辑。
    此函数已更新以使用 get_char_length 和新的 clean_segment_for_output 函数，并更严格地处理断句。
    参数与 split_long_srt_lines_logic 相同。
    接收的是已经合并后的逻辑块列表。
    """
    app.logger.warning("HanLP 模型未加载，使用回退的基于标点符号和长度的分割逻辑。")

    # 定义允许合并短行的额外显示字符数 (相当于 2个汉字)
    MERGE_OVERLAP_ALLOWANCE = 4

    new_srt_blocks = []
    current_block_index = 1

    # --- 遍历合并后的逻辑块进行分行 ---
    for merged_block_data in merged_logic_blocks:
        original_start_td = merged_block_data['start_td']
        original_end_td = merged_block_data['end_td']
        text_content_from_raw_merged_block = merged_block_data['text_content']
        original_duration = original_end_td - original_start_td

        if original_duration.total_seconds() <= 0:
            app.logger.warning(f"合并块的时长为0或负数，无法分配时间戳。跳过。文本: '{text_content_from_raw_merged_block[:20]}...'")
            continue

        app.logger.debug(f"处理合并后的块 (回退模式, 时间: {format_srt_time(original_start_td)} --> {format_srt_time(original_end_td)})，文本：'{repr(text_content_from_raw_merged_block)}'")
        text_content_for_splitting = text_content_from_raw_merged_block.translate(FULL_TO_HALF_MAP)

        pre_split_segments = []
        current_pre_segment_chars = []
        for char_idx, char in enumerate(text_content_for_splitting):
            current_pre_segment_chars.append(char)
            if char in HARD_BREAK_PUNCTUATIONS_FOR_SPLITTING:
                pre_split_segments.append("".join(current_pre_segment_chars).strip())
                current_pre_segment_chars = []
        if current_pre_segment_chars:
            pre_split_segments.append("".join(current_pre_segment_chars).strip())

        if not pre_split_segments:
            pre_split_segments = [text_content_from_raw_merged_block]

        raw_processed_text_segments = []
        for segment_to_process in pre_split_segments:
            if not segment_to_process: continue
            temp_sub_lines = []
            current_line_temp_chars = []
            temp_chars_as_tokens = list(segment_to_process)

            for char_i, char_token in enumerate(temp_chars_as_tokens):
                char_cleaned = char_token.strip()
                if not char_cleaned: continue
                char_len = get_char_length(char_cleaned)
                current_line_char_len = get_char_length("".join(current_line_temp_chars))
                potential_new_line_char_len = current_line_char_len + char_len
                should_break_before_char = False

                if char_cleaned == '，' and current_line_char_len >= min_chars_per_line:
                    should_break_before_char = True
                elif potential_new_line_char_len > max_chars_per_line:
                    if current_line_char_len >= min_chars_per_line:
                        should_break_before_char = True
                    elif char_len > max_chars_per_line: # 单个字符超长，强制截断
                        remaining_chars_to_cut = char_cleaned
                        while remaining_chars_to_cut:
                            cut_idx = 0
                            current_segment_len_for_cut = 0
                            for c_i, c in enumerate(remaining_chars_to_cut):
                                current_segment_len_for_cut += get_char_length(c)
                                if current_segment_len_for_cut > max_chars_per_line:
                                    cut_idx = c_i
                                    break
                            if cut_idx == 0 and len(remaining_chars_to_cut) > 0: cut_idx = 1
                            part = remaining_chars_to_cut[:cut_idx].strip()
                            remaining_chars_to_cut = remaining_chars_to_cut[cut_idx:].strip()
                            if part: temp_sub_lines.append(part)
                            if not part and remaining_chars_to_cut: break
                            if not remaining_chars_to_cut and part: break
                        current_line_temp_chars = []
                        continue # 跳过当前字符的常规处理
                    else:
                        should_break_before_char = True

                if should_break_before_char and current_line_temp_chars:
                    temp_sub_lines.append("".join(current_line_temp_chars).strip())
                    current_line_temp_chars = [char_cleaned]
                else:
                    current_line_temp_chars.append(char_cleaned)

            if current_line_temp_chars:
                temp_sub_lines.append("".join(current_line_temp_chars).strip())
            raw_processed_text_segments.extend(temp_sub_lines)

            # 最终后处理：移除所有空格，并有条件地移除行末句号或逗号
            cleaned_raw_segments = []

            for line in raw_processed_text_segments:
                # 调用统一的清理函数
                processed_line = clean_segment_for_output(line, remove_punctuation)

                if processed_line:
                    cleaned_raw_segments.append(processed_line)

            if not cleaned_raw_segments:
                continue

            # --- 后处理以强制最小行长度，通过合并短行 ---
            final_cleaned_segments = []
            if cleaned_raw_segments:
                final_cleaned_segments.append(cleaned_raw_segments[0])

                # 第一阶段：尝试将短行与前一行合并
                temp_merged_segments = [cleaned_raw_segments[0]]
                for i in range(1, len(cleaned_raw_segments)):
                    current_segment = cleaned_raw_segments[i]
                    last_temp_segment = temp_merged_segments[-1]

                    if (get_char_length(current_segment) < min_chars_per_line) and \
                       (get_char_length(last_temp_segment + current_segment) <= (max_chars_per_line + MERGE_OVERLAP_ALLOWANCE)):

                        temp_merged_segments[-1] += current_segment
                    else:
                        temp_merged_segments.append(current_segment)

                # 第二阶段：处理第一阶段后可能仍然存在的短行，尝试与下一行合并
                i = 0
                while i < len(temp_merged_segments):
                    current_seg = temp_merged_segments[i]
                    current_seg_len = get_char_length(current_seg)

                    if current_seg_len < min_chars_per_line:
                        if (i + 1) < len(temp_merged_segments):
                            next_seg = temp_merged_segments[i+1]
                            potential_merge_text_with_next = current_seg + next_seg
                            if get_char_length(potential_merge_text_with_next) <= (max_chars_per_line + MERGE_OVERLAP_ALLOWANCE):
                                final_cleaned_segments.append(potential_merge_text_with_next)
                                i += 1
                            else:
                                final_cleaned_segments.append(current_seg)
                        else:
                            final_cleaned_segments.append(current_seg)
                    else:
                        final_cleaned_segments.append(current_seg)
                    i += 1

            else:
                final_cleaned_segments = []

            final_text_segments_for_time_allocation = final_cleaned_segments
            write_debug_output("final_segments_fallback", final_text_segments_for_time_allocation, original_filename_prefix)

            num_new_lines = len(final_text_segments_for_time_allocation)
            if num_new_lines == 0: continue
            if original_duration.total_seconds() <= 0: continue
            total_original_char_length = sum(get_char_length(s) for s in final_text_segments_for_time_allocation)
            if total_original_char_length == 0: total_original_char_length = 1
            ms_per_display_unit = (original_duration.total_seconds() * 1000) / total_original_char_length

            current_segment_start_td = original_start_td
            for i, segment_text in enumerate(final_text_segments_for_time_allocation):
                segment_text_final = segment_text
                if not segment_text_final: continue
                segment_display_length = get_char_length(segment_text_final)
                estimated_segment_ms = segment_display_length * ms_per_display_unit
                estimated_segment_td = timedelta(milliseconds=estimated_segment_ms)
                segment_end_td = current_segment_start_td + estimated_segment_td
                if i == num_new_lines - 1: segment_end_td = original_end_td
                min_duration_td = timedelta(milliseconds=100)
                if segment_end_td <= current_segment_start_td + min_duration_td: segment_end_td = current_segment_start_td + min_duration_td
                if segment_end_td > original_end_td: segment_end_td = original_end_td
                if current_segment_start_td < original_start_td: current_segment_start_td = original_start_td

                new_srt_blocks.append(
                    f"{current_block_index}\n"
                    f"{format_srt_time(current_segment_start_td)} --> {format_srt_time(segment_end_td)}\n"
                    f"{segment_text_final}\n"
                )
                current_block_index += 1
                current_segment_start_td = segment_end_td

    return '\n'.join(new_srt_blocks).strip() + '\n'

def handle_file_processing(files, max_chars, remove_punctuation, min_chars_per_line, short_sentence_threshold):
    app.logger.debug(f"进入 handle_file_processing，处理 {len(files)} 个文件，remove_punctuation={remove_punctuation}。")
    processed_file_info_list = []
    processed_filepaths_for_zip = []
    file_sequence_counters = {}

    for file_obj in files:
        if file_obj and file_obj.filename:
            original_filename = secure_filename(file_obj.filename)
            if original_filename not in file_sequence_counters:
                file_sequence_counters[original_filename] = 0

            unique_id = uuid.uuid4().hex
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            unique_upload_filename = f"{unique_id}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_upload_filename)
            app.logger.debug(f"保存原始文件到: {filepath}")
            file_obj.save(filepath)
            app.logger.info(f"文件 '{original_filename}' 已保存到 {filepath}")

            srt_content = ""
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    srt_content = f.read()
                app.logger.debug(f"文件 '{original_filename}' 以 UTF-8 编码读取成功。")
            except UnicodeDecodeError:
                app.logger.warning(f"文件 '{original_filename}' UTF-8 解码失败，尝试 GBK。")
                try:
                    with open(filepath, 'r', encoding='gbk') as f:
                        srt_content = f.read()
                    app.logger.debug(f"文件 '{original_filename}' 以 GBK 编码读取成功。")
                except Exception as e:
                    app.logger.error(f"文件 {original_filename} 编码错误或内容无效: {e}", exc_info=True)
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    processed_file_info_list.append({
                        'original_name': original_filename,
                        'error': f"处理失败：文件编码错误或内容无效。请确保其为 UTF-8 或 GBK 编码。错误: {e}"
                    })
                    continue
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    app.logger.debug(f"已删除上传的临时文件: {filepath}")

            processed_srt_content = split_long_srt_lines_logic(
                srt_content,
                max_chars_per_line=max_chars,
                min_chars_per_line=min_chars_per_line,
                short_sentence_threshold=short_sentence_threshold,
                remove_punctuation=remove_punctuation,
                original_filename_prefix=os.path.splitext(original_filename)[0][:4] if len(os.path.splitext(original_filename)[0]) >= 4 else os.path.splitext(original_filename)[0]
            )
            app.logger.debug(f"文件 '{original_filename}' 处理完成，生成处理后内容前200字：{processed_srt_content[:200].replace(chr(10), ' ').replace(chr(13), ' ')}")

            base_name = os.path.splitext(original_filename)[0]
            prefix = base_name[:4] if len(base_name) >=4 else base_name
            current_date = datetime.now().strftime("%Y-%m-%d")
            file_sequence_counters[original_filename] += 1
            current_sequence = file_sequence_counters[original_filename]
            processed_filename = f"{prefix}-{current_date}-{current_sequence}.srt"
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

            if not os.path.exists(app.config['PROCESSED_FOLDER']):
                os.makedirs(app.config['PROCESSED_FOLDER'])

            with open(processed_filepath, 'w', encoding='utf-8') as f:
                f.write(processed_srt_content)
            app.logger.info(f"处理后的文件 '{original_filename}' 已保存为 '{processed_filename}' 到 {processed_filepath}")

            processed_file_info_list.append({
                'original_name': original_filename,
                'processed_name': processed_filename,
                'download_url': url_for('download_file', filename=processed_filename),
                'preview_url': url_for('preview_file', filename=processed_filename)
            })
            processed_filepaths_for_zip.append(processed_filepath)
        else:
            app.logger.warning(f"跳过空文件或没有文件名的文件：{file_obj.filename if file_obj else 'N/A'}")

    app.logger.debug("退出 handle_file_processing。")
    if not processed_file_info_list and files:
        raise ValueError("没有找到有效的SRT文件进行处理，或者所有文件处理均失败。")
    return {"web_info": processed_file_info_list, "zip_paths": processed_filepaths_for_zip, "file_error": False}

@app.route('/', methods=['GET', 'POST'])
def index():
    app.logger.debug(f"接收到 '{request.method}' 请求到根路由。")
    if request.method == 'POST':
        try:
            files = request.files.getlist('files')
            if not files or all(f.filename == '' for f in files):
                flash('没有选择文件！', 'error')
                app.logger.warning("用户未选择文件。")
                return redirect(url_for('index'))

            max_chars_str = request.form.get('max_chars_per_line', '30')
            remove_punc_option = request.form.get('remove_punctuation_checkbox') == 'on'

            print(f"DEBUG: Checkbox state (remove_punctuation_option): {remove_punc_option}")

            current_min_chars = 10
            current_short_sentence_threshold = 30

            try:
                max_chars = int(max_chars_str)
                if not (10 <= max_chars <= 100):
                    max_chars = 30
                    flash('每行最大字符数（显示宽度）应在10到100之间，已重置为30。', 'warning')
                    app.logger.warning(f"用户输入 max_chars_per_line 超出范围，重置为 {max_chars}。")
            except ValueError:
                max_chars = 30
                flash('每行最大字符数（显示宽度）输入无效，已重置为30。', 'warning')
                app.logger.warning(f"用户输入 max_chars_per_line 无效，重置为 {max_chars}。")

            app.logger.info(f"用户上传文件，max_chars_per_line={max_chars}, remove_punctuation={remove_punc_option}.")

            result = handle_file_processing(
                files,
                max_chars,
                remove_punc_option,
                min_chars_per_line=current_min_chars,
                short_sentence_threshold=current_short_sentence_threshold
            )

            if result.get("file_error"):
                if "error" in result:
                    flash(result["error"], 'error')
                app.logger.error(f"文件处理失败：{result.get('error', '未知错误')}")
                return render_template('index.html', processed_files=result.get("web_info", []), last_max_chars=max_chars, last_remove_punc=remove_punc_option)

            app.logger.info(f"文件处理成功，渲染结果页面，处理了 {len(result['web_info'])} 个文件。")
            return render_template('index.html', processed_files=result["web_info"], last_max_chars=max_chars, last_remove_punc=remove_punc_option)

        except Exception as e:
            app.logger.exception("处理POST请求时发生未预期错误：")
            flash(f"处理文件时发生内部错误：{e}", 'error')
            return render_template('index.html', processed_files=[], last_max_chars=30, last_remove_punc=True)

    app.logger.debug("GET 请求，渲染初始页面。")
    return render_template('index.html', processed_files=[], last_max_chars=30, last_remove_punc=True)

@app.route('/download/<filename>')
def download_file(filename):
    app.logger.info(f"请求下载文件: {filename}")
    try:
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        flash(f"下载文件 {filename} 未找到。", 'error')
        app.logger.error(f"下载文件 {filename} 未找到。")
        return redirect(url_for('index'))

@app.route('/preview/<filename>')
def preview_file(filename):
    app.logger.info(f"请求预览文件: {filename}")
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(content, mimetype='text/plain; charset=utf-8')
    except FileNotFoundError:
        app.logger.error(f"预览文件 {filename} 未找到。")
        return "文件未找到。", 404
    except Exception as e:
        app.logger.error(f"预览文件 {filename} 失败: {e}")
        return f"读取文件失败: {e}", 500

@app.route('/api/process_subtitles', methods=['POST'])
def process_subtitles_api():
    app.logger.debug("接收到 API 请求: /api/process_subtitles")
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        app.logger.warning("API 请求中未提供文件。")
        return jsonify({"error": "No files provided"}), 400

    max_chars_str = request.form.get('max_chars_per_line', '30')
    remove_punc_option = request.form.get('remove_punctuation_checkbox') == 'on'

    current_min_chars = 10
    current_short_sentence_threshold = 30

    try:
        max_chars = int(max_chars_str)
        if not (10 <= max_chars <= 100):
            max_chars = 30
            app.logger.warning(f"API 请求 max_chars_per_line 超出范围，重置为 {max_chars}。")
    except ValueError:
        max_chars = 30
        app.logger.warning(f"API 请求 max_chars_per_line 无效，重置为 {max_chars}。")

    app.logger.info(f"API 接收到 {len(files)} 个文件，max_chars_per_line={max_chars}, remove_punctuation={remove_punc_option}.")

    result = handle_file_processing(
        files,
        max_chars,
        remove_punc_option,
        min_chars_per_line=current_min_chars,
        short_sentence_threshold=current_short_sentence_threshold
    )

    if result.get("file_error"):
        app.logger.error(f"API 文件处理失败：{result['error']}")
        return jsonify({"error": result["error"], "details": result.get("web_info")}), 400

    processed_filepaths_for_zip = result["zip_paths"]

    if not processed_filepaths_for_zip:
        app.logger.error("没有有效文件被处理，无法生成ZIP。")
        return jsonify({"error": "No valid files were processed."}), 500

    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in processed_filepaths_for_zip:
                zf.write(file_path, os.path.basename(file_path))
                os.remove(file_path)
                app.logger.debug(f"已将文件 {os.path.basename(file_path)} 添加到ZIP并删除原始文件。")

        zip_buffer.seek(0)
        app.logger.info("成功生成处理后的ZIP文件。")
        return send_file(zip_buffer,
                         mimetype='application/zip',
                         as_attachment=True,
                         download_name='processed_subtitles.zip')
    except Exception as e:
        app.logger.error(f"生成或发送ZIP文件时发生错误: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate or send zip file: {e}"}), 500

if __name__ == '__main__':
    app.logger.info("Flask 应用即将启动...")
    app.run(debug=False, host='0.0.0.0', port=5000)
