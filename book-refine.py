import difflib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm


# ============================================================
# 配置区：主要改这里
# ============================================================

CONFIG = {
    # 0 = 保守校勘模式
    # 1 = 上下文缺损修复模式
    "MODE": 1,

    # 输入输出路径
    "INPUT_TXT": "book.txt",
    "OUTPUT_TXT": "outputs/book-refined.txt",

    # LM Studio API
    "BASE_URL": "http://172.25.160.1:1234/v1",
    "MODEL": "qwopus3.6-27b-v1-preview",

    # 模型生成参数
    # MODE 0 建议低温，避免乱改。
    # MODE 1 可以略高一点，但仍然要保守。
    "TEMPERATURE": {
        0: 0.15,
        1: 0.20,
    },
    "TOP_P": {
        0: 0.90,
        1: 0.90,
    },

    # 单次最大输出 token。
    # 如果模型输出被截断，可以调大。
    "MAX_TOKENS": 81920,

    # 章节过长时，按字符数切块。
    # 中文小说建议 3000-5000。
    "CHUNK_CHARS": 30000,

    # MODE 1 下，每个 chunk 会额外提供前后文，但要求模型只输出当前 chunk。
    "CONTEXT_CHARS": 50000,

    # 是否启用断点续跑。
    # True 表示如果某章结果已经存在，就直接读取，不重复调用模型。
    "RESUME": True,

    # 是否自动回退可疑输出。
    # 如果 True，审计发现模型改动过大时，直接保留原文。
    "REJECT_SUSPICIOUS": False,

    # API 调用失败重试次数
    "RETRIES": 2,
}


# ============================================================
# Prompt 区
# ============================================================

SYSTEM_PROMPT_MODE_0 = """你是一名保守的中文小说校勘助手。

你的任务是修正小说文本中的明显问题，包括：
1. 删除混入正文中的广告、网站推广、下载提示、阅读提示、更新提示、乱码推广语。
2. 修正明显错别字、乱码、标点错误、断句错误和明显语病。
3. 保持原文剧情、人物关系、事件顺序、叙事视角和文风不变。

严格禁止：
1. 不得扩写、缩写、总结或续写。
2. 不得美化语言，不得把原文改成另一种文风。
3. 不得改变人物名、地名、功法名、组织名等专有名词。
4. 不得删除正常正文，即使它看起来啰嗦、不流畅或文笔一般。
5. 不得加入解释、评论、标题或“以下是修改后文本”。

判断广告的原则：
如果某段内容明显跳出小说叙事，包含网站名、阅读提示、下载提示、收藏提示、更新提示、盗版站推广、乱码推广语等，应删除。
如果不确定是否为正文，宁可保留，不要删除。

你只能输出修正后的小说正文，不要输出任何解释。
"""


USER_PROMPT_MODE_0 = """请对下面这段小说文本进行保守校勘。

要求：
- 只删除广告、推广语、乱码和明显无关内容。
- 只修正明显错别字、标点错误、断句错误和明显病句。
- 不改变剧情、人物、事件顺序和文风。
- 不要解释。
- 不要加标题。
- 不要输出“修改如下”。

待处理文本：
<<<
{current_text}
>>>
"""


SYSTEM_PROMPT_MODE_1 = """你是一名极其保守的中文小说缺损修复与校勘助手。

你的任务不是改写小说，而是在严格保持原文故事的前提下，修复由于 OCR、复制、排版或盗版文本污染造成的局部问题。

你允许做的事情：
1. 删除混入正文中的广告、网站推广、下载提示、阅读提示、更新提示、乱码推广语。
2. 修正明显错别字、乱码、标点错误、断句错误和明显语病。
3. 修复明显的 OCR 错漏，例如缺字、错字、断句断裂、语义残缺、前后句无法衔接。
4. 在上下文强约束下，补足极少量缺失词语、短语或连接成分，使句子恢复通顺。
5. 只在缺失内容能够由上下文唯一或高度确定地推断时，才允许做最小补全。
6. 可以在不改变文本内容的前提下，通过拆分或者合并，恢复合理自然段。

你严格禁止做的事情：
1. 不得自由续写。
2. 不得扩写情节。
3. 不得新增人物行动、心理描写、环境描写、打斗过程或对话内容。
4. 不得改变剧情、人物关系、事件顺序、叙事视角和文风。
5. 不得为了让文字更优美而重写正常句子。
6. 不得替换人物名、地名、功法名、组织名等专有名词，除非它明显是 OCR 错字。
7. 不得删除正常正文，即使它看起来啰嗦、不流畅或文笔一般。
8. 不得输出解释、评论、标题、说明或“以下是修改后文本”。

缺损修复原则：
- 如果一个地方只是写得不好，但语义完整，不要改。
- 如果一个地方疑似缺失，但无法根据上下文可靠判断缺了什么，保留原文，不要猜。
- 如果需要补全，只补最小必要内容。
- 如果前后文冲突，不要自行创造新设定来圆。
- 上下文只用于理解，不要把上下文重复输出。

你只能输出修复后的【当前待处理文本】，不要输出前文、后文或任何解释。
"""


USER_PROMPT_MODE_1 = """请根据上下文，对【当前待处理文本】进行保守缺损修复与校勘。

注意：
- 【前文参考】和【后文参考】只用于理解上下文。
- 你只能输出修复后的【当前待处理文本】。
- 不要输出前文参考。
- 不要输出后文参考。
- 不要解释。
- 不要加标题。
- 不要输出“修改如下”。

【前文参考】
<<<
{prev_context}
>>>

【当前待处理文本】
<<<
{current_text}
>>>

【后文参考】
<<<
{next_context}
>>>
"""


PROMPTS = {
    0: {
        "system": SYSTEM_PROMPT_MODE_0,
        "user": USER_PROMPT_MODE_0,
    },
    1: {
        "system": SYSTEM_PROMPT_MODE_1,
        "user": USER_PROMPT_MODE_1,
    },
}


# ============================================================
# 章节切分与基础工具
# ============================================================

CHAPTER_RE = re.compile(
    r"(?m)^\s*("
    r"(?:正文\s*)?"
    r"(?:第[零〇一二两三四五六七八九十百千万\d]+[章节回卷集][^\n]{0,80}"
    r"|Chapter\s+\d+[^\n]{0,80}"
    r"|卷[零〇一二两三四五六七八九十百千万\d]+[^\n]{0,80})"
    r")\s*$"
)

BAD_OUTPUT_PATTERNS = [
    "以下是",
    "修改后",
    "修正后",
    "处理后",
    "我已经",
    "处理完成",
    "校勘说明",
    "作为AI",
    "作为一个",
    "前文参考",
    "后文参考",
    "当前待处理文本",
]


def read_text_auto(path: Path) -> Tuple[str, str]:
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return path.read_text(encoding=enc), enc
        except UnicodeDecodeError:
            continue

    return path.read_text(encoding="utf-8", errors="replace"), "utf-8-replace"


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def safe_filename(name: str, max_len: int = 50) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name[:max_len] if name else "untitled"


def split_chapters(text: str) -> List[Dict[str, str]]:
    text = normalize_newlines(text)
    matches = list(CHAPTER_RE.finditer(text))

    if not matches:
        return [{"title": "全文", "text": text.strip()}]

    chapters = []

    if matches[0].start() > 0:
        preface = text[:matches[0].start()].strip()
        if preface:
            chapters.append({"title": "前言", "text": preface})

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        title = m.group(1).strip()
        chapters.append({"title": title, "text": block})

    return chapters


def split_long_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks = []
    current = ""

    for line in lines:
        if len(current) + len(line) > max_chars and current.strip():
            chunks.append(current.strip())
            current = line
        else:
            current += line

    if current.strip():
        chunks.append(current.strip())

    return chunks


def get_context(chunks: List[str], idx: int, context_chars: int) -> Tuple[str, str]:
    prev_context = ""
    next_context = ""

    if idx > 0:
        prev_context = chunks[idx - 1][-context_chars:]

    if idx + 1 < len(chunks):
        next_context = chunks[idx + 1][:context_chars]

    return prev_context, next_context


# ============================================================
# LLM 调用
# ============================================================

def call_lmstudio(
    client: OpenAI,
    mode: int,
    model: str,
    current_text: str,
    prev_context: str = "",
    next_context: str = "",
) -> str:
    if mode not in PROMPTS:
        raise ValueError(f"Unsupported MODE: {mode}. Use 0 or 1.")

    temperature = CONFIG["TEMPERATURE"][mode]
    top_p = CONFIG["TOP_P"][mode]
    max_tokens = CONFIG["MAX_TOKENS"]
    retries = CONFIG["RETRIES"]

    system_prompt = PROMPTS[mode]["system"]

    if mode == 0:
        user_prompt = PROMPTS[mode]["user"].format(
            current_text=current_text
        )
    else:
        user_prompt = PROMPTS[mode]["user"].format(
            prev_context=prev_context,
            current_text=current_text,
            next_context=next_context,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_err = None

    for _ in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False,
            )
            out = resp.choices[0].message.content or ""
            return out.strip()
        except Exception as e:
            last_err = e
            time.sleep(2)

    raise RuntimeError(f"LM Studio call failed after retries: {last_err}")


# ============================================================
# 审计与 diff
# ============================================================

def audit_change(original: str, cleaned: str, mode: int) -> Dict:
    orig_len = len(original)
    clean_len = len(cleaned)

    flags = []

    if clean_len == 0:
        flags.append("empty_output")

    if orig_len > 0:
        drop_ratio = (orig_len - clean_len) / orig_len
        gain_ratio = (clean_len - orig_len) / orig_len
    else:
        drop_ratio = 0.0
        gain_ratio = 0.0

    # MODE 0 不允许明显扩写。
    # MODE 1 允许少量补全，但大幅扩写仍然可疑。
    if mode == 0:
        max_drop = 0.25
        max_gain = 0.10
    else:
        max_drop = 0.30
        max_gain = 0.20

    if drop_ratio > max_drop:
        flags.append(f"large_deletion:{drop_ratio:.2%}")

    if gain_ratio > max_gain:
        flags.append(f"large_expansion:{gain_ratio:.2%}")

    for p in BAD_OUTPUT_PATTERNS:
        if p in cleaned:
            flags.append(f"model_meta_text:{p}")

    orig_lines = len(original.splitlines())
    clean_lines = len(cleaned.splitlines())

    if orig_lines > 5 and clean_lines < orig_lines * 0.5:
        flags.append("too_many_lines_removed")

    if mode == 0 and clean_lines > orig_lines * 1.5 and orig_lines > 5:
        flags.append("too_many_lines_added")

    if mode == 1 and clean_lines > orig_lines * 1.8 and orig_lines > 5:
        flags.append("too_many_lines_added")

    return {
        "mode": mode,
        "original_chars": orig_len,
        "cleaned_chars": clean_len,
        "drop_ratio": round(drop_ratio, 4),
        "gain_ratio": round(gain_ratio, 4),
        "original_lines": orig_lines,
        "cleaned_lines": clean_lines,
        "flags": flags,
        "suspicious": len(flags) > 0,
    }


def save_diff_html(original: str, cleaned: str, out_path: Path, title: str):
    diff = difflib.HtmlDiff(wrapcolumn=120).make_file(
        original.splitlines(),
        cleaned.splitlines(),
        fromdesc=f"original: {title}",
        todesc=f"cleaned: {title}",
        context=True,
        numlines=3,
    )
    out_path.write_text(diff, encoding="utf-8")


# ============================================================
# 章节处理
# ============================================================

def clean_chapter(
    client: OpenAI,
    mode: int,
    model: str,
    chapter_text: str,
) -> str:
    chunk_chars = CONFIG["CHUNK_CHARS"]
    context_chars = CONFIG["CONTEXT_CHARS"]

    chunks = split_long_text(chapter_text, chunk_chars)

    # 短章节：整章处理
    if len(chunks) == 1:
        return call_lmstudio(
            client=client,
            mode=mode,
            model=model,
            current_text=chapter_text,
            prev_context="",
            next_context="",
        )

    # 长章节：章节内部切块处理
    cleaned_chunks = []

    for idx, chunk in enumerate(chunks):
        prev_context, next_context = get_context(chunks, idx, context_chars)

        if mode == 0:
            # 保守模式不需要上下文，避免模型借上下文发挥。
            cleaned = call_lmstudio(
                client=client,
                mode=mode,
                model=model,
                current_text=chunk,
            )
        else:
            # 缺损修复模式需要上下文，但只允许输出当前 chunk。
            cleaned = call_lmstudio(
                client=client,
                mode=mode,
                model=model,
                current_text=chunk,
                prev_context=prev_context,
                next_context=next_context,
            )

        cleaned_chunks.append(cleaned)

    return "\n\n".join(cleaned_chunks)


# ============================================================
# 主流程
# ============================================================

def main():
    mode = CONFIG["MODE"]

    if mode not in [0, 1]:
        raise ValueError("CONFIG['MODE'] must be 0 or 1.")

    input_path = Path(CONFIG["INPUT_TXT"])
    output_path = Path(CONFIG["OUTPUT_TXT"])

    out_dir = output_path.parent
    chapters_dir = out_dir / "chapters"
    logs_dir = out_dir / "logs"

    out_dir.mkdir(parents=True, exist_ok=True)
    chapters_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_text, enc = read_text_auto(input_path)
    raw_text = normalize_newlines(raw_text)

    chapters = split_chapters(raw_text)

    print(f"Input: {input_path}")
    print(f"Encoding: {enc}")
    print(f"Mode: {mode}")
    print(f"Chapters found: {len(chapters)}")
    print(f"Model: {CONFIG['MODEL']}")
    print(f"Base URL: {CONFIG['BASE_URL']}")

    client = OpenAI(
        base_url=CONFIG["BASE_URL"],
        api_key="lm-studio",
    )

    cleaned_all = []
    audit_path = logs_dir / "audit.jsonl"

    with audit_path.open("w", encoding="utf-8") as audit_f:
        for i, chap in enumerate(tqdm(chapters, desc="Cleaning chapters"), start=1):
            title = chap["title"]
            original = chap["text"]

            name = f"{i:04d}_{safe_filename(title)}"
            chapter_out = chapters_dir / f"{name}.txt"
            diff_out = logs_dir / f"{name}.diff.html"

            if CONFIG["RESUME"] and chapter_out.exists():
                final_text = chapter_out.read_text(encoding="utf-8")
                audit = audit_change(original, final_text, mode)
                audit["resume"] = True
            else:
                try:
                    cleaned = clean_chapter(
                        client=client,
                        mode=mode,
                        model=CONFIG["MODEL"],
                        chapter_text=original,
                    )
                except Exception as e:
                    cleaned = original
                    audit = {
                        "mode": mode,
                        "index": i,
                        "title": title,
                        "error": str(e),
                        "fallback_to_original": True,
                        "resume": False,
                    }
                    audit_f.write(json.dumps(audit, ensure_ascii=False) + "\n")
                    cleaned_all.append(cleaned)
                    continue

                audit = audit_change(original, cleaned, mode)
                audit["resume"] = False

                if CONFIG["REJECT_SUSPICIOUS"] and audit["suspicious"]:
                    final_text = original
                    audit["fallback_to_original"] = True
                else:
                    final_text = cleaned
                    audit["fallback_to_original"] = False

                chapter_out.write_text(final_text, encoding="utf-8")
                save_diff_html(original, final_text, diff_out, title)

            audit_record = {
                "index": i,
                "title": title,
                "chapter_file": str(chapter_out),
                "diff_file": str(diff_out),
                **audit,
            }

            audit_f.write(json.dumps(audit_record, ensure_ascii=False) + "\n")
            cleaned_all.append(final_text)

    output_path.write_text("\n\n".join(cleaned_all), encoding="utf-8")

    print(f"Done.")
    print(f"Output saved to: {output_path}")
    print(f"Audit log saved to: {audit_path}")
    print(f"Diff files saved to: {logs_dir}")


if __name__ == "__main__":
    main()
