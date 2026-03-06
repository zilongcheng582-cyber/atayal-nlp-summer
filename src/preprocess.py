# Atayal Text Preprocessing Pipeline
import urllib.request
import string


def download_data(num_chapters=12):
    """从 GitHub 下载原始数据"""
    for i in range(1, num_chapters + 1):
        url = f"https://raw.githubusercontent.com/zilongcheng582-cyber/atayal-nlp-summer/main/data/raw/{i}.pyu"
        urllib.request.urlretrieve(url, f"{i}.pyu")
    print(f"下载完成！共 {num_chapters} 章")


def load_and_clean(num_chapters=12):
    """读取并清洗数据，返回干净的句子列表"""
    def has_chinese(text):
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    all_lines = []
    for i in range(1, num_chapters + 1):
        with open(f"{i}.pyu", "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        all_lines.extend(lines)

    clean = [line for line in all_lines if not has_chinese(line)]
    print(f"清洗完成：{len(clean)} 句")
    return clean


def build_vocab(clean_lines):
    """建立字符集和映射字典"""
    full_text = "\n".join(clean_lines)
    chars = sorted(set(full_text))
    char2idx = {char: idx for idx, char in enumerate(chars)}
    idx2char = {idx: char for idx, char in enumerate(chars)}
    return chars, char2idx, idx2char


def encode_and_split(clean_lines, char2idx, seq_len=50):
    """编码文本并切成训练对"""
    full_text = "\n".join(clean_lines)
    encoded = [char2idx[c] for c in full_text]

    inputs, targets = [], []
    for i in range(0, len(encoded) - seq_len):
        inputs.append(encoded[i: i + seq_len])
        targets.append(encoded[i + 1: i + seq_len + 1])

    return inputs, targets
