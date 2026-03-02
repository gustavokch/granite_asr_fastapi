"""Forced Alignment with Wav2Vec2.
Ported and simplified from WhisperX (C. Max Bain).
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SAMPLE_RATE = 16000


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float


def load_align_model(model_id: str, device: str):
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    vocab = processor.tokenizer.get_vocab()
    # Ensure all keys are lower-case for mapping
    dictionary = {char.lower(): code for char, code in vocab.items()}
    return model, dictionary


def get_wildcard_emission(frame_emission, tokens, blank_id):
    tokens = torch.tensor(tokens) if not isinstance(tokens, torch.Tensor) else tokens
    wildcard_mask = (tokens == -1)
    regular_scores = frame_emission[tokens.clamp(min=0).long()]
    max_valid_score = frame_emission.clone()
    max_valid_score[blank_id] = float("-inf")
    max_valid_score = max_valid_score.max()
    return torch.where(wildcard_mask, max_valid_score, regular_scores)


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)
    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + get_wildcard_emission(emission[t], tokens[1:], blank_id),
        )
    return trellis


def backtrack_beam(trellis, emission, tokens, blank_id=0, beam_width=5):
    T, J = trellis.size(0) - 1, trellis.size(1) - 1

    @dataclass
    class BeamState:
        token_index: int
        time_index: int
        score: float
        path: List[Point]

    init_state = BeamState(
        token_index=J,
        time_index=T,
        score=trellis[T, J],
        path=[Point(J, T, emission[T, blank_id].exp().item())],
    )

    beams = [init_state]

    while beams and beams[0].token_index > 0:
        next_beams = []
        for beam in beams:
            t, j = beam.time_index, beam.token_index
            if t <= 0:
                continue

            p_stay = emission[t - 1, blank_id]
            p_change = get_wildcard_emission(emission[t - 1], [tokens[j]], blank_id)[0]

            stay_score = trellis[t - 1, j]
            change_score = (
                trellis[t - 1, j - 1] if j > 0 else torch.tensor(float("-inf"))
            )

            if not math.isinf(stay_score):
                new_path = beam.path.copy()
                new_path.append(Point(j, t - 1, p_stay.exp().item()))
                next_beams.append(
                    BeamState(
                        token_index=j,
                        time_index=t - 1,
                        score=stay_score,
                        path=new_path,
                    )
                )

            if j > 0 and not math.isinf(change_score):
                new_path = beam.path.copy()
                new_path.append(Point(j - 1, t - 1, p_change.exp().item()))
                next_beams.append(
                    BeamState(
                        token_index=j - 1,
                        time_index=t - 1,
                        score=change_score,
                        path=new_path,
                    )
                )

        beams = sorted(next_beams, key=lambda x: x.score, reverse=True)[:beam_width]
        if not beams:
            break

    if not beams:
        return None

    best_beam = beams[0]
    t, j = best_beam.time_index, best_beam.token_index
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        best_beam.path.append(Point(j, t - 1, prob))
        t -= 1

    return best_beam.path[::-1]


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def align(
    text: str,
    waveform: torch.Tensor,
    model: torch.nn.Module,
    dictionary: dict,
    device: str,
):
    """Align a single text segment to a waveform.

    Args:
        text: The transcription text.
        waveform: Mono waveform tensor (1, T).
        model: Wav2Vec2ForCTC model.
        dictionary: Character to ID mapping.
        device: Torch device.
    """
    if not text:
        return []

    # Clean text: keep only characters in dictionary or use '*' for wildcards
    clean_char = []
    clean_cdx = []
    for cdx, char in enumerate(text):
        char_ = char.lower().replace(" ", "|")
        if char_ in dictionary:
            clean_char.append(char_)
            clean_cdx.append(cdx)
        else:
            clean_char.append("*")
            clean_cdx.append(cdx)

    text_clean = "".join(clean_char)
    tokens = [dictionary.get(c, -1) for c in text_clean]

    with torch.inference_mode():
        logits = model(waveform.to(device)).logits
        emission = torch.log_softmax(logits, dim=-1)[0].cpu().detach()

    blank_id = dictionary.get("[pad]", dictionary.get("<pad>", 0))
    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack_beam(trellis, emission, tokens, blank_id)

    if path is None:
        return []

    char_segments = merge_repeats(path, text_clean)
    duration = waveform.size(-1) / SAMPLE_RATE
    ratio = duration / (trellis.size(0) - 1)

    # Convert back to word-level segments
    words = []
    current_word_chars = []
    
    # Split text into words by space
    raw_words = text.split(" ")
    word_start_idx = 0
    
    # Simple word splitting logic based on character alignment
    for word in raw_words:
        if not word:
            continue
            
        word_len = len(word)
        word_chars = []
        
        # Find characters belonging to this word
        for i in range(word_start_idx, word_start_idx + word_len):
            if i < len(char_segments):
                word_chars.append(char_segments[i])
        
        if word_chars:
            start_s = word_chars[0].start * ratio
            end_s = word_chars[-1].end * ratio
            avg_score = sum(c.score for c in word_chars) / len(word_chars)
            words.append({
                "word": word,
                "start": start_s,
                "end": end_s,
                "score": avg_score
            })
        
        word_start_idx += word_len + 1 # +1 for space
        
    return words
