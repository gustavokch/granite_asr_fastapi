"""Tests for forced alignment using Wav2Vec2."""

import numpy as np
import pytest
import torch
from granite_asr.alignment import align


class MockModel:
    def __init__(self, vocab_size):
        self.logits = torch.randn(1, 100, vocab_size)

    def __call__(self, x):
        class Output:
            def __init__(self, logits):
                self.logits = logits

        return Output(self.logits)

    def to(self, device):
        return self


def test_align_simple():
    """Test alignment with mock data."""
    dictionary = {"|": 0, "a": 1, "b": 2, "c": 3, "[pad]": 4}
    model = MockModel(len(dictionary))
    device = "cpu"
    
    text = "abc"
    waveform = torch.randn(1, 16000) # 1s
    
    words = align(text, waveform, model, dictionary, device)
    
    assert len(words) == 1
    assert words[0]["word"] == "abc"
    assert "start" in words[0]
    assert "end" in words[0]


def test_align_multi_word():
    """Test alignment with multiple words."""
    dictionary = {"|": 0, "a": 1, "b": 2, "c": 3, "[pad]": 4}
    model = MockModel(len(dictionary))
    device = "cpu"
    
    text = "abc abc"
    waveform = torch.randn(1, 16000)
    
    words = align(text, waveform, model, dictionary, device)
    
    assert len(words) == 2
    assert words[0]["word"] == "abc"
    assert words[1]["word"] == "abc"
