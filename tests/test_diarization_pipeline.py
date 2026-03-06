from unittest.mock import patch, MagicMock
from granite_asr.diarization import DiarizationPipeline

@patch("granite_asr.diarization.Pipeline.from_pretrained")
def test_diarization_pipeline_init(mock_from_pretrained):
    # Setup mock
    mock_pipeline = MagicMock()
    mock_from_pretrained.return_value = mock_pipeline
    
    # Instantiate with token
    pipeline = DiarizationPipeline(model_id="test-model", token="test-token")
    
    # Verify mock was called with 'token'
    mock_from_pretrained.assert_called_once_with("test-model", token="test-token")
    
    # Verify to() was called if pipeline not None
    mock_pipeline.to.assert_called()
