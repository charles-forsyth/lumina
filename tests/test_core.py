import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from generate_gemini_image.core import ImageGenerator

@pytest.fixture
def mock_genai_client():
    with patch("generate_gemini_image.core.genai.Client") as mock:
        yield mock

def test_init(mock_genai_client):
    generator = ImageGenerator("test-project", "us-central1", "test-model")
    assert generator.project_id == "test-project"
    # Client is lazy loaded, so we access it to trigger init
    _ = generator.client
    mock_genai_client.assert_called_with(
        vertexai=True,
        project="test-project",
        location="us-central1"
    )

def test_generate_images(mock_genai_client, tmp_path):
    # Setup Client Mock
    mock_client_instance = MagicMock()
    mock_genai_client.return_value = mock_client_instance
    
    # Setup Response Mock
    mock_part_image = MagicMock()
    mock_part_image.text = None
    mock_part_image.inline_data = True
    mock_part_image.as_image.return_value = MagicMock() # Returns a PIL image mock
    
    mock_response = MagicMock()
    mock_response.parts = [mock_part_image]
    
    mock_client_instance.models.generate_content.return_value = mock_response

    generator = ImageGenerator("test-project", "us-central1", "test-model")
    
    output_dir = tmp_path / "output"
    files = generator.generate("test prompt", count=1, output_dir=output_dir)

    # Verify Call
    mock_client_instance.models.generate_content.assert_called_once()
    assert len(files) == 1
    assert files[0].parent == output_dir
    mock_part_image.as_image().save.assert_called_once()

def test_generate_multiple_images_loop(mock_genai_client, tmp_path):
    # This tests the loop in the generate method for count > 1
    
    # Setup Client Mock
    mock_client_instance = MagicMock()
    mock_genai_client.return_value = mock_client_instance
    
    # Setup Response Mock (Single image per call)
    mock_part = MagicMock()
    mock_part.text = None
    mock_part.inline_data = True
    mock_part.as_image.return_value = MagicMock()
    
    mock_response = MagicMock()
    mock_response.parts = [mock_part]
    
    mock_client_instance.models.generate_content.return_value = mock_response

    generator = ImageGenerator("test-project", "us-central1", "test-model")
    
    files = generator.generate("test prompt", count=2, output_dir=tmp_path)

    # Should have called generate_content twice
    assert mock_client_instance.models.generate_content.call_count == 2
    assert len(files) == 2
