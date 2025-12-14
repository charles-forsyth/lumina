from unittest.mock import MagicMock, patch

import pytest

from generate_gemini_image.core import ImageGenerator


@pytest.fixture
def mock_vertexai():
    with patch("generate_gemini_image.core.vertexai") as mock:
        yield mock


@pytest.fixture
def mock_image_model():
    with patch("generate_gemini_image.core.ImageGenerationModel") as mock:
        yield mock


def test_init(mock_vertexai, mock_image_model):
    generator = ImageGenerator("test-project", "us-central1", "test-model")
    mock_vertexai.init.assert_called_with(
        project="test-project", location="us-central1"
    )
    assert generator.project_id == "test-project"


def test_generate_images(mock_vertexai, mock_image_model, tmp_path):
    # Setup Mock
    mock_model_instance = MagicMock()
    mock_image_model.from_pretrained.return_value = mock_model_instance
    
    # Mock Response
    mock_image_response = MagicMock()
    mock_image_response.save = MagicMock()
    mock_model_instance.generate_images.return_value = [mock_image_response]

    generator = ImageGenerator("test-project", "us-central1", "test-model")
    
    output_dir = tmp_path / "output"
    files = generator.generate("test prompt", count=1, output_dir=output_dir)

    # Verify Call
    mock_model_instance.generate_images.assert_called_once()
    assert len(files) == 1
    mock_image_response.save.assert_called_once()
    assert files[0].parent == output_dir


def test_generate_multiple_images(mock_vertexai, mock_image_model, tmp_path):
    # Setup Mock
    mock_model_instance = MagicMock()
    mock_image_model.from_pretrained.return_value = mock_model_instance
    
    # Mock Response (2 images)
    mock_img1 = MagicMock()
    mock_img2 = MagicMock()
    mock_model_instance.generate_images.return_value = [mock_img1, mock_img2]

    generator = ImageGenerator("test-project", "us-central1", "test-model")
    
    files = generator.generate("test prompt", count=2, output_dir=tmp_path)

    assert len(files) == 2
    assert files[0] != files[1]  # Unique filenames