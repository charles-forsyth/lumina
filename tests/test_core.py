from unittest.mock import MagicMock, patch

import pytest

from lumina.core import ImageGenerator


@pytest.fixture
def mock_genai_client():
    with patch("lumina.core.genai.Client") as mock:
        yield mock


def test_init(mock_genai_client):
    generator = ImageGenerator(
        model_name="test-model", project_id="test-project", location="us-central1"
    )
    assert generator.project_id == "test-project"
    # Client is lazy loaded
    _ = generator.client
    mock_genai_client.assert_called_with(
        vertexai=True, project="test-project", location="us-central1"
    )


def test_init_api_key(mock_genai_client):
    generator = ImageGenerator(model_name="test-model", api_key="fake-key")
    _ = generator.client
    mock_genai_client.assert_called_with(api_key="fake-key", vertexai=False)


def test_generate_images(mock_genai_client, tmp_path):
    # Setup Client Mock
    mock_client_instance = MagicMock()
    mock_genai_client.return_value = mock_client_instance

    # Setup Response Mock
    mock_part_image = MagicMock()
    mock_part_image.text = None
    mock_part_image.inline_data = True
    mock_part_image.as_image.return_value = MagicMock()

    mock_response = MagicMock()
    mock_response.parts = [mock_part_image]

    mock_client_instance.models.generate_content.return_value = mock_response

    generator = ImageGenerator(model_name="test-model", project_id="test-project")

    output_dir = tmp_path / "output"
    files = generator.generate("test prompt", count=1, output_dir=output_dir)

    assert len(files) == 1
    assert files[0].parent == output_dir
    mock_part_image.as_image().save.assert_called_once()


def test_generate_multiple_images_loop(mock_genai_client, tmp_path):
    mock_client_instance = MagicMock()
    mock_genai_client.return_value = mock_client_instance

    mock_part = MagicMock()
    mock_part.text = None
    mock_part.inline_data = True
    mock_part.as_image.return_value = MagicMock()

    mock_response = MagicMock()
    mock_response.parts = [mock_part]

    mock_client_instance.models.generate_content.return_value = mock_response

    generator = ImageGenerator(model_name="test-model", project_id="test-project")

    files = generator.generate("test prompt", count=2, output_dir=tmp_path)

    assert mock_client_instance.models.generate_content.call_count == 2
    assert len(files) == 2


def test_get_project_id_skips_gce_auth_when_api_key_set(monkeypatch):
    from lumina import cli

    monkeypatch.setattr(cli.settings, "project_id", None)
    with patch("lumina.cli.google.auth.default") as mock_auth_default:
        assert cli.get_project_id(None, has_api_key=True) is None
        mock_auth_default.assert_not_called()
