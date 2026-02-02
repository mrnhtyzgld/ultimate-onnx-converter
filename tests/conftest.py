"""
Configuration and fixtures for pytest.
"""

import pytest
from onnx_converter.core.tui import ONNXConverterTUI, AppState


@pytest.fixture
def app_state():
    """Fixture providing a fresh AppState instance."""
    return AppState()


@pytest.fixture
def tui_app():
    """Fixture providing a fresh ONNXConverterTUI instance."""
    return ONNXConverterTUI()


@pytest.fixture
def populated_state():
    """Fixture providing a populated AppState with sample data."""
    state = AppState()
    state.current_model = "sample_model.onnx"
    state.current_config = {
        "format": "onnx",
        "optimization": True,
        "validation": True
    }
    state.history = ["model_loaded", "conversion_started"]
    state.results = {
        "conversion": {
            "success": True,
            "duration": 5.2
        }
    }
    state.execution_provider = "cuda"
    return state


@pytest.fixture
def populated_tui():
    """Fixture providing a populated TUI instance."""
    tui = ONNXConverterTUI()
    tui.state.current_model = "sample_model.onnx"
    tui.state.execution_provider = "cuda"
    tui.state.current_config["format"] = "onnx"
    return tui


@pytest.fixture
def menu_navigation_sequence():
    """Fixture providing typical menu navigation sequences."""
    return {
        "conversion_flow": ["main", "conversion"],
        "quantization_flow": ["main", "quantization"],
        "testing_flow": ["main", "testing"],
        "settings_flow": ["main", "settings"],
        "complex_flow": ["main", "conversion", "quantization", "testing"]
    }


@pytest.fixture
def sample_configs():
    """Fixture providing sample configurations."""
    return {
        "pytorch_to_onnx": {
            "input_format": "pytorch",
            "output_format": "onnx",
            "optimization": True,
            "validation": True
        },
        "int8_quantization": {
            "method": "int8",
            "static": True,
            "calibration_data": "/path/to/calibration",
            "output_format": "onnx"
        },
        "fp16_quantization": {
            "method": "fp16",
            "dynamic": True,
            "output_format": "onnx"
        },
        "performance_test": {
            "iterations": 100,
            "warmup": 10,
            "batch_size": 1,
            "providers": ["cpu", "cuda"]
        }
    }
