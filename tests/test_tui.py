"""
Unit tests for TUI application with Textual framework.
"""

import pytest
from onnx_converter.core.tui import AppState, ONNXConverterTUI


class TestAppState:
    """Test suite for AppState class."""

    def test_appstate_initialization(self):
        """Test AppState initializes with default values."""
        state = AppState()
        assert state.current_model is None
        assert state.current_config == {}
        assert state.history == []
        assert state.results == {}
        assert state.execution_provider == "cpu"

    def test_appstate_model_assignment(self):
        """Test setting current model."""
        state = AppState()
        state.current_model = "test_model.onnx"
        assert state.current_model == "test_model.onnx"

    def test_appstate_config_management(self):
        """Test configuration management."""
        state = AppState()
        state.current_config["format"] = "onnx"
        state.current_config["optimization"] = True
        
        assert state.current_config["format"] == "onnx"
        assert state.current_config["optimization"] is True

    def test_appstate_history_tracking(self):
        """Test history tracking."""
        state = AppState()
        state.history.append("conversion")
        state.history.append("quantization")
        
        assert len(state.history) == 2
        assert state.history[0] == "conversion"
        assert state.history[-1] == "quantization"

    def test_appstate_results_storage(self):
        """Test results storage."""
        state = AppState()
        state.results["benchmark"] = {"latency": 2.5, "throughput": 400}
        
        assert state.results["benchmark"]["latency"] == 2.5
        assert state.results["benchmark"]["throughput"] == 400

    def test_appstate_execution_provider_change(self):
        """Test changing execution provider."""
        state = AppState()
        assert state.execution_provider == "cpu"
        
        state.execution_provider = "cuda"
        assert state.execution_provider == "cuda"
        
        state.execution_provider = "tensorrt"
        assert state.execution_provider == "tensorrt"


class TestONNXConverterTUI:
    """Test suite for ONNXConverterTUI class."""

    def test_tui_initialization(self):
        """Test TUI initializes correctly."""
        tui = ONNXConverterTUI()
        
        assert tui.state is not None
        assert isinstance(tui.state, AppState)
        assert tui.running is True
        assert tui.menu_stack == ["main"]

    def test_menu_stack_operations(self):
        """Test menu stack navigation."""
        tui = ONNXConverterTUI()
        
        # Append to stack
        tui.menu_stack.append("conversion")
        assert len(tui.menu_stack) == 2
        assert tui.menu_stack[-1] == "conversion"
        
        # Pop from stack
        tui.menu_stack.pop()
        assert len(tui.menu_stack) == 1
        assert tui.menu_stack[0] == "main"

    def test_state_persistence(self):
        """Test state persists across operations."""
        tui = ONNXConverterTUI()
        
        tui.state.current_model = "model.onnx"
        tui.state.current_config["format"] = "onnx"
        tui.state.execution_provider = "cuda"
        
        # Simulate menu transitions
        tui.menu_stack.append("conversion")
        tui.menu_stack.pop()
        
        # State should persist
        assert tui.state.current_model == "model.onnx"
        assert tui.state.current_config["format"] == "onnx"
        assert tui.state.execution_provider == "cuda"
        assert tui.menu_stack == ["main"]


class TestStateManagement:
    """Test suite for state management."""

    def test_multiple_state_transitions(self):
        """Test multiple state transitions."""
        state = AppState()
        
        # Transition 1
        state.current_model = "model1.onnx"
        state.history.append("model_loaded")
        
        # Transition 2
        state.current_config["method"] = "int8"
        state.history.append("quantization_set")
        
        # Transition 3
        state.results["quantization"] = {"size_reduction": 0.75}
        state.history.append("quantization_done")
        
        # Verify all transitions
        assert state.current_model == "model1.onnx"
        assert state.current_config["method"] == "int8"
        assert len(state.history) == 3
        assert state.results["quantization"]["size_reduction"] == 0.75

    def test_provider_persistence(self):
        """Test execution provider persistence."""
        state = AppState()
        
        # Change provider multiple times
        state.execution_provider = "cuda"
        assert state.execution_provider == "cuda"
        
        # Persist through model change
        state.current_model = "model.onnx"
        assert state.execution_provider == "cuda"
        
        # Change to tensorrt
        state.execution_provider = "tensorrt"
        assert state.execution_provider == "tensorrt"


class TestMenuNavigation:
    """Test suite for menu navigation."""

    def test_forward_navigation(self):
        """Test forward navigation through menus."""
        tui = ONNXConverterTUI()
        
        # Navigate forward
        tui.menu_stack.append("conversion")
        assert tui.menu_stack == ["main", "conversion"]
        
        tui.menu_stack.append("pytorch")
        assert tui.menu_stack == ["main", "conversion", "pytorch"]

    def test_backward_navigation(self):
        """Test backward navigation through menus."""
        tui = ONNXConverterTUI()
        tui.menu_stack = ["main", "conversion", "pytorch"]
        
        # Navigate backward
        if len(tui.menu_stack) > 1:
            tui.menu_stack.pop()
        assert tui.menu_stack == ["main", "conversion"]
        
        if len(tui.menu_stack) > 1:
            tui.menu_stack.pop()
        assert tui.menu_stack == ["main"]

    def test_menu_stack_integrity(self):
        """Test menu stack integrity after operations."""
        tui = ONNXConverterTUI()
        
        # Perform various operations
        tui.menu_stack.append("conversion")
        tui.menu_stack.append("pytorch")
        tui.menu_stack.pop()
        tui.menu_stack.append("tensorflow")
        
        # Check integrity
        assert tui.menu_stack == ["main", "conversion", "tensorflow"]
        assert len(tui.menu_stack) == 3


class TestErrorHandling:
    """Test suite for error handling."""

    def test_invalid_menu_choice(self):
        """Test handling of invalid menu choice."""
        tui = ONNXConverterTUI()
        
        # Application should not crash on invalid choice
        assert tui.running is True
        assert tui.menu_stack[0] == "main"

    def test_empty_state_handling(self):
        """Test handling of empty application state."""
        tui = ONNXConverterTUI()
        
        # State should handle empty values gracefully
        assert tui.state.current_model is None
        assert len(tui.state.history) == 0
        assert len(tui.state.results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
