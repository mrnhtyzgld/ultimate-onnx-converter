"""
Core TUI application using Textual framework.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from textual.app import ComposeResult, on
from textual.containers import Container, Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Static, Input, Label, Select
from textual.binding import Binding
from enum import Enum


class MenuItem(str, Enum):
    """Main menu items."""
    CONVERSION = "1"
    QUANTIZATION = "2"
    TESTING = "3"
    PROFILING = "4"
    OPTIMIZATION = "5"
    BATCH_OPS = "6"
    SETTINGS = "7"
    EXIT = "8"


@dataclass
class AppState:
    """Application state management."""
    current_model: Optional[str] = None
    current_config: Dict[str, Any] = field(default_factory=dict)
    history: list = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    execution_provider: str = "cpu"  # cpu, cuda, tensorrt


class Header(Static):
    """Application header."""
    
    def render(self) -> str:
        return "[bold cyan]ULTIMATE ONNX CONVERTER[/bold cyan]\n[dim]Terminal User Interface v0.1.0[/dim]"


class MainMenuScreen(Screen):
    """Main menu screen."""

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.menu_stack = ["main"]

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        
        with Vertical(id="main-menu"):
            yield Label("Select Operation:", id="menu-title")
            yield Button("Model Conversion", id="btn-conversion", variant="primary")
            yield Button("Model Quantization", id="btn-quantization", variant="primary")
            yield Button("Model Testing & Validation", id="btn-testing", variant="primary")
            yield Button("Model Profiling", id="btn-profiling", variant="primary")
            yield Button("Model Optimization", id="btn-optimization", variant="primary")
            yield Button("Batch Operations", id="btn-batch", variant="primary")
            yield Button("Settings", id="btn-settings", variant="primary")
            yield Button("Exit", id="btn-exit", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-conversion":
            self.app.push_screen(ConversionMenuScreen(self.app_state, self.menu_stack))
        elif button_id == "btn-quantization":
            self.app.push_screen(QuantizationMenuScreen(self.app_state, self.menu_stack))
        elif button_id == "btn-testing":
            self.app.push_screen(TestingMenuScreen(self.app_state, self.menu_stack))
        elif button_id == "btn-profiling":
            self.app.push_screen(ProfilingMenuScreen(self.app_state, self.menu_stack))
        elif button_id == "btn-optimization":
            self.app.push_screen(OptimizationMenuScreen(self.app_state, self.menu_stack))
        elif button_id == "btn-batch":
            self.app.push_screen(PlaceholderMenuScreen("Batch Operations", self.app_state, self.menu_stack))
        elif button_id == "btn-settings":
            self.app.push_screen(SettingsMenuScreen(self.app_state, self.menu_stack))
        elif button_id == "btn-exit":
            self.app.exit()


class ConversionMenuScreen(Screen):
    """Model Conversion menu screen."""

    def __init__(self, app_state: AppState, menu_stack: list):
        super().__init__()
        self.app_state = app_state
        self.menu_stack = menu_stack

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        
        with Vertical(id="submenu"):
            yield Label("[bold cyan]Model Conversion[/bold cyan]", id="menu-title")
            yield Label("Select input format:", id="submenu-subtitle")
            yield Button("PyTorch (.pt, .pth)", id="btn-pytorch", variant="primary")
            yield Button("TensorFlow (.pb, .h5, .ckpt)", id="btn-tensorflow", variant="primary")
            yield Button("ONNX (Optimize)", id="btn-onnx", variant="primary")
            yield Button("Back to Main Menu", id="btn-back", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id in ("btn-pytorch", "btn-tensorflow", "btn-onnx"):
            self.app.push_screen(PlaceholderMenuScreen("Model Conversion Details", self.app_state, self.menu_stack))
        elif button_id == "btn-back":
            self.app.pop_screen()


class QuantizationMenuScreen(Screen):
    """Model Quantization menu screen."""

    def __init__(self, app_state: AppState, menu_stack: list):
        super().__init__()
        self.app_state = app_state
        self.menu_stack = menu_stack

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        
        with Vertical(id="submenu"):
            yield Label("[bold cyan]Model Quantization[/bold cyan]", id="menu-title")
            yield Label("Select quantization method:", id="submenu-subtitle")
            yield Button("Integer (INT8)", id="btn-int8", variant="primary")
            yield Button("Half Precision (FP16)", id="btn-fp16", variant="primary")
            yield Button("Signed-Unsigned (U8S8/S8U8)", id="btn-su", variant="primary")
            yield Button("Unsigned (U8U8)", id="btn-uu", variant="primary")
            yield Button("Signed (S8S8)", id="btn-ss", variant="primary")
            yield Button("QDQ Format", id="btn-qdq", variant="primary")
            yield Button("Custom Configuration", id="btn-custom", variant="primary")
            yield Button("Back to Main Menu", id="btn-back", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id in ("btn-int8", "btn-fp16", "btn-su", "btn-uu", "btn-ss", "btn-qdq", "btn-custom"):
            self.app.push_screen(PlaceholderMenuScreen("Model Quantization", self.app_state, self.menu_stack))
        elif button_id == "btn-back":
            self.app.pop_screen()


class TestingMenuScreen(Screen):
    """Model Testing menu screen."""

    def __init__(self, app_state: AppState, menu_stack: list):
        super().__init__()
        self.app_state = app_state
        self.menu_stack = menu_stack

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        
        with Vertical(id="submenu"):
            yield Label("[bold cyan]Model Testing & Validation[/bold cyan]", id="menu-title")
            yield Label("Select test type:", id="submenu-subtitle")
            yield Button("Performance Benchmark", id="btn-bench", variant="primary")
            yield Button("Accuracy Verification", id="btn-accuracy", variant="primary")
            yield Button("Output Difference Analysis", id="btn-diff", variant="primary")
            yield Button("Stress Test", id="btn-stress", variant="primary")
            yield Button("Back to Main Menu", id="btn-back", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id in ("btn-bench", "btn-accuracy", "btn-diff", "btn-stress"):
            self.app.push_screen(PlaceholderMenuScreen("Model Testing", self.app_state, self.menu_stack))
        elif button_id == "btn-back":
            self.app.pop_screen()


class ProfilingMenuScreen(Screen):
    """Model Profiling menu screen."""

    def __init__(self, app_state: AppState, menu_stack: list):
        super().__init__()
        self.app_state = app_state
        self.menu_stack = menu_stack

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        
        with Vertical(id="submenu"):
            yield Label("[bold cyan]Model Profiling[/bold cyan]", id="menu-title")
            yield Label("Select profiling type:", id="submenu-subtitle")
            yield Button("Node Analysis", id="btn-node", variant="primary")
            yield Button("Memory Profiling", id="btn-memory", variant="primary")
            yield Button("FLOPs Calculation", id="btn-flops", variant="primary")
            yield Button("Bottleneck Analysis", id="btn-bottleneck", variant="primary")
            yield Button("Back to Main Menu", id="btn-back", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id in ("btn-node", "btn-memory", "btn-flops", "btn-bottleneck"):
            self.app.push_screen(PlaceholderMenuScreen("Model Profiling", self.app_state, self.menu_stack))
        elif button_id == "btn-back":
            self.app.pop_screen()


class OptimizationMenuScreen(Screen):
    """Model Optimization menu screen."""

    def __init__(self, app_state: AppState, menu_stack: list):
        super().__init__()
        self.app_state = app_state
        self.menu_stack = menu_stack

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        
        with Vertical(id="submenu"):
            yield Label("[bold cyan]Model Optimization[/bold cyan]", id="menu-title")
            yield Label("Select optimization type:", id="submenu-subtitle")
            yield Button("Graph Rewriting", id="btn-graph", variant="primary")
            yield Button("Operator Fusion", id="btn-fusion", variant="primary")
            yield Button("Graph Simplification", id="btn-simplify", variant="primary")
            yield Button("Constant Folding", id="btn-folding", variant="primary")
            yield Button("Back to Main Menu", id="btn-back", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id in ("btn-graph", "btn-fusion", "btn-simplify", "btn-folding"):
            self.app.push_screen(PlaceholderMenuScreen("Model Optimization", self.app_state, self.menu_stack))
        elif button_id == "btn-back":
            self.app.pop_screen()


class SettingsMenuScreen(Screen):
    """Settings menu screen."""

    def __init__(self, app_state: AppState, menu_stack: list):
        super().__init__()
        self.app_state = app_state
        self.menu_stack = menu_stack

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        
        with Vertical(id="submenu"):
            yield Label("[bold cyan]Settings[/bold cyan]", id="menu-title")
            yield Label(f"Current Provider: {self.app_state.execution_provider.upper()}", id="current-provider")
            yield Label("Select execution provider:", id="submenu-subtitle")
            yield Button("CPU", id="btn-cpu", variant="primary")
            yield Button("CUDA", id="btn-cuda", variant="primary")
            yield Button("TensorRT", id="btn-tensorrt", variant="primary")
            yield Button("Back to Main Menu", id="btn-back", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-cpu":
            self.app_state.execution_provider = "cpu"
            self.query_one("#current-provider", Label).update(f"Current Provider: CPU")
        elif button_id == "btn-cuda":
            self.app_state.execution_provider = "cuda"
            self.query_one("#current-provider", Label).update(f"Current Provider: CUDA")
        elif button_id == "btn-tensorrt":
            self.app_state.execution_provider = "tensorrt"
            self.query_one("#current-provider", Label).update(f"Current Provider: TENSORRT")
        elif button_id == "btn-back":
            self.app.pop_screen()


class PlaceholderMenuScreen(Screen):
    """Placeholder screen for not-yet-implemented features."""

    def __init__(self, title: str, app_state: AppState, menu_stack: list):
        super().__init__()
        self.title = title
        self.app_state = app_state
        self.menu_stack = menu_stack

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        
        with Vertical(id="placeholder"):
            yield Label(f"[yellow]⚠ {self.title}[/yellow]", id="menu-title")
            yield Label("[dim]This feature is under development.[/dim]", id="placeholder-msg")
            yield Button("Back to Menu", id="btn-back", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-back":
            self.app.pop_screen()


class ONNXConverterTUI:
    """Main TUI Application - compatibility wrapper."""
    
    def __init__(self):
        """Initialize TUI state."""
        self.state = AppState()
        self.running = True
        self.menu_stack = ["main"]
        # Backward compatibility properties
        self.console = None


# Compatibility aliases for tests
def create_tui() -> "ONNXConverterTUI":
    """Create a new TUI instance for testing."""
    return ONNXConverterTUI()
