"""
Main application entry point with CLI support.

This module provides the CLI interface for the Ultimate ONNX Converter.
Users can launch the TUI interface or use command-line commands.

Example:
    Launch TUI:
        $ onnx-converter tui
    
    Show version:
        $ onnx-converter version
    
    Convert model (future):
        $ onnx-converter convert model.pt -o model.onnx -f onnx
"""

import sys
import typer
from rich.console import Console
from textual.app import App
from .core.tui import AppState, MainMenuScreen

console = Console()
app = typer.Typer(
    help="Ultimate ONNX Converter - Terminal User Interface for ONNX model conversion, "
         "quantization, testing, profiling, and optimization"
)


class ONNXConverterApp(App):
    """Main Textual application."""
    
    def on_mount(self):
        """Initialize the app on mount."""
        app_state = AppState()
        self.push_screen(MainMenuScreen(app_state))


@app.command()
def tui():
    """Launch the Terminal User Interface.
    
    Starts the interactive TUI for ONNX model operations including:
    - Model conversion
    - Quantization with multiple methods
    - Performance testing
    - Model profiling  
    - Graph optimization
    """
    textual_app = ONNXConverterApp()
    textual_app.run()


@app.command()
def version():
    """Show version information.
    
    Displays the current version of Ultimate ONNX Converter.
    """
    from . import __version__
    console.print(f"[bold cyan]Ultimate ONNX Converter[/bold cyan] v{__version__}")


@app.command()
def convert(
    input_file: str = typer.Argument(..., help="Input model file path"),
    output_file: str = typer.Option(..., "--output", "-o", help="Output model file path"),
    output_format: str = typer.Option("onnx", "--format", "-f", help="Output format (onnx, pytorch, etc)"),
):
    """Convert model between formats.
    
    Args:
        input_file: Path to input model file
        output_file: Path for output model file  
        output_format: Target format (default: onnx)
        
    Example:
        onnx-converter convert model.pt -o model.onnx -f onnx
    """
    console.print("[yellow]⏳ Feature not yet implemented. Please use TUI: onnx-converter tui[/yellow]")


if __name__ == "__main__":
    app()
