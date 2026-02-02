"""
Ultimate ONNX Converter - Main Entry Point
Terminal User Interface for ONNX Model Conversion and Optimization
"""

__version__ = "0.1.0"
__author__ = "Ultimate ONNX Team"

from .core.tui import ONNXConverterTUI


def main():
    """Main entry point for the application."""
    app = ONNXConverterTUI()
    app.run()


if __name__ == "__main__":
    main()
