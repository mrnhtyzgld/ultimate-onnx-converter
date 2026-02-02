# Ultimate ONNX Converter

Terminal tool to convert, quantize, test, profile, and optimize ONNX models.

Features

- Model conversion: Convert between PyTorch, TensorFlow, and ONNX formats
- Model quantization: Support for INT8, FP16, QDQ and multiple quantization modes
- Model testing: Performance benchmarking and accuracy checks
- Model profiling: Node-level analysis and memory usage
- Model optimization: Graph optimization, operator fusion, and constant folding

Status

Early development (v0.1.0). Basic TUI framework and menu structure are implemented.

Installation

Prerequisites

- Python 3.8+
- pip

Setup

```bash
git clone https://github.com/mrnhtyzgld/ultimate-onnx-converter.git
cd ultimate-onnx-converter
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -e "[dev]"
```

Optional extras

```bash
pip install -e "[gpu]"
pip install -e "[viz]"
```

Usage

Launch the interactive TUI:

```bash
onnx-converter tui
# or
python -m onnx_converter.main tui
```

Project layout

```
src/onnx_converter/
├── main.py          # CLI entry point
├── core/tui.py      # Textual TUI application
├── modules/         # converter, quantizer, tester, profiler, optimizer
└── utils/           # helpers, validators, logging

tests/               # unit and integration tests
```

Running tests

```bash
pytest -q
pytest --cov=onnx_converter
```

Configuration

Environment variables may be used for defaults, for example `ONNX_EXECUTION_PROVIDER` and `ONNX_OUTPUT_DIR`.

Contributing

Fork the repository, create a feature branch, and open a pull request.

License

MIT

Notes

This project is in active development. The interactive TUI and tests are the primary focus.
