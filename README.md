# KnowlegeLLM

A powerful AI-driven knowledge system built on MiniGPT-4, integrating visual and textual understanding with enhanced search capabilities and knowledge graph visualization.

## Features

- Visual and textual understanding using MiniGPT-4
- Enhanced search capabilities with SERPAPI integration
- Knowledge graph visualization for FAOSTAT data
- Interactive chat interface with basic and enhanced modes
- Image upload and processing capabilities

## Requirements

- Python 3.8+
- PyTorch
- Gradio
- OpenCV
- NetworkX
- Plotly
- SERPAPI
- python-dotenv

## Setup

1. Clone the repository:
```bash
git clone https://github.com/greatroboticslab/knowlegeLLM.git
cd knowlegeLLM
```

2. Create a `.env` file in the root directory and add your SERPAPI key:
```
SERP_API_KEY=your_api_key_here
```

3. Install the required dependencies:
```bash
pip install torch torchvision opencv-python gradio serpapi python-dotenv networkx pandas plotly
```

## Usage

Run the demo script:
```bash
python minigptfour/demo_v3.py
```

The interface will be available at `http://localhost:7860` by default.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 