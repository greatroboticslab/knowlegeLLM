# KnowlegeLLM
(work in progress - more features coming soon!)

ai-driven knowledge sys built on MiniGPT-4, integrating visual+textual understanding w/ enhanced search capabilities & knowledge graph viz.

## features (so far)

- visual+textual understanding using minigpt4
- enhanced search w/ SERPAPI integration
- knowledge graph viz for FAOSTAT data
- interactive chat interface (basic + enhanced modes)
- img upload & processing capabilities

## requirements
- python 3.8+
- pytorch
- gradio
- opencv
- networkx
- plotly
- serpapi
- python-dotenv

## setup

1. clone repo:
```bash
git clone https://github.com/greatroboticslab/knowlegeLLM.git
cd knowlegeLLM
```

2. create `.env` file in root dir & add serpapi key:
```
SERP_API_KEY=your_api_key_here
```

3. install deps:
```bash
pip install torch torchvision opencv-python gradio serpapi python-dotenv networkx pandas plotly
```

## usage

run demo:
```bash
python minigptfour/demo_v3.py
```

interface -> http://localhost:7860 (default)

## incoming/todo
- knowledge graph data files:
  - kg_nodes_faostat.csv (node data for graph)
  - kg_relationships_faostat.csv (relationship data)
- model checkpoints & weights
- example images & outputs
- more detailed setup instructions
- proper error handling
- better documentation
- testing suite
- contribution guidelines
- performance optimizations
- containerization
- ci/cd pipeline setup
- api documentation
- more examples
- benchmarks

## known issues
- need to add data files
- some paths might need fixing
- documentation needs work
- might have compatibility issues w/ different pytorch versions
- knowledge graph visualization needs optimization for large datasets
- error handling could be better
- need more examples

## license

mit license (for now)

## contributing

contributions welcome! just submit PR. guidelines coming soon...

note: this is alpha stage software, expect frequent updates & changes 