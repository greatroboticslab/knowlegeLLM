import argparse
import os
import random
from collections import defaultdict
import cv2
import re
import numpy as np
from PIL import Image
import torch
import html
import gradio as gr
from serpapi import GoogleSearch
from dotenv import load_dotenv

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat
from minigpt4.conversation.conversation import CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# Load environment variables
load_dotenv()

# SERPAPI Configuration
SERP_API_KEY = os.getenv('SERP_API_KEY')
if not SERP_API_KEY:
    print("Warning: SERP_API_KEY not found in environment variables. Enhanced search will not work.")
    print("Please create a .env file with SERP_API_KEY=your_api_key_here")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", default="eval_configs/minigptv2_eval.yaml", help="path to configuration file.")
parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
parser.add_argument("--options", default=None, help="additional options to override the configuration.")
args = parser.parse_args()

# Load configuration
cfg = Config(args.cfg_path)
if args.options is not None:
    cfg.update_with_str(args.options)

# Set device
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# Initialize model
model_config = cfg.model_cfg
model = registry.get_model_class(model_config.arch)(model_config)
model.load_checkpoint(cfg.model_cfg.ckpt)
model = model.to(device)
model.eval()

# Initialize visual processor
vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

# Initialize chat
chat = Chat(model, vis_processor, device=device)
print('Initialization Finished')

def fetch_serp_context(query):
    """Fetch additional context from SERPAPI."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": 3,
        "gl": "us",
        "hl": "en",
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        context = " ".join([result.get("snippet", "") for result in organic_results[:3]])
        return context
    except Exception as e:
        print(f"SERPAPI Error: {e}")
        return ""

def create_knowledge_graph():
    """Create knowledge graph visualization from FAOSTAT data."""
    try:
        print("Reading CSV files...")
        nodes_df = pd.read_csv('kg_nodes_faostat.csv')
        relationships_df = pd.read_csv('kg_relationships_faostat.csv')
        
        print(f"Loaded {len(nodes_df)} nodes and {len(relationships_df)} relationships")
        
        G = nx.DiGraph()
        
        # Add nodes and edges
        for _, row in nodes_df.iterrows():
            G.add_node(row['id'], label=row['label'], name=row['name'])
        
        for _, row in relationships_df.head(1000).iterrows():
            G.add_edge(row['start_id'], row['end_id'], type=row['type'])
        
        # Create visualization
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge trace
        edge_x, edge_y, edge_text = [], [], []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(edge[2]['type'])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines')
        
        # Create node trace
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{G.nodes[node]['label']}: {G.nodes[node]['name']}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=[G.degree(node) for node in G.nodes()],
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='FAOSTAT Knowledge Graph',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=800,
                           height=600
                       ))
        
        return fig
    except Exception as e:
        print(f"Error creating knowledge graph: {str(e)}")
        return go.Figure()

def process_chat_with_image(user_message, chatbot, chat_state, gr_img, img_list, is_enhanced=False):
    """Process chat with optional SERPAPI enhancement."""
    try:
        if gr_img is not None and isinstance(gr_img, Image.Image):
            # Process new image
            image_tensor = vis_processor(gr_img).to(device)
            img_list = [image_tensor]
            chat_state = CONV_VISION.copy()
            chat.upload_img(gr_img, chat_state, img_list)

        if not img_list:
            return chatbot + [[user_message, "Please upload an image first."]], chat_state, img_list

        # Get basic response
        chat.ask(user_message, chat_state)
        basic_response = chat.answer(conv=chat_state,
                                   img_list=img_list,
                                   temperature=0.7,
                                   max_new_tokens=500,
                                   max_length=2000)[0]

        if not is_enhanced:
            return chatbot + [[user_message, basic_response]], chat_state, img_list

        # Get enhanced response with SERPAPI
        serp_context = fetch_serp_context(basic_response)
        if serp_context:
            chat.ask(f"Add this context to your response and respond again: {serp_context}", chat_state)
            enhanced_response = chat.answer(conv=chat_state,
                                         img_list=img_list,
                                         temperature=0.7,
                                         max_new_tokens=500,
                                         max_length=2000)[0]
            return chatbot + [[user_message, enhanced_response]], chat_state, img_list

        return chatbot + [[user_message, basic_response]], chat_state, img_list

    except Exception as e:
        print(f"Error in chat processing: {str(e)}")
        return chatbot + [[user_message, f"Error: {str(e)}"]], chat_state, img_list

def reset_chat(chat_state, img_list):
    """Reset chat state and image list."""
    return [], None, "", CONV_VISION.copy(), []

# Create Gradio interface- gotta be careful with gradio compatiability issues
with gr.Blocks() as demo:
    gr.Markdown("# MiniGPT-v2 Demo with Knowledge Graph")
    
    # Knowledge Graph Section
    graph_plot = gr.Plot(label="Knowledge Graph", value=create_knowledge_graph())
    
    with gr.Row():
        # Image Upload Section
        with gr.Column(scale=1):
            image = gr.Image(type="pil", tool='sketch', brush_radius=20)
            temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature")
            clear = gr.Button("Reset")

        # Chat Section
        with gr.Column(scale=2):
            # Basic Chat
            gr.Markdown("### Basic Chat")
            basic_chatbot = gr.Chatbot()
            basic_chat_state = gr.State(CONV_VISION.copy())
            basic_img_list = gr.State([])
            
            with gr.Row():
                basic_input = gr.Textbox(placeholder="Enter message for basic chat...", scale=8)
                basic_send = gr.Button("Send", scale=1)

            # Enhanced Chat
            gr.Markdown("### Enhanced Chat (with SERPAPI)")
            enhanced_chatbot = gr.Chatbot()
            enhanced_chat_state = gr.State(CONV_VISION.copy())
            enhanced_img_list = gr.State([])
            
            with gr.Row():
                enhanced_input = gr.Textbox(placeholder="Enter message for enhanced chat...", scale=8)
                enhanced_send = gr.Button("Send", scale=1)

    # Event handlers
    basic_send.click(
        process_chat_with_image,
        inputs=[basic_input, basic_chatbot, basic_chat_state, image, basic_img_list],
        outputs=[basic_chatbot, basic_chat_state, basic_img_list]
    )

    enhanced_send.click(
        lambda *args: process_chat_with_image(*args, is_enhanced=True),
        inputs=[enhanced_input, enhanced_chatbot, enhanced_chat_state, image, enhanced_img_list],
        outputs=[enhanced_chatbot, enhanced_chat_state, enhanced_img_list]
    )

    clear.click(
        reset_chat,
        inputs=[basic_chat_state, basic_img_list],
        outputs=[basic_chatbot, image, basic_input, basic_chat_state, basic_img_list]
    ).then(
        reset_chat,
        inputs=[enhanced_chat_state, enhanced_img_list],
        outputs=[enhanced_chatbot, image, enhanced_input, enhanced_chat_state, enhanced_img_list]
    )

demo.queue()
demo.launch(share=True)
