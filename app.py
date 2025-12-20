"""Streamlit UI for LangGraph RAG Agent.

A professional interface for interacting with the LangGraph agent,
managing documents, and monitoring RAG ingestion.

Features:
- Aceternity-style meteor animation effect for chat screen
- Moon background for document processing sidebar
"""

import asyncio
import base64
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the LangGraph agent
from react_agent import graph
from react_agent.context import Context

# Constants
DATA_DIR = Path(__file__).parent / "src" / "data"
INGESTION_SCRIPT = Path(__file__).parent / "src" / "rag" / "ingestion.py"
TRACKING_FILE = Path(__file__).parent / "processed_documents.json"
ASSETS_DIR = Path(__file__).parent / "assets"
MOON_IMAGE_PATH = ASSETS_DIR / "moon.png"
MAX_FILE_SIZE_MB = 3
MAX_FILES_UPLOAD = 2
QDRANT_COLLECTION_NAME = "document_chunks"

# Page config
st.set_page_config(
    page_title="LangGraph RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Suggestion prompts for first-time users
SUGGESTIONS = {
    "📊 Analyze my documents": "Can you analyze all the documents I've uploaded and give me a summary?",
    "🔍 Find specific information": "Help me find information about [topic] in my documents",
    "💡 What can you do?": "What are your capabilities? What can you help me with?",
    "🌤️ Check the weather": "What's the weather like today?",
}


def get_moon_image_base64():
    """Load moon image and convert to base64 for CSS background."""
    try:
        if MOON_IMAGE_PATH.exists():
            with open(MOON_IMAGE_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception:
        pass
    return None


# Get moon image base64
MOON_BASE64 = get_moon_image_base64()

# Build sidebar background CSS based on whether moon image exists
SIDEBAR_BG_CSS = ""
if MOON_BASE64:
    SIDEBAR_BG_CSS = f"""
    /* Main sidebar container */
    [data-testid="stSidebar"] {{
        background-image: url('data:image/png;base64,{MOON_BASE64}') !important;
        background-size: cover !important;
        background-position: center center !important;
        background-repeat: no-repeat !important;
        background-attachment: local !important;
    }}
    
    /* Override ALL inner sidebar backgrounds */
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div,
    [data-testid="stSidebar"] > div > div > div,
    [data-testid="stSidebar"] [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] section > div {{
        background: transparent !important;
        background-color: transparent !important;
    }}
    
    /* Add semi-transparent overlay for readability */
    [data-testid="stSidebar"]::before {{
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        background: rgba(10, 10, 15, 0.7) !important;
        backdrop-filter: blur(3px) !important;
        z-index: 0 !important;
        pointer-events: none !important;
    }}
    
    /* Ensure content is above the overlay */
    [data-testid="stSidebar"] > div {{
        position: relative !important;
        z-index: 1 !important;
    }}
    
    /* Style sidebar text for better contrast */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: #fafafa !important;
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: #fafafa !important;
    }}
    """
else:
    SIDEBAR_BG_CSS = """
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1a 100%) !important;
    }
    
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] section > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    """

# Professional Dark Mode CSS - UI Styling with NEW COLOR PALETTE
DARK_MODE_CSS = f"""
<style>
    /* ============================================
       PROFESSIONAL DARK THEME UI - NEW PALETTE
       ============================================ */
    
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    :root {{
        --primary-color: rgb(191, 9, 47);
        --secondary-color: rgb(19, 36, 64);
        --accent-blue: rgb(22, 71, 106);
        --accent-teal: rgb(59, 151, 151);
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --bg-main: #0a0a0f;
        --bg-secondary: #0f0f1a;
        --bg-card: rgba(20, 20, 35, 0.8);
        --text-primary: #fafafa;
        --text-secondary: #94a3b8;
        --border-color: rgba(191, 9, 47, 0.2);
        --glow-color: rgba(191, 9, 47, 0.4);
    }}
    
    /* Main app background - deep space black */
    .main {{
        background: radial-gradient(ellipse at top, #0f0f23 0%, #0a0a0f 50%, #050507 100%);
        min-height: 100vh;
    }}
    
    .stApp {{
        background: radial-gradient(ellipse at top, #0f0f23 0%, #0a0a0f 50%, #050507 100%);
    }}
    
    /* ============================================
       SIDEBAR STYLING - Moon Background
       ============================================ */
    {SIDEBAR_BG_CSS}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: var(--text-primary);
    }}
    
    /* ============================================
       HEADER STYLING - NEW GRADIENT
       ============================================ */
    .main-header {{
        background: linear-gradient(135deg, 
            rgba(191, 9, 47, 0.9) 0%, 
            rgba(22, 71, 106, 0.9) 50%,
            rgba(59, 151, 151, 0.9) 100%
        );
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 
            0 4px 20px rgba(191, 9, 47, 0.3),
            0 0 40px rgba(22, 71, 106, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        z-index: 2;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }}
    
    .main-header h1 {{
        color: white;
        margin: 0;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }}
    
    .main-header p {{
        color: rgba(255, 255, 255, 0.9);
        margin: 0.75rem 0 0 0;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        font-weight: 500;
    }}
    
    /* ============================================
       CHAT CONTAINER & MESSAGES
       ============================================ */
    .chat-container {{
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem;
        position: relative;
        z-index: 2;
    }}
    
    [data-testid="stChatMessage"] {{
        background: rgba(20, 20, 35, 0.85);
        border: 1px solid rgba(22, 71, 106, 0.15);
        border-radius: 16px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        backdrop-filter: blur(12px);
        position: relative;
        z-index: 2;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }}
    
    [data-testid="stChatMessage"]:hover {{
        border-color: rgba(59, 151, 151, 0.3);
        box-shadow: 
            0 8px 24px rgba(22, 71, 106, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }}
    
    /* ============================================
       WELCOME SCREEN
       ============================================ */
    .welcome-container {{
        text-align: center;
        padding: 4rem 2rem;
        max-width: 800px;
        margin: 0 auto;
        position: relative;
        z-index: 2;
    }}
    
    .welcome-icon {{
        font-size: 5rem;
        margin-bottom: 1.5rem;
        line-height: 1;
        animation: float 4s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(191, 9, 47, 0.5));
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0) rotate(0deg); }}
        25% {{ transform: translateY(-8px) rotate(-2deg); }}
        75% {{ transform: translateY(-4px) rotate(2deg); }}
    }}
    
    .welcome-title {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #fff 0%, rgba(59, 151, 151, 0.8) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .welcome-subtitle {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.2rem;
        color: var(--text-secondary);
        margin-bottom: 2.5rem;
        font-weight: 500;
    }}
    
    /* ============================================
       SUGGESTION PILLS
       ============================================ */
    .suggestion-container {{
        text-align: center;
        padding: 2rem 0;
        position: relative;
        z-index: 2;
    }}
    
    .suggestion-title {{
        color: var(--text-secondary);
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }}
    
    [data-baseweb="tag"] {{
        position: relative;
        z-index: 2;
        background: rgba(22, 71, 106, 0.1) !important;
        border: 1px solid rgba(59, 151, 151, 0.3) !important;
        transition: all 0.3s ease !important;
    }}
    
    [data-baseweb="tag"]:hover {{
        background: rgba(22, 71, 106, 0.2) !important;
        border-color: rgba(59, 151, 151, 0.5) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 151, 151, 0.2);
    }}
    
    /* ============================================
       FILE CARD STYLING
       ============================================ */
    .file-card {{
        background: rgba(20, 20, 35, 0.9);
        border: 1px solid rgba(22, 71, 106, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(8px);
    }}
    
    .file-card:hover {{
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.3),
            0 0 20px rgba(59, 151, 151, 0.15);
        transform: translateY(-3px);
        border-color: rgba(59, 151, 151, 0.4);
    }}
    
    .file-name {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.95rem;
    }}
    
    .file-meta {{
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-secondary);
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }}
    
    /* ============================================
       STATUS BADGES
       ============================================ */
    .status-badge {{
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    
    .status-success {{
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
    }}
    
    .status-warning {{
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.2);
    }}
    
    .status-info {{
        background: rgba(59, 151, 151, 0.15);
        color: rgb(59, 151, 151);
        border: 1px solid rgba(59, 151, 151, 0.3);
        box-shadow: 0 0 12px rgba(59, 151, 151, 0.2);
    }}
    
    /* ============================================
       TERMINAL OUTPUT
       ============================================ */
    .terminal-output {{
        background: #0d0d12;
        color: #10b981;
        font-family: 'JetBrains Mono', monospace;
        padding: 1.25rem;
        border-radius: 12px;
        font-size: 0.85rem;
        max-height: 350px;
        overflow-y: auto;
        margin: 1rem 0;
        box-shadow: 
            inset 0 2px 8px rgba(0, 0, 0, 0.5),
            0 0 20px rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }}
    
    /* ============================================
       STATS CARDS
       ============================================ */
    .stats-card {{
        background: rgba(20, 20, 35, 0.9);
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        border-left: 4px solid rgb(191, 9, 47);
        border: 1px solid rgba(22, 71, 106, 0.2);
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }}
    
    .stats-card:hover {{
        transform: translateY(-2px);
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.3),
            0 0 20px rgba(59, 151, 151, 0.15);
    }}
    
    .stats-number {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, rgb(191, 9, 47) 0%, rgb(59, 151, 151) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .stats-label {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 0.25rem;
        font-weight: 500;
    }}
    
    /* ============================================
       SESSION INFO
       ============================================ */
    .session-info {{
        background: rgba(22, 71, 106, 0.08);
        border: 1px solid rgba(59, 151, 151, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.85rem;
        color: rgba(59, 151, 151, 0.9);
        backdrop-filter: blur(4px);
    }}
    
    .session-info code {{
        font-family: 'JetBrains Mono', monospace;
        background: rgba(0, 0, 0, 0.3);
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
    }}
    
    /* ============================================
       BUTTONS
       ============================================ */
    .stButton > button {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(22, 71, 106, 0.3);
        position: relative;
        z-index: 2;
        background: rgba(22, 71, 106, 0.1);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 
            0 8px 20px rgba(191, 9, 47, 0.25),
            0 0 30px rgba(59, 151, 151, 0.15);
        border-color: rgba(59, 151, 151, 0.5);
        background: rgba(22, 71, 106, 0.2);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* ============================================
       INPUT STYLING
       ============================================ */
    [data-testid="stChatInput"] {{
        border-radius: 14px;
        border: 1px solid rgba(22, 71, 106, 0.2);
        background: rgba(20, 20, 35, 0.9);
        backdrop-filter: blur(12px);
        position: relative;
        z-index: 2;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }}
    
    [data-testid="stChatInput"]:focus-within {{
        border-color: rgba(59, 151, 151, 0.5);
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.2),
            0 0 20px rgba(59, 151, 151, 0.15);
    }}
    
    /* ============================================
       EXPANDER STYLING
       ============================================ */
    .streamlit-expanderHeader {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        background: rgba(20, 20, 35, 0.8);
        border-radius: 10px;
        border: 1px solid rgba(22, 71, 106, 0.15);
    }}
    
    /* ============================================
       SCROLLBAR STYLING
       ============================================ */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(20, 20, 35, 0.5);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, rgb(191, 9, 47) 0%, rgb(59, 151, 151) 100%);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, rgb(22, 71, 106) 0%, rgb(59, 151, 151) 100%);
    }}
    
    /* ============================================
       MAIN CONTENT Z-INDEX
       ============================================ */
    .main > .block-container {{
        position: relative;
        z-index: 1;
    }}
    
    /* Ensure all interactive elements are above meteors */
    .stMarkdown, .stTextInput, .stSelectbox, .stMultiSelect, 
    .stSlider, .stCheckbox, .stRadio, .stFileUploader {{
        position: relative;
        z-index: 2;
    }}
</style>
"""

# Meteors using pure CSS - Updated with new colors
METEORS_HTML = """
<div class="meteors-bg-container">
    <div class="ambient-glow glow-1">&nbsp;</div>
    <div class="ambient-glow glow-2">&nbsp;</div>
    <div class="ambient-glow glow-3">&nbsp;</div>
    <div class="meteor-effect m1">&nbsp;</div>
    <div class="meteor-effect premium m2">&nbsp;</div>
    <div class="meteor-effect teal m3">&nbsp;</div>
    <div class="meteor-effect m4">&nbsp;</div>
    <div class="meteor-effect premium m5">&nbsp;</div>
    <div class="meteor-effect m6">&nbsp;</div>
    <div class="meteor-effect teal m7">&nbsp;</div>
    <div class="meteor-effect m8">&nbsp;</div>
    <div class="meteor-effect premium m9">&nbsp;</div>
    <div class="meteor-effect m10">&nbsp;</div>
    <div class="meteor-effect m11">&nbsp;</div>
    <div class="meteor-effect teal m12">&nbsp;</div>
    <div class="meteor-effect premium m13">&nbsp;</div>
    <div class="meteor-effect m14">&nbsp;</div>
    <div class="meteor-effect m15">&nbsp;</div>
    <div class="meteor-effect teal m16">&nbsp;</div>
    <div class="meteor-effect m17">&nbsp;</div>
    <div class="meteor-effect premium m18">&nbsp;</div>
    <div class="meteor-effect m19">&nbsp;</div>
    <div class="meteor-effect teal m20">&nbsp;</div>
</div>

<style>
    /* ============================================
       METEORS BACKGROUND CONTAINER
       ============================================ */
    .meteors-bg-container {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        overflow: hidden !important;
        pointer-events: none !important;
        z-index: 0 !important;
    }
    
    /* ============================================
       AMBIENT GLOW EFFECTS - NEW COLORS
       ============================================ */
    .ambient-glow {
        position: absolute !important;
        border-radius: 50% !important;
        filter: blur(80px) !important;
        opacity: 0.2 !important;
        pointer-events: none !important;
        font-size: 0 !important;
        line-height: 0 !important;
    }
    
    .glow-1 {
        width: 400px !important;
        height: 400px !important;
        background: radial-gradient(circle, rgb(191, 9, 47) 0%, transparent 70%) !important;
        top: -100px !important;
        right: 10% !important;
        animation: pulse-glow 8s ease-in-out infinite !important;
    }
    
    .glow-2 {
        width: 300px !important;
        height: 300px !important;
        background: radial-gradient(circle, rgb(22, 71, 106) 0%, transparent 70%) !important;
        bottom: 10% !important;
        left: 5% !important;
        animation: pulse-glow 10s ease-in-out infinite reverse !important;
    }
    
    .glow-3 {
        width: 250px !important;
        height: 250px !important;
        background: radial-gradient(circle, rgb(59, 151, 151) 0%, transparent 70%) !important;
        top: 40% !important;
        right: 20% !important;
        animation: pulse-glow 12s ease-in-out infinite !important;
        opacity: 0.15 !important;
    }
    
    @keyframes pulse-glow {
        0%, 100% { opacity: 0.2; transform: scale(1); }
        50% { opacity: 0.35; transform: scale(1.1); }
    }
    
    /* ============================================
       METEOR EFFECT - Updated Colors
       ============================================ */
    .meteor-effect {
        position: absolute !important;
        top: -5px !important;
        width: 3px !important;
        height: 3px !important;
        border-radius: 50% !important;
        background: #fff !important;
        box-shadow: 
            0 0 0 1px rgba(255, 255, 255, 0.1),
            0 0 6px 2px rgba(191, 9, 47, 0.8),
            0 0 15px 3px rgba(22, 71, 106, 0.5) !important;
        transform: rotate(215deg) !important;
        animation: meteor-fall linear infinite !important;
        font-size: 0 !important;
        line-height: 0 !important;
        color: transparent !important;
    }
    
    /* Meteor gradient tail using box-shadow */
    .meteor-effect::before {
        content: '' !important;
        position: absolute !important;
        top: 50% !important;
        right: -2px !important;
        width: 60px !important;
        height: 2px !important;
        transform: translateY(-50%) !important;
        background: linear-gradient(90deg, 
            rgba(255, 255, 255, 0.9) 0%,
            rgba(191, 9, 47, 0.6) 30%,
            rgba(22, 71, 106, 0.3) 60%,
            transparent 100%
        ) !important;
        border-radius: 100% !important;
    }
    
    /* Premium meteor - longer brighter tail */
    .meteor-effect.premium::before {
        width: 100px !important;
        background: linear-gradient(90deg, 
            rgba(255, 255, 255, 1) 0%,
            rgba(59, 151, 151, 0.8) 25%,
            rgba(191, 9, 47, 0.5) 50%,
            transparent 100%
        ) !important;
    }
    
    /* Teal accent meteor */
    .meteor-effect.teal {
        box-shadow: 
            0 0 0 1px rgba(255, 255, 255, 0.1),
            0 0 6px 2px rgba(59, 151, 151, 0.8),
            0 0 15px 3px rgba(59, 151, 151, 0.5) !important;
    }
    
    .meteor-effect.teal::before {
        background: linear-gradient(90deg, 
            rgba(59, 151, 151, 1) 0%,
            rgba(59, 151, 151, 0.6) 40%,
            transparent 100%
        ) !important;
    }
    
    /* ============================================
       METEOR ANIMATION
       ============================================ */
    @keyframes meteor-fall {
        0% {
            opacity: 1;
            transform: rotate(215deg) translateX(0);
        }
        70% {
            opacity: 1;
        }
        100% {
            opacity: 0;
            transform: rotate(215deg) translateX(-600px);
        }
    }
    
    /* Individual meteor positions and timings */
    .m1 { left: 5% !important; animation-duration: 2.5s !important; animation-delay: 0s !important; }
    .m2 { left: 12% !important; animation-duration: 4s !important; animation-delay: 0.3s !important; }
    .m3 { left: 20% !important; animation-duration: 3s !important; animation-delay: 0.7s !important; }
    .m4 { left: 28% !important; animation-duration: 4.5s !important; animation-delay: 1.1s !important; }
    .m5 { left: 35% !important; animation-duration: 3.2s !important; animation-delay: 0.4s !important; }
    .m6 { left: 42% !important; animation-duration: 3.8s !important; animation-delay: 1.4s !important; }
    .m7 { left: 50% !important; animation-duration: 3s !important; animation-delay: 0.2s !important; }
    .m8 { left: 58% !important; animation-duration: 4.2s !important; animation-delay: 0.8s !important; }
    .m9 { left: 65% !important; animation-duration: 2.8s !important; animation-delay: 1.6s !important; }
    .m10 { left: 72% !important; animation-duration: 3.5s !important; animation-delay: 0.5s !important; }
    .m11 { left: 80% !important; animation-duration: 4.8s !important; animation-delay: 1.9s !important; }
    .m12 { left: 88% !important; animation-duration: 3.3s !important; animation-delay: 1.0s !important; }
    .m13 { left: 95% !important; animation-duration: 4s !important; animation-delay: 2.2s !important; }
    .m14 { left: 8% !important; animation-duration: 3.6s !important; animation-delay: 2.5s !important; }
    .m15 { left: 25% !important; animation-duration: 2.9s !important; animation-delay: 1.8s !important; }
    .m16 { left: 38% !important; animation-duration: 4.3s !important; animation-delay: 2.8s !important; }
    .m17 { left: 55% !important; animation-duration: 3.1s !important; animation-delay: 0.6s !important; }
    .m18 { left: 68% !important; animation-duration: 3.7s !important; animation-delay: 2.1s !important; }
    .m19 { left: 82% !important; animation-duration: 2.7s !important; animation-delay: 1.3s !important; }
    .m20 { left: 92% !important; animation-duration: 4.1s !important; animation-delay: 0.9s !important; }
</style>
"""


SIDEBAR_WIDTH_CSS = """
<style>
/* Make sidebar 35% when expanded */
[data-testid="stSidebar"][aria-expanded="true"] {
    width: 35% !important;
    max-width: 45% !important;
    min-width: 280px !important;
}

/* Keep collapsed sidebar narrow */
[data-testid="stSidebar"][aria-expanded="false"] {
    width: 56px !important;
}

/* Shift main content to account for the sidebar */
.main > .block-container,
[data-testid="stAppViewContainer"] > div:first-child {
    margin-left: 35% !important;
    width: calc(100% - 35%) !important;
    transition: margin-left 0.18s ease, width 0.18s ease;
}

/* If your app uses a slightly different container class, include this too (safe fallback) */
.reportview-container .main .block-container {
    margin-left: 35% !important;
    width: calc(100% - 35%) !important;
}

/* Responsive: on small screens let the sidebar overlay the content */
@media (max-width: 900px) {
    [data-testid="stSidebar"][aria-expanded="true"] {
        position: fixed !important;
        z-index: 999 !important;
        width: 100% !important;
    }
    .main > .block-container,
    [data-testid="stAppViewContainer"] > div:first-child {
        margin-left: 0 !important;
        width: 100% !important;
    }
}
</style>
"""


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingestion_running" not in st.session_state:
        st.session_state.ingestion_running = False
    if "ingestion_output" not in st.session_state:
        st.session_state.ingestion_output = []
    if "show_ingestion_logs" not in st.session_state:
        st.session_state.show_ingestion_logs = False
    if "ingestion_success" not in st.session_state:
        st.session_state.ingestion_success = None
    if "ingestion_log_output" not in st.session_state:
        st.session_state.ingestion_log_output = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "initial_question" not in st.session_state:
        st.session_state.initial_question = None
    if "selected_suggestion" not in st.session_state:
        st.session_state.selected_suggestion = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = "groq/openai/gpt-oss-120b"


def get_files_in_data_dir() -> List[dict]:
    """Get list of files in the data directory with metadata."""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return []

    files = []
    for file_path in DATA_DIR.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            files.append({
                "name": file_path.name,
                "path": file_path,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
            })

    return sorted(files, key=lambda x: x["modified"], reverse=True)


def get_processed_documents_stats() -> dict:
    """Get statistics from tracking file about processed documents."""
    try:
        if not TRACKING_FILE.exists():
            return {"count": 0, "total_chunks": 0}

        import json
        with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
            tracking_data = json.load(f)

        processed_docs = tracking_data.get("processed_documents", {})
        count = len(processed_docs)
        total_chunks = sum(doc["chunks_count"] for doc in processed_docs.values())

        return {"count": count, "total_chunks": total_chunks}
    except:
        return {"count": 0, "total_chunks": 0}


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def delete_file(file_path: Path) -> bool:
    """Delete a file from the data directory."""
    try:
        file_path.unlink()
        return True
    except Exception as e:
        st.error(f"Error deleting file: {e}")
        return False


def clear_processing_history() -> tuple[bool, str]:
    """Clear the processing history (tracking JSON file)."""
    try:
        if TRACKING_FILE.exists():
            TRACKING_FILE.unlink()
            return True, "Successfully cleared processing history"
        else:
            return True, "No processing history to clear"
    except Exception as e:
        return False, f"Error clearing processing history: {str(e)}"


def clear_vector_store() -> tuple[bool, str]:
    """Clear all vectors from the Qdrant collection."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        qdrant_url = os.getenv("QDRANT_URL", "https://f58f1067-58c2-413d-acd3-8e6058389e20.us-east4-0.gcp.cloud.qdrant.io")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.UvgYpKQ7MmBOsYDTXxSx-Pg_g5l0Ti-oDxgqG2I80SQ")

        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=True,
        )

        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if QDRANT_COLLECTION_NAME not in collection_names:
            return False, f"Collection '{QDRANT_COLLECTION_NAME}' not found"

        info_before = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        points_count = info_before.points_count

        if points_count == 0:
            return True, "Vector store is already empty"

        offset = None
        batch_size = 100
        total_deleted = 0

        while True:
            points, offset = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )

            if not points:
                break

            point_ids = [point.id for point in points]

            client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )

            total_deleted += len(point_ids)

            if offset is None:
                break

        return True, f"Successfully deleted {total_deleted} vectors from collection '{QDRANT_COLLECTION_NAME}'"

    except Exception as e:
        return False, f"Error clearing vector store: {str(e)}"


def run_ingestion_script(log_container):
    """Run the ingestion script and capture output in real-time."""
    st.session_state.ingestion_running = True
    st.session_state.ingestion_output = []

    try:
        project_root = Path(__file__).parent
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'

        process = subprocess.Popen(
            [sys.executable, "-u", str(INGESTION_SCRIPT), "--yes"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            universal_newlines=True,
            env=env,
            cwd=str(project_root),
            encoding='utf-8',
            errors='replace',
        )

        output_lines = []
        error_lines = []

        with log_container:
            log_display = st.empty()
            status_display = st.empty()

            import time

            while True:
                stdout_line = process.stdout.readline()
                if stdout_line:
                    line = stdout_line.rstrip()
                    if line:
                        output_lines.append(line)
                        st.session_state.ingestion_output.append(line)
                        log_display.code('\n'.join(output_lines[-50:]), language='bash')

                stderr_line = process.stderr.readline()
                if stderr_line:
                    line = stderr_line.rstrip()
                    if line:
                        error_lines.append(f"[ERROR] {line}")
                        output_lines.append(f"[ERROR] {line}")
                        log_display.code('\n'.join(output_lines[-50:]), language='bash')

                if process.poll() is not None:
                    remaining_stdout = process.stdout.read()
                    if remaining_stdout:
                        for line in remaining_stdout.split('\n'):
                            if line.strip():
                                output_lines.append(line.strip())

                    remaining_stderr = process.stderr.read()
                    if remaining_stderr:
                        for line in remaining_stderr.split('\n'):
                            if line.strip():
                                error_lines.append(f"[ERROR] {line.strip()}")
                                output_lines.append(f"[ERROR] {line.strip()}")
                    break

                time.sleep(0.01)

            log_display.code('\n'.join(output_lines), language='bash')
            status_display.info(f"Process finished with return code: {process.returncode}")

        st.session_state.ingestion_running = False

        all_output = output_lines + error_lines

        if process.returncode == 0:
            return True, all_output if all_output else ["Ingestion completed successfully (no output)"]
        else:
            return False, all_output if all_output else [f"Process failed with return code: {process.returncode}"]

    except Exception as e:
        st.session_state.ingestion_running = False
        import traceback
        error_details = traceback.format_exc()
        error_lines = [f"EXCEPTION: {str(e)}", "Full traceback:", error_details]

        with log_container:
            st.error('\n'.join(error_lines))

        return False, error_lines


async def chat_with_agent(user_message: str, context: Context):
    """Send a message to the LangGraph agent and get response."""
    try:
        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }
        
        result = await graph.ainvoke(
            {"messages": [("user", user_message)]},
            config=config,
            context=context,
        )

        if result and "messages" in result and len(result["messages"]) > 0:
            final_message = result["messages"][-1]
            return final_message.content
        else:
            return "No response received from agent."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error: {str(e)}\n\nDetails:\n{error_details}"


def render_sidebar():
    """Render the sidebar with file management."""
    with st.sidebar:
        st.markdown("### 📁 Document Management")

        files = get_files_in_data_dir()
        stats = get_processed_documents_stats()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(files)}</div>
                <div class="stats-label">Documents</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{stats["count"]}</div>
                <div class="stats-label">Processed</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{stats["total_chunks"]}</div>
                <div class="stats-label">Chunks</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("#### ⬆️ Upload Documents")

        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=["pdf", "txt", "doc", "docx", "md"],
            help=f"Maximum {MAX_FILES_UPLOAD} files, {MAX_FILE_SIZE_MB}MB each",
            key="file_uploader",
        )

        if uploaded_files:
            if len(uploaded_files) > MAX_FILES_UPLOAD:
                st.error(f"⚠️ Maximum {MAX_FILES_UPLOAD} files allowed at once!")
            else:
                valid_files = []
                for uploaded_file in uploaded_files:
                    size_mb = uploaded_file.size / (1024 * 1024)
                    if size_mb > MAX_FILE_SIZE_MB:
                        st.error(f"⚠️ {uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit!")
                    else:
                        valid_files.append(uploaded_file)

                if valid_files and st.button("💾 Save Files", type="primary"):
                    for uploaded_file in valid_files:
                        file_path = DATA_DIR / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    st.success(f"✅ Saved {len(valid_files)} file(s)!")
                    st.rerun()

        st.markdown("---")

        st.markdown("#### 🔄 Pipeline Controls")

        col1, col2, col3 = st.columns(3)

        with col1:
            parse_clicked = st.button("🚀\nParse", disabled=st.session_state.ingestion_running, help="Parse and ingest documents", use_container_width=True)

        if parse_clicked:
            if len(files) == 0:
                st.warning("⚠️ No files to parse!")
            elif not INGESTION_SCRIPT.exists():
                st.error(f"❌ Ingestion script not found at: {INGESTION_SCRIPT}")
            else:
                st.markdown("---")
                st.markdown("### 📋 Ingestion Progress")
                st.markdown("*Live output from ingestion script*")

                log_container = st.container()

                try:
                    success, output = run_ingestion_script(log_container)

                    st.session_state.show_ingestion_logs = True
                    st.session_state.ingestion_success = success
                    st.session_state.ingestion_log_output = output

                    st.markdown("---")
                    if success:
                        st.success("✅ Ingestion completed successfully!")
                        st.markdown(f"**Processed:** {len(output)} log lines")
                        st.balloons()
                    else:
                        st.error("❌ Ingestion failed! Check the logs above for details.")

                    col_close1, col_close2, col_close3 = st.columns([1, 1, 1])
                    with col_close2:
                        if st.button("✖️ Close Logs & Refresh", type="primary", use_container_width=True):
                            st.session_state.show_ingestion_logs = False
                            st.session_state.ingestion_success = None
                            st.session_state.ingestion_log_output = []
                            st.rerun()

                except Exception as e:
                    st.error(f"❌ Critical error: {str(e)}")
                    import traceback
                    with st.expander("🐛 Debug Info"):
                        st.code(traceback.format_exc())

                    if st.button("✖️ Close Error Log"):
                        st.rerun()

        elif st.session_state.show_ingestion_logs:
            st.markdown("---")
            st.markdown("### 📋 Ingestion Logs (Previous Run)")

            if st.session_state.ingestion_log_output:
                st.code('\n'.join(st.session_state.ingestion_log_output), language='bash')

            st.markdown("---")
            if st.session_state.ingestion_success:
                st.success("✅ Ingestion completed successfully!")
            else:
                st.error("❌ Ingestion failed! Check the logs above for details.")

            col_close1, col_close2, col_close3 = st.columns([1, 1, 1])
            with col_close2:
                if st.button("✖️ Close Logs & Refresh", type="primary", use_container_width=True, key="close_persistent"):
                    st.session_state.show_ingestion_logs = False
                    st.session_state.ingestion_success = None
                    st.session_state.ingestion_log_output = []
                    st.rerun()

        with col2:
            if st.button("🧹\nClear\nHistory", help="Clear processing history", use_container_width=True):
                if stats["count"] == 0:
                    st.info("No processing history to clear")
                else:
                    with st.spinner("Clearing..."):
                        success, message = clear_processing_history()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                    st.rerun()

        with col3:
            if st.button("💥\nClear\nVectors", help="Clear vector store", use_container_width=True):
                with st.spinner("Clearing..."):
                    success, message = clear_vector_store()
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                st.rerun()

        st.markdown("---")

        with st.expander(f"📄 Documents ({len(files)})", expanded=len(files) > 0 and len(files) <= 3):
            if len(files) == 0:
                st.info("No documents uploaded")
            else:
                for file_info in files:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"""
                        <div class="file-card">
                            <div class="file-name">📄 {file_info['name']}</div>
                            <div class="file-meta">{format_file_size(file_info['size'])}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        if st.button("🗑️", key=f"delete_{file_info['name']}", help="Delete"):
                            if delete_file(file_info['path']):
                                st.success("Deleted!")
                                st.rerun()

        st.markdown("---")

        st.markdown("#### ⚙️ Settings")

        with st.expander("LLM Configuration"):
            model_name = st.text_input(
                "Model",
                value=st.session_state.model_name,
                help="Format: provider/model-name",
            )
            st.session_state.model_name = model_name
        
        with st.expander("💾 Session Memory Info"):
            st.markdown(f"""
            <div class="session-info">
                <strong>Session ID:</strong><br>
                <code>{st.session_state.thread_id[:8]}...{st.session_state.thread_id[-8:]}</code><br>
                <small>This ID tracks your conversation context</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="session-info">
                <strong>Messages in Memory:</strong> {len(st.session_state.messages)}<br>
                <small>Agent remembers all messages in this session</small>
            </div>
            """, unsafe_allow_html=True)


def render_chat_interface():
    """Render the main chat interface with professional design."""
    
    # Check if user just asked initial question or clicked suggestion
    user_just_asked_initial_question = (
        st.session_state.initial_question is not None and st.session_state.initial_question
    )
    
    user_just_clicked_suggestion = (
        st.session_state.selected_suggestion is not None and st.session_state.selected_suggestion
    )
    
    user_first_interaction = (
        user_just_asked_initial_question or user_just_clicked_suggestion
    )
    
    has_message_history = len(st.session_state.messages) > 0
    
    # Welcome screen for first-time users
    if not user_first_interaction and not has_message_history:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">🤖</div>
            <div class="welcome-title">Welcome to LangGraph RAG Agent</div>
            <div class="welcome-subtitle">Ask me anything about your documents or general questions</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat input at the bottom
        st.chat_input("Ask a question...", key="initial_question")
        
        # Suggestion pills
        st.markdown('<div class="suggestion-container">', unsafe_allow_html=True)
        st.markdown('<div class="suggestion-title">Try asking:</div>', unsafe_allow_html=True)
        st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=list(SUGGESTIONS.keys()),
            key="selected_suggestion",
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.stop()
    
    # Title row with restart button
    title_row = st.container()
    with title_row:
        col1, col2 = st.columns([6, 1])
        
        with col1:
            st.markdown("""
            <div class="main-header">
                <h1>💬 Chat with Your Documents</h1>
                <p>Session memory enabled - I remember our conversation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("🔄 Restart", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.session_state.initial_question = None
                st.session_state.selected_suggestion = None
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    user_message = None
    
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
        st.session_state.initial_question = None  # Reset
    elif user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]
        st.session_state.selected_suggestion = None  # Reset
    else:
        user_message = st.chat_input("Ask a follow-up...")
    
    # Process user message
    if user_message:
        # Escape dollar signs for LaTeX
        user_message = user_message.replace("$", r"\$")
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_message)
        
        # Add to history
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                context = Context(
                    model=st.session_state.model_name
                )
                response = asyncio.run(chat_with_agent(user_message, context))
            
            st.markdown(response)
        
        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()


def main():
    """Main application."""
    initialize_session_state()
    st.markdown(SIDEBAR_WIDTH_CSS, unsafe_allow_html=True)

    # Apply dark mode CSS
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
    
    # Inject meteors HTML and styles
    st.markdown(METEORS_HTML, unsafe_allow_html=True)
    
    render_sidebar()
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-secondary); font-size: 0.9rem; padding: 1rem; position: relative; z-index: 2; font-family: 'Plus Jakarta Sans', sans-serif;">
        <p>Powered by LangGraph 🦜 | Built with Streamlit 🎈 | Session Memory Enabled 💾</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()