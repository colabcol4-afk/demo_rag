import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from qdrant_client import QdrantClient
from umap import UMAP
import json
import os

# --- CONFIGURATION ---
COLLECTION_NAME = "document_chunks"
QDRANT_URL = os.environ.get("QDRANT_URL")
API_KEY = os.environ.get("QDRANT_API_KEY")

LIMIT = 5000 

# --- 1. FETCH DATA ---
print(f"🔌 Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)

print(f"📥 Fetching vectors from '{COLLECTION_NAME}'...")
vectors = []
payloads = []
ids = []
next_offset = None

while len(vectors) < LIMIT:
    records, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_vectors=True,
        with_payload=True,
        offset=next_offset
    )
    
    if not records:
        break
        
    for record in records:
        vectors.append(record.vector)
        ids.append(record.id)
        # Simplify payload for display
        payloads.append(json.dumps(record.payload, indent=2))
        
    if next_offset is None:
        break

vectors = np.array(vectors)
count = len(vectors)
print(f"✅ Loaded {count} data points.")

# --- 2. UMAP REDUCTION ---
if count < 5:
    print("⚠️ Dataset too small for UMAP. Using dummy coordinates.")
    projections = np.zeros((count, 3))
    for i in range(count):
        projections[i] = [i, i, i] 
else:
    print(f"🧠 Computing 3D Map (UMAP)...")
    n_neighbors = min(15, count - 1)
    reducer = UMAP(
        n_components=3, 
        n_neighbors=n_neighbors,
        min_dist=0.1, 
        metric='cosine', 
        random_state=42
    )
    projections = reducer.fit_transform(vectors)

# --- 3. BUILD THE "GAME" ASSET (THE PLOT) ---
print("🎨 Designing Interface...")

df = pd.DataFrame(projections, columns=['x', 'y', 'z'])
df['id'] = ids
df['metadata'] = payloads

# Create the figure with a darker, cleaner look
fig = px.scatter_3d(
    df, x='x', y='y', z='z',
    color='x', # Color by X dimension for a gradient effect
    hover_data={'x':False, 'y':False, 'z':False, 'id':True, 'metadata':True},
    color_continuous_scale='Plasma', # Neon colors
)

# Customize the 3D Scene to look like "Space"
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", # Transparent background
    plot_bgcolor="rgba(0,0,0,0)",
    scene=dict(
        xaxis=dict(visible=False), # Hide axes for "floating data" look
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor='rgba(0,0,0,0)'
    ),
    margin=dict(l=0, r=0, t=0, b=0), # Full bleed
    coloraxis_showscale=False # Hide color bar
)

# Make markers glow slightly
fig.update_traces(
    marker=dict(size=4, opacity=0.8, line=dict(width=0))
)

# Get the raw HTML div for the plot (we will embed this)
plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

# --- 4. INJECT CUSTOM "GAMIFIED" HTML/CSS ---
# This string defines the UI wrapper
custom_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Vector Space</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500&display=swap" rel="stylesheet">
    
    <style>
        :root {{
            --neon-blue: #00f3ff;
            --neon-purple: #bc13fe;
            --bg-dark: #050505;
            --glass: rgba(255, 255, 255, 0.05);
        }}

        body {{
            margin: 0;
            padding: 0;
            background-color: var(--bg-dark);
            color: white;
            font-family: 'Rajdhani', sans-serif;
            overflow: hidden; /* No scrollbars */
        }}

        /* Background Grid Animation */
        .bg-grid {{
            position: absolute;
            width: 200vw;
            height: 200vh;
            background: 
                linear-gradient(rgba(0, 243, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 243, 255, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            transform: perspective(500px) rotateX(60deg) translateY(-100px) translateZ(-200px);
            animation: gridMove 20s linear infinite;
            z-index: -1;
            pointer-events: none;
        }}

        @keyframes gridMove {{
            0% {{ transform: perspective(500px) rotateX(60deg) translateY(0) translateZ(-200px); }}
            100% {{ transform: perspective(500px) rotateX(60deg) translateY(50px) translateZ(-200px); }}
        }}

        /* The Plot Container */
        #plot-container {{
            width: 100vw;
            height: 100vh;
            z-index: 1;
            position: absolute;
            top: 0;
            left: 0;
        }}

        /* HUD - Heads Up Display Overlay */
        .hud-header {{
            position: absolute;
            top: 20px;
            left: 30px;
            z-index: 10;
            pointer-events: none; /* Let clicks pass through to graph */
        }}

        h1 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 2.5rem;
            text-transform: uppercase;
            margin: 0;
            text-shadow: 0 0 10px var(--neon-blue);
            letter-spacing: 2px;
        }}

        .subtitle {{
            color: var(--neon-purple);
            font-size: 1.1rem;
            letter-spacing: 1px;
            margin-top: 5px;
        }}

        /* Stats Card (Glassmorphism) */
        .stats-card {{
            position: absolute;
            bottom: 30px;
            left: 30px;
            background: var(--glass);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 10px;
            border-left: 3px solid var(--neon-blue);
            box-shadow: 0 0 20px rgba(0, 243, 255, 0.1);
            z-index: 10;
            min-width: 200px;
            animation: slideIn 1s ease-out;
        }}

        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 1.2rem;
        }}
        
        .stat-label {{ color: #aaa; font-size: 0.9rem; }}
        .stat-value {{ color: var(--neon-blue); font-weight: bold; font-family: 'Orbitron'; }}

        /* Decorative Corners */
        .corner {{
            position: absolute;
            width: 50px;
            height: 50px;
            border: 2px solid var(--neon-purple);
            z-index: 10;
            pointer-events: none;
            opacity: 0.5;
        }}
        .top-right {{ top: 20px; right: 20px; border-bottom: none; border-left: none; }}
        .bottom-right {{ bottom: 20px; right: 20px; border-top: none; border-left: none; }}

        /* Animations */
        @keyframes slideIn {{
            from {{ transform: translateX(-100px); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}

        /* Loading Overlay (Optional) */
        #loader {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: black;
            z-index: 999;
            display: flex;
            justify-content: center;
            align-items: center;
            color: var(--neon-blue);
            font-family: 'Orbitron';
            font-size: 2rem;
            transition: opacity 1s ease;
        }}

    </style>
</head>
<body>

    <div id="loader">INITIALIZING NEURAL LINK...</div>

    <div class="bg-grid"></div>

    <div class="hud-header">
        <h1>Vector Space</h1>
        <div class="subtitle">PROJECT: {COLLECTION_NAME} // VIZ_MODE: UMAP</div>
    </div>

    <div class="stats-card">
        <div class="stat-row">
            <span class="stat-label">TOTAL NODES</span>
            <span class="stat-value">{count}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">DIMENSIONS</span>
            <span class="stat-value">3D</span>
        </div>
        <div class="stat-row" style="margin-top:10px; font-size:0.8rem; color:white;">
            STATUS: <span style="color:#0f0">ONLINE</span>
        </div>
    </div>

    <div class="corner top-right"></div>
    <div class="corner bottom-right"></div>

    <div id="plot-container">
        {plot_div}
    </div>

    <script>
        // Simple script to fade out loader
        window.addEventListener('load', () => {{
            setTimeout(() => {{
                const loader = document.getElementById('loader');
                loader.style.opacity = '0';
                setTimeout(() => {{ loader.style.display = 'none'; }}, 1000);
            }}, 1500); // Fake load time for effect
        }});
    </script>
</body>
</html>
"""

# --- 5. SAVE ---
with open("gamified_vectors.html", "w", encoding="utf-8") as f:
    f.write(custom_html)

print("🚀 System Online! Open 'gamified_vectors.html' to enter the matrix.")