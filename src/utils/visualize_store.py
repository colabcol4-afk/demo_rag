import numpy as np
import pandas as pd
import plotly.express as px
from qdrant_client import QdrantClient
from umap import UMAP
from tqdm import tqdm
import os

# ----------------------------
# CONFIGURATION (edit as needed)
# ----------------------------
COLLECTION_NAME = "document_chunks"
QDRANT_URL = os.environ.get("QDRANT_URL")
API_KEY = os.environ.get("QDRANT_API_KEY")
LIMIT = 5000
OUTPUT_FILE = "qdrant_viz_gamified.html"

# ----------------------------
# FETCH VECTORS FROM QDRANT
# ----------------------------
client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)
print(f"--- Fetching vectors from '{COLLECTION_NAME}' ---")

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
        payloads.append(record.payload)
    if next_offset is None:
        break

vectors = np.array(vectors)
count = len(vectors)
print(f"\n✅ Successfully fetched {count} vectors.")

# ----------------------------
# DIMENSIONALITY REDUCTION
# ----------------------------
if count < 5:
    print("⚠️  DATASET TOO SMALL FOR UMAP (Need > 5 points). Switching to simple layout.")
    projections = np.zeros((count, 3))
    for i in range(count):
        projections[i] = [i, i, i]
else:
    print(f"🧠 Running UMAP on {count} vectors...")
    n_neighbors = min(15, count - 1)
    reducer = UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    projections = reducer.fit_transform(vectors)

# ----------------------------
# BUILD DATAFRAME & PLOTLY FIGURE
# ----------------------------
print("🎨 Generating Plot...")
df = pd.DataFrame(projections, columns=['x', 'y', 'z'])
df['id'] = ids
# human-readable metadata for the hover and gamified popup
df['metadata'] = [str(p) for p in payloads]

fig = px.scatter_3d(
    df, x='x', y='y', z='z',
    color='id',
    hover_data=['metadata'],
    custom_data=['metadata'],  # ensure metadata is available in click events
    title=f"Visualization of {COLLECTION_NAME} ({count} items)",
    opacity=0.85
)
fig.update_traces(marker=dict(size=5))

# We will embed this Plotly figure into a custom gamified HTML shell.
# Use a fixed div id so the JS below can attach events reliably.
fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='qdrant_plot')

# ----------------------------
# GAMIFIED DARK-THEME HTML TEMPLATE
# ----------------------------
html_start = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gamified Qdrant Viz</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg:#0b1020;
      --panel: rgba(255,255,255,0.04);
      --accent:#7c5cff;
      --accent-2:#00e5a8;
      --muted: rgba(255,255,255,0.55);
      --glass: rgba(255,255,255,0.03);
    }
    *{box-sizing:border-box}
    html,body{height:100%;margin:0;font-family:Inter,system-ui,Segoe UI,Roboto;background:radial-gradient(1200px 600px at 10% 10%, rgba(124,92,255,0.08), transparent),
                 radial-gradient(1000px 500px at 90% 90%, rgba(0,229,168,0.03), transparent), var(--bg); color:#eaf0ff}

    /* animated floating blobs */
    .blob {position:fixed; filter:blur(40px); opacity:0.25; mix-blend-mode:screen; pointer-events:none}
    .b1{width:520px;height:520px;left:-120px;top:-140px;background:linear-gradient(135deg,var(--accent),#5be0ff);animation:float1 12s ease-in-out infinite}
    .b2{width:380px;height:380px;right:-80px;bottom:-100px;background:linear-gradient(135deg,var(--accent-2),#2b6bff);animation:float2 18s ease-in-out infinite}
    @keyframes float1{0%{transform:translateY(0) rotate(0deg)}50%{transform:translateY(20px) rotate(10deg)}100%{transform:translateY(0) rotate(0deg)}}
    @keyframes float2{0%{transform:translateY(0) rotate(0deg)}50%{transform:translateY(-18px) rotate(-6deg)}100%{transform:translateY(0) rotate(0deg)}}

    /* layout */
    .app{display:grid;grid-template-columns:320px 1fr;gap:20px;padding:28px;height:100vh}
    .panel{background:var(--panel);backdrop-filter:blur(6px);border-radius:14px;padding:18px;box-shadow:0 6px 30px rgba(3,8,23,0.6);}

    /* left sidebar */
    .sidebar{display:flex;flex-direction:column;gap:14px}
    .brand{display:flex;align-items:center;gap:12px}
    .logo{width:44px;height:44px;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent-2));display:flex;align-items:center;justify-content:center;font-weight:800;color:white}
    h1{font-size:18px;margin:0}
    .muted{color:var(--muted);font-size:13px}

    .score{display:flex;align-items:center;justify-content:space-between}
    .score .value{font-weight:800;font-size:28px}
    .xpbar{height:12px;border-radius:999px;background:rgba(255,255,255,0.06);overflow:hidden}
    .xpbar > i{display:block;height:100%;width:0%;background:linear-gradient(90deg,var(--accent),var(--accent-2));border-radius:999px;transition:width 1s cubic-bezier(.2,.9,.2,1)}

    .stats{display:flex;gap:10px;flex-wrap:wrap}
    .stat{flex:1;min-width:120px;padding:12px;border-radius:10px;background:linear-gradient(180deg,rgba(255,255,255,0.02),transparent);}
    .stat h3{margin:0;font-size:12px;color:var(--muted)}
    .stat p{margin:6px 0 0 0;font-weight:700}

    .badges{display:flex;gap:8px;flex-wrap:wrap}
    .badge{padding:8px 10px;border-radius:999px;background:rgba(255,255,255,0.03);font-weight:600;font-size:13px;display:inline-flex;align-items:center;gap:8px}
    .badge.glow{box-shadow:0 6px 30px rgba(124,92,255,0.18);transform:translateY(0);transition:transform .2s}

    /* main content */
    .main{display:flex;flex-direction:column;gap:16px}
    .controls{display:flex;gap:12px;align-items:center}
    .search{flex:1}
    .input{background:var(--glass);border:none;padding:12px;border-radius:10px;color:var(--muted);outline:none;width:100%}
    .btn{padding:10px 14px;border-radius:10px;border:none;background:linear-gradient(90deg,var(--accent),var(--accent-2));color:#021;cursor:pointer;font-weight:700}

    .viz-panel{height:calc(100vh - 220px);padding:12px;border-radius:12px}
    /* make plotly fill container */
    #qdrant_plot{height:100%;width:100%}

    /* popup */
    .popup{position:fixed;right:32px;bottom:32px;background:linear-gradient(180deg,rgba(255,255,255,0.03),transparent);padding:14px;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.6);max-width:360px;display:none}
    .popup.show{display:block;animation:pop .35s ease-out}
    @keyframes pop{0%{transform:translateY(12px) scale(.98);opacity:0}100%{transform:none;opacity:1}}

    .footer{color:var(--muted);font-size:13px}

    /* responsive */
    @media (max-width:880px){
      .app{grid-template-columns:1fr;padding:12px}
      .sidebar{flex-direction:row;overflow:auto}
    }
  </style>
</head>
<body>
  <div class="blob b1"></div>
  <div class="blob b2"></div>
  <div class="app">

    <aside class="panel sidebar">
      <div class="brand">
        <div class="logo">Q</div>
        <div>
          <h1>Qdrant Explorer</h1>
          <div class="muted">Gamified dark UI • Explore your vector store</div>
        </div>
      </div>

      <div class="panel score" style="margin-top:12px">
        <div>
          <div class="muted">Explorer Score</div>
          <div class="value" id="score">0</div>
        </div>
        <div style="width:140px">
          <div class="muted" style="font-size:11px;margin-bottom:6px">Progress to next level</div>
          <div class="xpbar"><i id="xpFill" style="width:0%"></i></div>
        </div>
      </div>

      <div class="panel stats">
        <div class="stat">
          <h3>Total points</h3>
          <p id="totalPoints">0</p>
        </div>
        <div class="stat">
          <h3>UMAP mode</h3>
          <p id="umapMode">3D</p>
        </div>
        <div class="stat">
          <h3>Badges</h3>
          <p id="badgeCount">0</p>
        </div>
      </div>

      <div style="margin-top:12px">
        <div class="muted">Achievements</div>
        <div class="badges" id="badges"></div>
      </div>

      <div style="margin-top:14px" class="muted">Tip: Click a point on the plot to inspect metadata and earn score.</div>

    </aside>

    <main class="main">
      <div style="display:flex;align-items:center;justify-content:space-between">
        <div class="controls" style="flex:1">
          <div class="search" style="max-width:620px">
            <input id="searchInput" class="input" placeholder="Search metadata or ID (type and press Enter)"/>
          </div>
          <button class="btn" id="resetBtn">Reset Score</button>
        </div>
        <div style="margin-left:12px;text-align:right">
          <div class="muted">Collection</div>
          <div style="font-weight:700">''' + COLLECTION_NAME + '''</div>
        </div>
      </div>

      <div class="panel viz-panel">
'''

html_end = '''
      </div>

    </main>
  </div>

  <div class="popup" id="popup">
    <div style="font-weight:800" id="popupTitle">Item</div>
    <div class="muted" style="font-size:13px;margin-top:6px" id="popupMeta">metadata</div>
    <div style="display:flex;gap:8px;margin-top:10px;align-items:center">
      <button class="btn" id="copyBtn">Copy metadata</button>
      <div class="muted" style="font-size:13px;margin-left:auto">+<span id="popupScore">0</span> XP</div>
    </div>
  </div>

  <div style="position:fixed;left:12px;bottom:12px;color:var(--muted);font-size:12px">Made with ❤️ — interactive gamified viz</div>

  <script>
    // simple gamification logic
    const STATE = {
      score: Number(localStorage.getItem('q_score')||0),
      clicks: Number(localStorage.getItem('q_clicks')||0),
      badges: JSON.parse(localStorage.getItem('q_badges')||'[]')
    }
    function save(){
      localStorage.setItem('q_score', STATE.score)
      localStorage.setItem('q_clicks', STATE.clicks)
      localStorage.setItem('q_badges', JSON.stringify(STATE.badges))
    }
    function renderState(){
      document.getElementById('score').textContent = STATE.score
      document.getElementById('totalPoints').textContent = ''' + str(count) + '''
      document.getElementById('badgeCount').textContent = STATE.badges.length
      // xp calculation: level every 100 pts
      const xpPercent = Math.min(100, (STATE.score % 100))
      document.getElementById('xpFill').style.width = xpPercent + '%'
      const badgesEl = document.getElementById('badges')
      badgesEl.innerHTML = ''
      STATE.badges.forEach(b=>{ const el=document.createElement('div'); el.className='badge glow'; el.textContent=b; badgesEl.appendChild(el) })
    }
    renderState()

    document.getElementById('resetBtn').addEventListener('click', ()=>{
      STATE.score=0; STATE.clicks=0; STATE.badges=[]; save(); renderState();
    })

    // search
    document.getElementById('searchInput').addEventListener('keydown', (e)=>{
      if(e.key==='Enter'){
        const q = e.target.value.toLowerCase().trim()
        if(!q) return
        // filter points by text match
        const gd = document.getElementById('qdrant_plot')
        const update = { marker:{opacity:0.06} }
        // naive approach: use Plotly.restyle to dim & highlight
        // This example does a simple full reset by calling relayout (for small datasets fine)
        // For larger datasets you'd implement server-side filtering.
        Plotly.restyle(gd, {'marker.opacity':0.06})
        // find matching points and highlight
        const points = []
        const gdData = gd.data
        for(let t=0;t<gdData.length;t++){
          const cd = gdData[t].customdata || []
          for(let i=0;i<cd.length;i++){
            const meta = String(cd[i]).toLowerCase()
            if(meta.includes(q) || String(gdData[t].ids && gdData[t].ids[i] || '').toLowerCase().includes(q)){
              points.push({trace:t, point:i})
            }
          }
        }
        // highlight matched points
        points.forEach(p=>{
          Plotly.restyle(gd, {'marker.opacity':1,'marker.size':8}, [p.trace])
        })
      }
    })

    // popup interactions
    const popup = document.getElementById('popup')
    document.getElementById('copyBtn').addEventListener('click', ()=>{
      const text = document.getElementById('popupMeta').textContent
      navigator.clipboard.writeText(text)
    })

  </script>
'''

# Combine template with Plotly figure HTML and add click-handling JS that hooks into the known div id
html_middle = f"""
{fig_html}

<script>
// attach plotly click events to reward the user and show metadata popup
const gd = document.getElementById('qdrant_plot');
if (gd) {{
  gd.on('plotly_click', function(ev) {{
    try {{
      const p = ev.points[0];
      const meta =
        (p.customdata && p.customdata[0]) ||
        p.text ||
        JSON.stringify(p);

      // update popup
      document.getElementById('popupTitle').textContent =
        'Item: ' + (p.pointNumber !== undefined ? p.pointNumber : '');
      document.getElementById('popupMeta').textContent = meta;

      // score logic
      const prevClicks = Number(localStorage.getItem('q_clicks') || 0);
      const newClicks = prevClicks + 1;
      localStorage.setItem('q_clicks', newClicks);

      let score = Number(localStorage.getItem('q_score') || 0);
      const gained = 10 + Math.min(40, Math.floor(newClicks / 3));
      score += gained;
      localStorage.setItem('q_score', score);

      document.getElementById('popupScore').textContent = gained;

      // badges
      let badges = JSON.parse(localStorage.getItem('q_badges') || '[]');
      if (newClicks === 1 && !badges.includes('Explorer')) badges.push('Explorer');
      if (newClicks === 5 && !badges.includes('Curious')) badges.push('Curious');
      localStorage.setItem('q_badges', JSON.stringify(badges));

      // update UI
      document.getElementById('score').textContent = score;
      document.getElementById('badgeCount').textContent = badges.length;
      document.getElementById('xpFill').style.width = Math.min(100, score % 100) + '%';

      const badgeEl = document.getElementById('badges');
      badgeEl.innerHTML = '';
      badges.forEach(b => {{
        const el = document.createElement('div');
        el.className = 'badge glow';
        el.textContent = b;
        badgeEl.appendChild(el);
      }});

      // popup animation
      const popup = document.getElementById('popup');
      popup.classList.add('show');
      setTimeout(() => popup.classList.remove('show'), 6000);

    }} catch (e) {{
      console.error(e);
    }}
  }});
}}
</script>
"""


# write final html
final_html = html_start + html_middle + html_end
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(final_html)

print(f"🚀 Done! Wrote gamified visualization to '{OUTPUT_FILE}'. Open it in your browser.")
