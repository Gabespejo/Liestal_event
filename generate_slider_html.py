# generate_slider_html.py
import os
from pathlib import Path

# ================== CONFIG ==================
# Folders containing the PNGs (relative to where the HTML will live)
cosmo_dir = "ZELL_PLOTS/Zell_2m_COSMO"
combi_dir = "ZELL_PLOTS/Zell_2m_Combiprecip"

# 11 ensembles r1..r11
ensembles = list(range(1, 12))

# COSMO lead times available (images named wd_r{r}_lead_time{L}_zoom.png)
lead_times_ens = list(range(1, 10))      # 1..9

# Combiprecip image uses L+3: wd_det_lead_time{4..12}_zoom.png
lead_offset_combi = 3

# Optional colorbar image (shown on the right)
colorbar_png = "water_depth_cbar_vertical.png"

# Output HTML filename
output_html = "slider_cosmo11_plus_combiprecip_4x3.html"
# ============================================

max_idx = len(lead_times_ens) - 1

html_start = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>COSMO Ensembles + Combiprecip (4×3)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --panel-w: 360px;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      background: #f7f7f7;
      font-family: Arial, sans-serif;
    }}
    .slider-container {{
      text-align: center;
      padding: 10px 10px 0 10px;
      background-color: #eeeeee;
      position: sticky;
      top: 0;
      z-index: 2;
      border-bottom: 1px solid #ddd;
    }}
    .slider-container input[type=range] {{
      width: 40%;
      height: 24px;
    }}
    .slider-labels {{
      margin: 6px 0 10px 0;
      font-size: 14px;
      color: #333;
    }}
    .wrapper {{
      display: flex;
      flex-direction: row;
      min-height: calc(100vh - 80px);
    }}
    .grid {{
      flex-grow: 1;
      overflow: auto;
      padding: 16px;
    }}
    .grid-inner {{
      display: grid;
      grid-template-columns: repeat(4, var(--panel-w));
      gap: 18px;
      justify-content: center;
    }}
    .panel {{
      width: var(--panel-w);
    }}
    .panel h4 {{
      margin: 6px 0 8px 0;
      font-size: 14px;
      font-weight: 600;
    }}
    .panel img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #ccc;
      background: #fff;
    }}
    .side-panel {{
      width: 120px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background-color: #eeeeee;
      padding: 10px;
      border-left: 1px solid #ddd;
    }}
    .side-panel img {{
      width: 80px;
      margin-top: 10px;
    }}
    .dim {{
      color: #666;
      font-weight: normal;
    }}
  </style>
</head>
<body>

<div class="slider-container">
  <div>
    <label for="leadSlider"><strong>Lead Time</strong> (ensembles): </label>
    <input type="range" min="0" max="{max_idx}" value="0" id="leadSlider">
  </div>
  <div class="slider-labels">
    Ensembles show <code>lead_time1..9</code>. Combiprecip panel shows <code>lead_time4..12</code> (offset +3).
    &nbsp;Use ← / → to change the step.
  </div>
</div>

<div class="wrapper">
  <div class="grid">
    <div class="grid-inner">
"""

# Build 11 ensemble panels (r1..r11) + 1 combiprecip panel
panels = []
first_L = lead_times_ens[0]  # 1
for r in ensembles:
    src = os.path.join(cosmo_dir, f"wd_r{r}_lead_time{first_L}_zoom.png")
    panels.append(f"""
      <div class="panel">
        <h4>Ensemble {r} <span class="dim">(r{r})</span></h4>
        <img id="img_r{r}" src="{src}" alt="Ensemble r{r}">
      </div>
    """)

first_combi = first_L + lead_offset_combi  # 1+3 = 4
src_combi = os.path.join(combi_dir, f"wd_det_lead_time{first_combi}_zoom.png")
panels.append(f"""
      <div class="panel">
        <h4>Combiprecip <span class="dim">(deterministic)</span></h4>
        <img id="img_combi" src="{src_combi}" alt="Combiprecip det">
      </div>
""")

html_mid = "\n".join(panels)

side_panel = f"""
    </div> <!-- grid-inner -->
  </div>   <!-- grid -->
  <div class="side-panel">
    <img src="{colorbar_png}" alt="Colorbar">
  </div>
</div>  <!-- wrapper -->
"""

# JS: update images when slider changes; also arrow-key navigation
lead_times_js = "[" + ",".join(str(L) for L in lead_times_ens) + "]"
script = f"""
<script>
  const leadEns = {lead_times_js};            // [1..9]
  const leadOffsetCombi = {lead_offset_combi}; // +3 -> 4..12

  const cosmoDir = "{cosmo_dir}".replace(/\\\\/g, "/");
  const combiDir = "{combi_dir}".replace(/\\\\/g, "/");

  const ensembleIds = [{",".join(f'"img_r{r}"' for r in ensembles)}];
  const rNums       = [{",".join(str(r) for r in ensembles)}];

  function setImages(idx) {{
    if (idx < 0) idx = 0;
    if (idx > leadEns.length - 1) idx = leadEns.length - 1;

    const L  = leadEns[idx];           // ensembles lead_time
    const Lc = L + leadOffsetCombi;    // combiprecip lead_time

    // Update ensemble panels
    for (let i = 0; i < ensembleIds.length; i++) {{
      const id  = ensembleIds[i];
      const r   = rNums[i];
      const img = document.getElementById(id);
      if (img) img.src = `${{cosmoDir}}/wd_r${{r}}_lead_time${{L}}_zoom.png?v=${{L}}`;
    }}

    // Update combiprecip panel
    const combi = document.getElementById("img_combi");
    if (combi) combi.src = `${{combiDir}}/wd_det_lead_time${{Lc}}_zoom.png?v=${{Lc}}`;
  }}

  const slider = document.getElementById("leadSlider");
  slider.addEventListener("input", e => setImages(parseInt(e.target.value, 10)));

  // Arrow keys
  window.addEventListener("keydown", e => {{
    if (e.key === "ArrowRight") slider.value = Math.min(leadEns.length-1, parseInt(slider.value, 10)+1);
    if (e.key === "ArrowLeft")  slider.value = Math.max(0, parseInt(slider.value, 10)-1);
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") setImages(parseInt(slider.value, 10));
  }});

  // Initialize
  setImages(parseInt(slider.value, 10));
</script>
"""

html_end = """
</body>
</html>
"""

full_html = html_start + html_mid + side_panel + script + html_end
Path(output_html).write_text(full_html, encoding="utf-8")

print("✅ Wrote:", Path(output_html).resolve())
print("📂 COSMO images like:", os.path.join(cosmo_dir, "wd_r1_lead_time1_zoom.png"))
print("📂 COMBI image like:", os.path.join(combi_dir, "wd_det_lead_time4_zoom.png"))
