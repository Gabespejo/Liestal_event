import os
from pathlib import Path

# ================== CONFIG ==================
cosmo_dir = "ZELL_PLOTS/Zell_2m_COSMO"              # ensemble forecast PNGs live here
combi_dir = "ZELL_PLOTS/Zell_2m_Combiprecip"        # Combiprecip timestamp PNGs live here

ensembles = list(range(1, 12))                     # r1..r11
lead_times_ens = list(range(0, 8))                 # slider indices (0..7)  -> you can change length

# Combiprecip filenames are timestamp-based:
# Combiprecip_2022-05-05T12-00-00_zoom.png
# Combiprecip_2022-05-05T13-00-00_zoom.png
# ...
# IMPORTANT: length must match lead_times_ens length (same slider index)
combi_times = [
    "2022-05-05T12-00-00",
    "2022-05-05T13-00-00",
    "2022-05-05T14-00-00",
    "2022-05-05T15-00-00",
    "2022-05-05T16-00-00",
    "2022-05-05T17-00-00",
    "2022-05-05T18-00-00",
    "2022-05-05T19-00-00",
]

# choose your ensemble filename pattern (pick the one you actually have)
# If your ensemble PNGs are named like: Forecast_r1_lead_time0_zoom.png
ENS_PATTERN = "Forecast_r{r}_lead_time{L}_zoom.png"

# Output
output_html = "slider_cosmo11_plus_combiprecip_4x3.html"
# ============================================

# sanity: keep slider length consistent
n_slider = len(lead_times_ens)
if len(combi_times) != n_slider:
    raise ValueError(
        f"combi_times has {len(combi_times)} entries but lead_times_ens has {n_slider}. "
        f"Make them the same length."
    )

max_idx = n_slider - 1

html_start = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>COSMO Ensembles + Combiprecip (4×3)</title>
  <style>
    :root {{
      --panel-w: 360px;
    }}
    body {{
      font-family: Arial, sans-serif;
      background-color: #f7f7f7;
      margin: 0;
      padding: 0;
    }}
    .slider-container {{
      text-align: center;
      padding: 10px 10px 0 10px;
      background-color: #eeeeee;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    .slider-container input[type=range] {{
      width: 40%;
      height: 25px;
    }}
    .slider-labels {{
      margin: 6px 0 10px 0;
      font-size: 14px;
      color: #333;
    }}
    .wrapper {{
      display: flex;
      flex-direction: row;
      height: calc(100vh - 80px);
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
    <label for="leadSlider"><strong>Lead Time</strong> (slider index): </label>
    <input type="range" min="0" max="{max_idx}" value="0" id="leadSlider">
  </div>
  <div class="slider-labels">
    Slider updates all 11 ensembles + Combiprecip timestamp map.
  </div>
</div>

<div class="wrapper">
  <div class="grid">
    <div class="grid-inner">
"""

# ---------------- Panels 1..11: ensembles ----------------
panels = []
first_L = lead_times_ens[0]

for r in ensembles:
    src = os.path.join(cosmo_dir, ENS_PATTERN.format(r=r, L=first_L))
    panels.append(f"""
      <div class="panel">
        <h4>Ensemble {r} <span class="dim">(r{r})</span></h4>
        <img id="img_r{r}" src="{src}" alt="Ensemble r{r}">
      </div>
    """)

# ---------------- Panel 12: Combiprecip ----------------
src_combi = os.path.join(combi_dir, f"Combiprecip_{combi_times[0]}_zoom.png")
panels.append(f"""
      <div class="panel">
        <h4>Combiprecip <span class="dim">(deterministic)</span></h4>
        <img id="img_combi" src="{src_combi}" alt="Combiprecip det">
      </div>
""")

html_mid = "\n".join(panels)

side_panel = """
    </div> <!-- grid-inner -->
  </div>   <!-- grid -->
  <div class="side-panel">
    <img src="water_depth_cbar_vertical.png" alt="Colorbar">
  </div>
</div>  <!-- wrapper -->
"""

# ---------------- JS ----------------
lead_times_js = "[" + ",".join(str(L) for L in lead_times_ens) + "]"
combi_times_js = "[" + ",".join(f'"{t}"' for t in combi_times) + "]"
ensemble_ids_js = "[" + ",".join(f'"img_r{r}"' for r in ensembles) + "]"
rnums_js = "[" + ",".join(str(r) for r in ensembles) + "]"

# ENS_PATTERN used in JS too
ens_pattern_js = ENS_PATTERN.replace("\\", "/")

script = f"""
<script>
  const leadEns = {lead_times_js};               // e.g. [0,1,2,3,4,5,6,7]
  const combiTimes = {combi_times_js};           // timestamp tags, same length as leadEns

  const cosmoDir = "{cosmo_dir}".replace(/\\\\/g, "/");
  const combiDir = "{combi_dir}".replace(/\\\\/g, "/");

  const ensembleIds = {ensemble_ids_js};
  const rNums       = {rnums_js};

  // pattern for ensemble filenames:
  // "Forecast_r{r}_lead_time{L}_zoom.png"
  const ensPattern = "{ens_pattern_js}";

  function ensFilename(r, L) {{
    return ensPattern
      .replaceAll("{r}", String(r))
      .replaceAll("{L}", String(L));
  }}

  function setImages(idx) {{
    if (idx < 0) idx = 0;
    if (idx > leadEns.length - 1) idx = leadEns.length - 1;

    const L = leadEns[idx];

    // ensembles
    for (let i = 0; i < ensembleIds.length; i++) {{
      const id = ensembleIds[i];
      const r  = rNums[i];
      const src = `${{cosmoDir}}/${{ensFilename(r, L)}}`;
      const img = document.getElementById(id);
      if (img) img.src = src;
    }}

    // combiprecip (timestamp-based, same slider index)
    const combi = document.getElementById("img_combi");
    if (combi) {{
      const tag = combiTimes[idx];
      combi.src = `${{combiDir}}/Combiprecip_${{tag}}_zoom.png`;
    }}
  }}

  const slider = document.getElementById("leadSlider");
  slider.addEventListener("input", e => setImages(parseInt(e.target.value, 10)));

  setImages(parseInt(slider.value, 10));
</script>
"""

html_end = """
</body>
</html>
"""

full_html = html_start + html_mid + side_panel + script + html_end

Path(output_html).write_text(full_html, encoding="utf-8")
print(Path(output_html).resolve())


