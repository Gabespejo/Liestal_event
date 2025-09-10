# generate_slider_html.py
import os, pathlib

# --- Config (filenames like 0000..0012) ---
time_steps = [f"{i:04d}" for i in range(13)]  # 0000..0012
left_dir  = "ZELL_PLOTS/Zell_2m_Combiprecip"        # Only rainfall
right_dir = "ZELL_PLOTS/Zell_2m_bach_Combiprecip"   # Rainfall + discharge
left_prefix, right_prefix = "Zell_2m-", "Zell_2m_bach-"
image_suffix = "_nocbar.png"

# Write the page at the REPO ROOT (GitHub Pages folder = root)
output_html = "index.html"

html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Zell – Only Rainfall vs Rainfall + Discharge</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ margin:0; font-family: Arial, sans-serif; background:#f7f7f7; }}
    .slider-container {{ text-align:center; padding:10px; background:#eee; position:sticky; top:0; z-index:10; }}
    input[type=range] {{ width:40%; height:25px; }}
    .wrapper {{ display:flex; flex-direction:row; height:calc(100vh - 60px); }}
    .grid {{ display:flex; flex-wrap:wrap; justify-content:center; align-items:flex-start; gap:20px; padding:20px; flex-grow:1; overflow-y:auto; }}
    .panel {{ width:48%; min-width:320px; background:#fff; padding:12px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.08); }}
    .panel h4 {{ font-size:14px; margin:5px 0 10px 0; }}
    .panel img {{ max-width:100%; height:auto; border:1px solid #ccc; border-radius:6px; }}
    .side-panel {{ width:120px; display:flex; flex-direction:column; align-items:center; justify-content:center; background:#eee; padding:10px; }}
    .side-panel img {{ width:80px; margin-top:10px; }}
    .step-label {{ font-weight:600; }}
  </style>
</head>
<body>

<div class="slider-container">
  <div><span class="step-label">Step:</span> <span id="stepTxt">{time_steps[0]}</span></div>
  <input type="range" min="0" max="{len(time_steps)-1}" value="0" id="leadSlider">
</div>

<div class="wrapper">
  <div class="grid">
    <div class="panel">
      <h4>Only rainfall</h4>
      <img id="img_left" src="{left_dir}/{left_prefix}{time_steps[0]}{image_suffix}" alt="Only rainfall">
    </div>
    <div class="panel">
      <h4>Only rainfall + Discharge</h4>
      <img id="img_right" src="{right_dir}/{right_prefix}{time_steps[0]}{image_suffix}" alt="Rainfall + discharge">
    </div>
  </div>

  <!-- keep this image next to index.html or change the path -->
  <div class="side-panel">
    <img src="water_depth_cbar_vertical.png" alt="Colorbar">
  </div>
</div>

<script>
  const steps = {time_steps};
  const slider = document.getElementById("leadSlider");
  const stepTxt = document.getElementById("stepTxt");
  const leftImg = document.getElementById("img_left");
  const rightImg = document.getElementById("img_right");

  const leftDir = "{left_dir}";
  const rightDir = "{right_dir}";
  const leftPrefix = "{left_prefix}";
  const rightPrefix = "{right_prefix}";
  const suffix = "{image_suffix}";

  function setStep(i){{
    i = Math.max(0, Math.min(steps.length-1, Number(i)||0));
    const s = steps[i];
    stepTxt.textContent = s;
    leftImg.src  = `${{leftDir}}/${{leftPrefix}}${{s}}${{suffix}}?v=${{s}}`;
    rightImg.src = `${{rightDir}}/${{rightPrefix}}${{s}}${{suffix}}?v=${{s}}`;
    slider.value = i;
  }}

  slider.addEventListener("input", e => setStep(e.target.value));
  window.addEventListener("keydown", e => {{
    if (e.key === "ArrowRight") setStep(Number(slider.value)+1);
    if (e.key === "ArrowLeft")  setStep(Number(slider.value)-1);
  }});

  setStep(0);
</script>
</body>
</html>
"""

with open(output_html, "w", encoding="utf-8") as f:
    f.write(html)

print("✅ HTML saved to:", pathlib.Path(output_html).resolve())
print("📂 Expected images (case-sensitive):")
print("   ZELL_PLOTS/Zell_2m_Combiprecip/Zell_2m-0000_nocbar.png … 0012")
print("   ZELL_PLOTS/Zell_2m_bach_Combiprecip/Zell_2m_bach-0000_nocbar.png … 0012")
print("ℹ️ Put water_depth_cbar_vertical.png next to index.html or adjust its path.")

