# generate_slider_html.py
import os
import pathlib

# === Configuration ===
# Steps 0000..0012 (13 frames). Change if your files differ.
steps = [f"{i:04d}" for i in range(13)]

# Folders RELATIVE to the HTML file:
# Use "." if PNGs are in the same folder as the HTML,
# or e.g. "ZELL_PLOTS/Zell_2m_Combiprecip" etc. for GitHub Pages under docs/.
left_dir  = "."   # folder for Zell_2m-*.png (only rainfall)
right_dir = "."   # folder for Zell_2m_bach-*.png (rainfall + discharge)

# Filenames
left_prefix  = "Zell_2m-"
right_prefix = "Zell_2m_bach-"
image_suffix = "_nocbar.png"

# Output HTML path (same folder as script by default)
output_html = os.path.join(".", "slider_viewer.html")

# === HTML content ===
html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Zell – Only Rainfall vs Rainfall + Discharge</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ margin:0; font-family: Arial, sans-serif; background:#f7f7f7; }}
    .bar {{ text-align:center; padding:12px; background:#eee; position:sticky; top:0; z-index:10; }}
    .wrap {{ display:flex; gap:20px; padding:20px; justify-content:center; align-items:flex-start; flex-wrap:wrap; }}
    .panel {{ background:#fff; border-radius:12px; padding:12px; box-shadow:0 2px 10px rgba(0,0,0,.08); width:min(48vw, 680px); }}
    .panel h3 {{ margin:0 0 8px 0; font-size:16px; text-align:left; }}
    .panel img {{ width:100%; height:auto; border:1px solid #ccc; border-radius:8px; }}
    input[type=range] {{ width:min(60vw, 720px); }}
    .label {{ font-weight:600; }}
    .hint {{ color:#555; font-size:12px; margin-top:6px; }}
  </style>
</head>
<body>

  <div class="bar">
    <div><span class="label">Step:</span> <span id="stepTxt">{steps[0]}</span></div>
    <input id="slider" type="range" min="0" max="{len(steps)-1}" value="0" step="1">
    <div class="hint">Use ← / → to change step</div>
  </div>

  <div class="wrap">
    <div class="panel">
      <h3>Only rainfall</h3>
      <img id="leftImg" src="{left_dir}/{left_prefix}{steps[0]}{image_suffix}" alt="Only rainfall">
    </div>
    <div class="panel">
      <h3>Only rainfall + Discharge</h3>
      <img id="rightImg" src="{right_dir}/{right_prefix}{steps[0]}{image_suffix}" alt="Rainfall + discharge">
    </div>
  </div>

  <script>
    const steps = {steps};
    const leftDir = "{left_dir}";
    const rightDir = "{right_dir}";
    const leftPrefix = "{left_prefix}";
    const rightPrefix = "{right_prefix}";
    const suffix = "{image_suffix}";

    const slider  = document.getElementById("slider");
    const stepTxt = document.getElementById("stepTxt");
    const leftImg = document.getElementById("leftImg");
    const rightImg= document.getElementById("rightImg");

    function setStep(idx) {{
      idx = Math.max(0, Math.min(steps.length-1, Number(idx)||0));
      const s = steps[idx];
      stepTxt.textContent = s;
      // cache-buster to avoid stale images from browser cache
      leftImg.src  = `${{leftDir}}/${{leftPrefix}}${{s}}${{suffix}}?v=${{s}}`;
      rightImg.src = `${{rightDir}}/${{rightPrefix}}${{s}}${{suffix}}?v=${{s}}`;
      slider.value = idx;
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

# === Write file
with open(output_html, "w", encoding="utf-8") as f:
    f.write(html)

print("✅ HTML viewer created at:", pathlib.Path(output_html).resolve())
print("📂 Expected images (adjust folders if needed):")
print(f"   {left_dir}/{left_prefix}0000{image_suffix} … 0012")
print(f"   {right_dir}/{right_prefix}0000{image_suffix} … 0012")