import os, pathlib

# Steps 0000..0012
time_steps = [f"{i:04d}" for i in range(13)]

# Folders relative to the HTML file
left_dir  = "ZELL_PLOTS/Zell_2m_Combiprecip"        # Only rainfall
right_dir = "ZELL_PLOTS/Zell_2m_bach_Combiprecip"   # Rainfall + discharge

left_prefix  = "Zell_2m-"
right_prefix = "Zell_2m_bach-"
image_suffix = "_nocbar.png"

output_html = os.path.join("docs", "slider_all_ensemble_with_cbar_slider_top.html")
os.makedirs(os.path.dirname(output_html), exist_ok=True)

html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Zell – Only Rainfall vs Rainfall + Discharge</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ margin:0; font-family:system-ui, Arial, sans-serif; background:#f7f7f7; }}
    .bar {{ padding:10px 14px; background:#eee; position:sticky; top:0; z-index:10; }}
    .wrap {{ display:flex; gap:20px; padding:20px; justify-content:center; align-items:flex-start; flex-wrap:wrap; }}
    .panel {{ background:#fff; border-radius:12px; padding:12px; box-shadow:0 2px 10px rgba(0,0,0,.08); width:min(48vw, 680px); }}
    .panel h3 {{ margin:0 0 8px 0; font-size:16px; }}
    .panel img {{ width:100%; height:auto; border:1px solid #ddd; border-radius:8px; }}
    .label {{ font-weight:600; }}
    input[type=range] {{ width:min(60vw, 720px); }}
    .hint {{ color:#555; font-size:12px; margin-top:6px; }}
  </style>
</head>
<body>
  <div class="bar">
    <div><span class="label">Step:</span> <span id="stepTxt">{time_steps[0]}</span></div>
    <input id="slider" type="range" min="0" max="{len(time_steps)-1}" step="1" value="0">
    <div class="hint">Tip: use ← / → arrows to change step</div>
  </div>

  <div class="wrap">
    <div class="panel">
      <h3>Only rainfall</h3>
      <img id="leftImg" src="{left_dir}/{left_prefix}{time_steps[0]}{image_suffix}" alt="Only rainfall">
    </div>
    <div class="panel">
      <h3>Only rainfall + Discharge</h3>
      <img id="rightImg" src="{right_dir}/{right_prefix}{time_steps[0]}{image_suffix}" alt="Rainfall + discharge">
    </div>
  </div>

  <script>
    const steps = {time_steps};
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

with open(output_html, "w", encoding="utf-8") as f:
    f.write(html)

print("✅ HTML saved to:", pathlib.Path(output_html).resolve())
print("📂 Put images under:")
print("   docs/ZELL_PLOTS/Zell_2m_Combiprecip/Zell_2m-0000_nocbar.png … 0012")
print("   docs/ZELL_PLOTS/Zell_2m_bach_Combiprecip/Zell_2m_bach-0000_nocbar.png … 0012")

