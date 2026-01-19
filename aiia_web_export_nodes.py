
import os
import json
import torchaudio
import folder_paths

class AIIA_Web_Export:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "segments_info": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "aiia_lesson_01"}),
                "template": (["Split Screen (Left/Right)", "Presentation (Top/Bottom)"], {"default": "Split Screen (Left/Right)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("html_file_path",)
    OUTPUT_NODE = True
    FUNCTION = "export_web"
    CATEGORY = "AIIA/Web"

    def export_web(self, audio, segments_info, filename_prefix, template):
        # 1. Prepare Output Directory
        output_dir = folder_paths.get_output_directory()
        # Create a subfolder to keep things clean? or just flat
        # Let's use a subfolder named after the prefix to keep assets together
        save_dir = os.path.join(output_dir, filename_prefix)
        os.makedirs(save_dir, exist_ok=True)

        # 2. Save Audio
        audio_name = f"audio.wav"
        audio_path = os.path.join(save_dir, audio_name)
        
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.ndim == 3: waveform = waveform.squeeze(0)
        torchaudio.save(audio_path, waveform, sample_rate)

        # 3. Parse Segments
        try:
            segments = json.loads(segments_info)
        except:
            segments = []

        # 4. Generate HTML
        html_content = self._generate_html(segments, audio_name, template, filename_prefix)
        
        html_name = f"index.html"
        html_path = os.path.join(save_dir, html_name)
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[AIIA Web Export] Saved to {html_path}")
        return (html_path,)

    def _generate_html(self, segments, audio_filename, template, title):
        # Serialize segments to JSON for JS
        segments_json = json.dumps(segments)

        # CSS Styles
        css = """
            body { font-family: 'Segoe UI', sans-serif; margin: 0; background: #121212; color: #fff; height: 100vh; overflow: hidden; }
            #app { display: flex; width: 100%; height: 100%; }
            
            /* Split Screen */
            .layout-split { flex-direction: row; }
            .layout-split #visual-container { flex: 2; border-right: 1px solid #333; position: relative; }
            .layout-split #sidebar { flex: 1; display: flex; flex-direction: column; background: #1e1e1e; }

            /* Presentation */
            .layout-presentation { flex-direction: column; }
            .layout-presentation #visual-container { flex: 3; border-bottom: 1px solid #333; position: relative; }
            .layout-presentation #sidebar { flex: 1; display: flex; flex-direction: row; background: #1e1e1e; }
            .layout-presentation #subtitle-container { flex: 1; }
            .layout-presentation #controls { width: 300px; }

            /* Visual Area */
            #visual-container iframe, #visual-container img { width: 100%; height: 100%; border: none; object-fit: contain; }
            #visual-placeholder { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #666; }

            /* Subtitle Area */
            #subtitle-container { flex: 1; padding: 20px; overflow-y: auto; scroll-behavior: smooth; position: relative; }
            .seg { padding: 10px; margin-bottom: 10px; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
            .seg:hover { background: #333; }
            .seg.active { background: #3a3a3a; border-left: 4px solid #007bff; }
            .seg-time { font-size: 0.8em; color: #888; margin-bottom: 4px; }
            .seg-speaker { font-weight: bold; color: #4fc3f7; margin-right: 8px; }
            .seg-text { font-size: 1.1em; line-height: 1.5; }
            
            /* Controls */
            #controls { padding: 20px; background: #252525; border-top: 1px solid #333; }
            audio { width: 100%; }
        """
        
        layout_class = "layout-split" if "Split" in template else "layout-presentation"

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{css}</style>
</head>
<body>
    <div id="app" class="{layout_class}">
        <div id="visual-container">
            <div id="visual-placeholder">Waiting for Visual Content...</div>
        </div>
        
        <div id="sidebar">
            <div id="subtitle-container">
                <!-- Segments go here -->
            </div>
            <div id="controls">
                <audio id="player" controls>
                    <source src="{audio_filename}" type="audio/wav">
                </audio>
            </div>
        </div>
    </div>

    <script>
        const segments = {segments_json};
        const visualContainer = document.getElementById('visual-container');
        const subtitleContainer = document.getElementById('subtitle-container');
        const player = document.getElementById('player');
        
        let currentVisual = null;

        // Render Subtitle List
        segments.forEach((seg, index) => {{
            const div = document.createElement('div');
            div.className = 'seg';
            div.id = 'seg-' + index;
            div.onclick = () => {{ player.currentTime = seg.start; player.play(); }};
            
            div.innerHTML = `
                <div class="seg-time">${{formatTime(seg.start)}}</div>
                <div><span class="seg-speaker">${{seg.speaker || 'Unknown'}}:</span> <span class="seg-text">${{seg.text}}</span></div>
            `;
            subtitleContainer.appendChild(div);
            seg.el = div;
        }});

        // Time Update Loop
        player.ontimeupdate = () => {{
            const t = player.currentTime;
            
            // Find active segment
            let activeIdx = -1;
            let activeVisual = null;

            for (let i = 0; i < segments.length; i++) {{
                const seg = segments[i];
                if (t >= seg.start && t < seg.end) {{
                    activeIdx = i;
                }}
                
                // Track visual based on passed time (last valid visual)
                if (t >= seg.start && seg.visual) {{
                    activeVisual = seg.visual;
                }}
            }}

            // Highlight Subtitle
            document.querySelectorAll('.seg.active').forEach(el => el.classList.remove('active'));
            if (activeIdx !== -1) {{
                const el = segments[activeIdx].el;
                el.classList.add('active');
                scrollIntoViewIfNeeded(el, subtitleContainer);
            }}

            // Update Visual
            updateVisual(activeVisual);
        }};

        function updateVisual(url) {{
            if (url === currentVisual) return;
            currentVisual = url;
            
            visualContainer.innerHTML = '';
            if (!url) {{
                visualContainer.innerHTML = '<div id="visual-placeholder">No Visual Content</div>';
                return;
            }}

            // Detect Type (Image vs Iframe)
            const isImage = url.match(/\\.(jpeg|jpg|gif|png|webp)$/i);
            
            if (isImage) {{
                const img = document.createElement('img');
                img.src = url;
                visualContainer.appendChild(img);
            }} else {{
                const iframe = document.createElement('iframe');
                iframe.src = url;
                visualContainer.appendChild(iframe);
            }}
        }}

        function formatTime(s) {{
            const m = Math.floor(s / 60);
            const ss = Math.floor(s % 60);
            return `${{m}}:${{ss.toString().padStart(2, '0')}}`;
        }}

        function scrollIntoViewIfNeeded(target, parent) {{
            const rect = target.getBoundingClientRect();
            const parentRect = parent.getBoundingClientRect();
            if (rect.bottom > parentRect.bottom || rect.top < parentRect.top) {{
                target.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}
    </script>
</body>
</html>
        """

NODE_CLASS_MAPPINGS = {
    "AIIA_Web_Export": AIIA_Web_Export
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Web_Export": "🌐 AIIA Web Export (Interactive)"
}
