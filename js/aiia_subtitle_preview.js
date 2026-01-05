
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.AIIA.SubtitlePreview",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AIIA_Subtitle_Preview") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const widget = {
                    type: "div",
                    name: "subtitle_preview_container",
                    draw(ctx, node, widget_width, y, widget_height) {
                        // Custom drawing if needed, but we rely on DOM
                    },
                    computeSize: () => [400, 300], // Initial size
                };
                
                // Create DOM Elements
                this.previewContainer = document.createElement("div");
                this.previewContainer.style.display = "flex";
                this.previewContainer.style.flexDirection = "column";
                this.previewContainer.style.backgroundColor = "#000";
                this.previewContainer.style.borderRadius = "8px";
                this.previewContainer.style.padding = "10px";
                this.previewContainer.style.gap = "10px";
                this.previewContainer.style.width = "100%";
                this.previewContainer.style.boxSizing = "border-box";
                this.previewContainer.style.minHeight = "200px";

                // Audio Player
                this.audioEl = document.createElement("audio");
                this.audioEl.controls = true;
                this.audioEl.style.width = "100%";
                
                // Subtitle Display Area
                this.subEl = document.createElement("div");
                this.subEl.style.flexGrow = "1";
                this.subEl.style.display = "flex";
                this.subEl.style.justifyContent = "center";
                this.subEl.style.alignItems = "center";
                this.subEl.style.textAlign = "center";
                this.subEl.style.backgroundColor = "#1a1a1a";
                this.subEl.style.color = "#fff";
                this.subEl.style.fontSize = "20px";
                this.subEl.style.padding = "20px";
                this.subEl.style.minHeight = "100px";
                this.subEl.style.borderRadius = "4px";
                this.subEl.style.position = "relative";
                this.subEl.innerText = "Waiting for content...";

                this.previewContainer.appendChild(this.subEl);
                this.previewContainer.appendChild(this.audioEl);
                
                // Add to DOM widget (ComfyUI specific)
                // We use addDOMWidget if available or append to node widgets mechanism
                // Modern ComfyUI uses `this.addDOMWidget`
                if (this.addDOMWidget) {
                    this.addDOMWidget("subtitle_preview", "preview", this.previewContainer);
                } else {
                   // Fallback for older ComfyUI (unlikely needed but safe)
                   document.body.appendChild(this.previewContainer); // This is bad, just hope addDOMWidget exists
                }

                // Data Store
                this.subtitleEvents = [];
                this.assStyles = {};

                // Audio Time Update
                this.audioEl.ontimeupdate = () => {
                    const currentTime = this.audioEl.currentTime;
                    this.updateSubtitle(currentTime);
                };
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);

                const data = message; // { text: [...], audio: [...] }
                
                if (data.text && data.text.length > 0) {
                    const rawText = data.text[0];
                    this.parseSubtitles(rawText);
                }

                if (data.audio && data.audio.length > 0) {
                    const audioInfo = data.audio[0];
                    const filename = audioInfo.filename;
                    const type = audioInfo.type || "temp";
                    const subfolder = audioInfo.subfolder || "";
                    
                    const src = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`);
                    this.audioEl.src = src;
                    this.audioEl.play().catch(e => console.log("Auto-play blocked", e));
                }
            };

            // Helper: Update Subtitle Text
            nodeType.prototype.updateSubtitle = function(currentTime) {
                const activeEvents = this.subtitleEvents.filter(e => currentTime >= e.start && currentTime <= e.end);
                
                if (activeEvents.length > 0) {
                    // Prioritize last started event (layering)
                    const evt = activeEvents[activeEvents.length - 1];
                    
                    // Update Text
                    // Handle \N for newlines
                    let htmlText = evt.text.replace(/\\N/g, "<br>");
                    this.subEl.innerHTML = htmlText;

                    // Apply Style (if ASS)
                    if (evt.style && this.assStyles[evt.style]) {
                        const s = this.assStyles[evt.style];
                        this.subEl.style.color = s.primaryColor || "#ffffff";
                        this.subEl.style.fontSize = (s.fontSize || 20) + "px";
                        this.subEl.style.fontFamily = s.fontName || "Arial";
                        
                        // Outline/Shadow simulation
                        if (s.outlineColor) {
                             // Basic text shadow to simulate outline
                             this.subEl.style.textShadow = `0px 0px 2px ${s.outlineColor}, 0px 0px 2px ${s.outlineColor}`;
                        }
                    } else {
                        // Reset defaults
                        this.subEl.style.color = "#ffffff";
                        this.subEl.style.fontSize = "20px";
                        this.subEl.style.textShadow = "none";
                    }
                } else {
                    this.subEl.innerHTML = "";
                    this.subEl.style.textShadow = "none";
                }
            };

            // Helper: Parse Subtitles
            nodeType.prototype.parseSubtitles = function(text) {
                this.subtitleEvents = [];
                this.assStyles = {};

                if (text.trim().startsWith("[Script Info]")) {
                    this.parseASS(text);
                } else {
                    this.parseSRT(text);
                }
                
                console.log("[AIIA] Parsed subtitle events:", this.subtitleEvents.length);
            };

            // Parser: SRT
            nodeType.prototype.parseSRT = function(text) {
                // Regex for standard SRT block
                const regex = /(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\n$|$)/g;
                let match;
                
                const timeToSec = (t) => {
                    const [h, m, s_ms] = t.split(":");
                    const [s, ms] = s_ms.split(",");
                    return parseInt(h)*3600 + parseInt(m)*60 + parseInt(s) + parseInt(ms)/1000;
                };

                while ((match = regex.exec(text)) !== null) {
                    this.subtitleEvents.push({
                        start: timeToSec(match[2]),
                        end: timeToSec(match[3]),
                        text: match[4].trim(),
                        style: null
                    });
                }
            };

            // Parser: ASS
            nodeType.prototype.parseASS = function(text) {
                const lines = text.split("\n");
                let section = "";
                let formatEvents = [];
                let formatStyles = [];
                
                for (let line of lines) {
                    line = line.trim();
                    if (!line) continue;
                    
                    if (line.startsWith("[")) {
                        section = line;
                        continue;
                    }

                    if (section === "[V4+ Styles]") {
                        if (line.startsWith("Format:")) {
                            formatStyles = line.substring(7).trim().split(",").map(s => s.trim().toLowerCase());
                        } else if (line.startsWith("Style:")) {
                            const parts = line.substring(6).trim().split(",");
                            if (parts.length > formatStyles.length) { 
                                // Handle commas in font name? ASS is CSV but rigid.
                                // Usually safe to split by comma for standard fields.
                            }
                            
                            const styleObj = {};
                            formatStyles.forEach((key, idx) => styleObj[key] = parts[idx]);
                            
                            // Parse Color &HBBGGRR -> #RRGGBB
                            const parseColor = (assColor) => {
                                if (!assColor) return "#ffffff";
                                // &H00BBGGRR or &HBBGGRR
                                const hex = assColor.replace(/&H/g, "").replace(/&/g, ""); 
                                // Remove alpha if present (first 2 chars if len=8)
                                let c = hex;
                                if (hex.length >= 8) c = hex.substring(2);
                                if (c.length === 6) {
                                    // B G R
                                    const b = c.substring(0, 2);
                                    const g = c.substring(2, 4);
                                    const r = c.substring(4, 6);
                                    return `#${r}${g}${b}`;
                                }
                                return "#ffffff";
                            };

                            this.assStyles[styleObj.name] = {
                                primaryColor: parseColor(styleObj.primarycolour),
                                outlineColor: parseColor(styleObj.outlinecolour),
                                fontSize: parseInt(styleObj.fontsize),
                                fontName: styleObj.fontname
                            };
                        }
                    } else if (section === "[Events]") {
                        if (line.startsWith("Format:")) {
                            formatEvents = line.substring(7).trim().split(",").map(s => s.trim().toLowerCase());
                        } else if (line.startsWith("Dialogue:")) {
                            // Dialogue is tricky because Text can contain commas.
                            // Split by comma only for known fields-1, then rest is text.
                            const LIMIT = formatEvents.length - 1; 
                            const parts = line.substring(9).trim().split(",");
                            
                            // Re-join text part
                            const easyParts = parts.slice(0, LIMIT);
                            const textPart = parts.slice(LIMIT).join(",");
                            
                            const evtObj = {};
                            formatEvents.slice(0, LIMIT).forEach((key, idx) => evtObj[key] = easyParts[idx]);
                            
                            const timeToSec = (t) => {
                                const [parts1, cs] = t.split(".");
                                const [h, m, s] = parts1.split(":");
                                return parseInt(h)*3600 + parseInt(m)*60 + parseInt(s) + parseInt(cs)/100;
                            };

                            this.subtitleEvents.push({
                                start: timeToSec(evtObj.start),
                                end: timeToSec(evtObj.end),
                                style: evtObj.style,
                                text: textPart
                            });
                        }
                    }
                }
            };

        }
    }
});
