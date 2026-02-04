
import { app } from "../../../scripts/app.js";

// Extension to handle static labels (identifying by 'is_label' flag in metadata)
app.registerExtension({
    name: "AIIA.Labels",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Iterate through required and optional inputs to find ones marked as is_label
        const requiredInputs = nodeData.input?.required || {};
        const optionalInputs = nodeData.input?.optional || {};

        let labelWidgetNames = [];
        const checkInputs = (inputs) => {
            for (const [name, inputDef] of Object.entries(inputs)) {
                if (inputDef[1] && inputDef[1].is_label === true) {
                    labelWidgetNames.push(name);
                }
            }
        };

        checkInputs(requiredInputs);
        checkInputs(optionalInputs);

        if (labelWidgetNames.length > 0) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                for (const w of this.widgets) {
                    if (labelWidgetNames.includes(w.name)) {
                        // Change type to avoid standard text box rendering
                        w.type = "AIIA_STATIC_TEXT";

                        // Helper to wrap text
                        const wrapText = (ctx, text, maxWidth) => {
                            const words = text.split(" ");
                            const lines = [];
                            let currentLine = words[0];

                            for (let i = 1; i < words.length; i++) {
                                const word = words[i];
                                const width = ctx.measureText(currentLine + " " + word).width;
                                if (width < maxWidth) {
                                    currentLine += " " + word;
                                } else {
                                    lines.push(currentLine);
                                    currentLine = word;
                                }
                            }
                            lines.push(currentLine);
                            return lines;
                        };

                        w.draw = function (ctx, node, widget_width, y, widget_height) {
                            ctx.save();
                            ctx.fillStyle = "#AAAAAA"; // Label color
                            ctx.font = "italic 12px Arial";

                            const margin = 15;
                            const maxWidth = widget_width - margin * 2;
                            const lines = wrapText(ctx, this.value, maxWidth);

                            let lineY = y + 15;
                            for (const line of lines) {
                                ctx.fillText(line, margin, lineY);
                                lineY += 16; // Line height
                            }

                            // Store the actual height for computeSize
                            this.last_height = (lines.length * 16) + 10;

                            ctx.restore();
                        };

                        // Disable interaction 
                        w.mouse = () => { };
                        w.computeSize = function (width) {
                            return [width || 200, this.last_height || 20];
                        };

                        // Prevent this widget from being converted to a socket (though it's already a Primitive STRING)
                        w.inputKey = null;
                        w.serializeValue = async () => ""; // Don't send label text back to Python
                    }
                }
                return r;
            };
        }
    }
});
