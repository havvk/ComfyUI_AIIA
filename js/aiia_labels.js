
import { app } from "../../../scripts/app.js";

// Extension to handle static labels (AIIA_LABEL)
app.registerExtension({
    name: "AIIA.Labels",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Iterate through inputs to find ones marked as AIIA_LABEL
        const allInputs = { ...nodeData.input?.required, ...nodeData.input?.optional };

        let hasLabel = false;
        for (const [name, input] of Object.entries(allInputs)) {
            if (input[0] === "AIIA_LABEL") {
                hasLabel = true;
                break;
            }
        }

        if (hasLabel) {
            // Hijack onNodeCreated to style our label widgets
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                for (const w of this.widgets) {
                    // Check if this widget was defined as AIIA_LABEL in Python
                    // Since nodeData.input is available, we can verify by name
                    const inputDef = allInputs[w.name];
                    if (inputDef && inputDef[0] === "AIIA_LABEL") {
                        w.type = "text"; // Change to text so it renders
                        w.draw = function (ctx, node, widget_width, y, widget_height) {
                            const show_text = true;
                            const outline_color = app.canvas.ds.outline_color;

                            ctx.save();
                            ctx.fillStyle = "#AAAAAA"; // Label color
                            ctx.font = "italic 12px Arial";
                            // Center or offset as needed
                            ctx.fillText(this.value, 15, y + widget_height * 0.7);
                            ctx.restore();
                        };
                        // Disable interaction
                        w.mouse = () => { };
                        w.computeSize = () => [200, 20];
                    }
                }
                return r;
            };
        }
    }
});
