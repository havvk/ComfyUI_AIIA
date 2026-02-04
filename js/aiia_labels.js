
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

                        w.draw = function (ctx, node, widget_width, y, widget_height) {
                            ctx.save();
                            // Background or subtle underline if needed
                            // ctx.fillStyle = "#222222";
                            // ctx.fillRect(0, y, widget_width, widget_height);

                            ctx.fillStyle = "#AAAAAA"; // Label color
                            ctx.font = "italic 12px Arial";
                            // Draw the text
                            ctx.fillText(this.value, 15, y + widget_height * 0.7);
                            ctx.restore();
                        };

                        // Disable interaction 
                        w.mouse = () => { };
                        w.computeSize = () => [200, 20];

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
