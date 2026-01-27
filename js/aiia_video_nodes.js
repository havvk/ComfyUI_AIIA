// 文件: js/aiia_video_nodes.js 

import { app } from "../../../scripts/app.js";

// 全局存储原始属性，模仿ttN
let origProps = {};
const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);

// 核心的UI切换函数，模仿ttN
function toggleWidget(node, widget, show = false) {
    // --- Debug Start ---
    // console.log(`AIIA Debug (toggleWidget): Toggling '${widget.name}'. Should show: ${show}`);
    // --- Debug End ---

    if (!widget) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = {
            origType: widget.type,
            origComputeSize: widget.computeSize
        };
    }
    widget.type = show ? origProps[widget.name].origType : "AIIA_HIDDEN";
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];
    node.setDirtyCanvas(true);
}

// 辅助函数，来自VHS
function chainCallback(object, property, callback) {
    if (object[property]) {
        const original = object[property];
        object[property] = function () {
            original.apply(this, arguments);
            callback.apply(this, arguments);
        };
    } else {
        object[property] = callback;
    }
}

app.registerExtension({
    name: "AIIA.VideoNodes.DynamicWidgets.Final",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AIIA_VideoCombine") {

            const widgetsByFormat = nodeData.input.required.format[1].formats;
            if (!widgetsByFormat) return;

            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                const node = this;
                const formatWidget = findWidgetByName(node, "format");
                if (!formatWidget) return;

                const allDynamicWidgetNames = new Set(Object.values(widgetsByFormat).flat().map(p => p[0]));

                const updateWidgetsVisibility = (formatValue) => {
                    // 【核心修正】: 从蓝图数组中，只提取出widget的名称字符串
                    const visibleWidgetNames = new Set(
                        (widgetsByFormat[formatValue] || []).map(p => p[0])
                    );

                    for (const widgetName of allDynamicWidgetNames) {
                        const widget = findWidgetByName(node, widgetName);
                        if (widget) {
                            toggleWidget(node, widget, visibleWidgetNames.has(widgetName));
                        }
                    }

                    node.setSize([node.size[0], node.computeSize()[1]]);
                };

                // 为format widget的callback链接上更新函数
                chainCallback(formatWidget, "callback", updateWidgetsVisibility);

                // Expose function for onConfigure
                node.aiiaUpdateVideoWidgets = updateWidgetsVisibility;

                // 初始加载时触发
                updateWidgetsVisibility(formatWidget.value);
            });

            chainCallback(nodeType.prototype, "onConfigure", function () {
                const node = this;
                // 使用 requestAnimationFrame 确保在所有widget值被写入后再执行更新
                requestAnimationFrame(() => {
                    const formatWidget = findWidgetByName(node, "format");
                    if (formatWidget && node.aiiaUpdateVideoWidgets) {
                        node.aiiaUpdateVideoWidgets(formatWidget.value);
                    }
                });
            });
        }
    }
});