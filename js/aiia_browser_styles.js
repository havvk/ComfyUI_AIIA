// aiia_browser_styles.js (V82 - 虚拟滚动引擎重构)
export const browserStyles = `
    /* Main Browser Styles */
    #aiia-browser-menu-button { margin-left: 10px; }
    .comfy-modal.aiia-browser-dialog-root { top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80vw; height: 85vh; max-width: 1400px; max-height: 1000px; min-width: 800px; min-height: 500px; display: flex; flex-direction: column; padding: 0; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); resize: both; overflow: hidden; }
    .aiia-browser-titlebar { background: var(--comfy-box-bg); padding: 4px 8px; font-weight: bold; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); flex-shrink: 0; color: #F9FAFB; cursor: default; }
    .aiia-browser-main-container { display: flex; flex-direction: column; flex-grow: 1; padding: 8px; overflow: hidden; }
    .aiia-browser-header-controls { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-shrink: 0; gap: 8px; flex-wrap: wrap; }
    .aiia-browser-nav-container { display: flex; flex-grow: 1; align-items: center; gap: 4px; border: 1px solid var(--border-color); border-radius: 4px; padding-right: 4px; }
    .aiia-browser-nav-button { background: none; border: none; font-size: 18px; cursor: pointer; color: #E5E7EB; padding: 4px 8px; line-height: 1; }
    .aiia-browser-nav-button:disabled { color: #555; cursor: not-allowed; }
    .aiia-browser-nav-button:hover:not(:disabled) { background-color: var(--comfy-hover-bg); border-radius: 2px; }
    .aiia-browser-breadcrumbs { flex-grow: 1; padding: 4px 8px; border: none; display: flex; flex-wrap: nowrap; gap: 4px; color: #D1D5DB; overflow: hidden; cursor: text; }
    .aiia-browser-breadcrumb-link { cursor: pointer; color: var(--link-color); white-space: nowrap; }
    .aiia-browser-path-input-container { flex-grow: 1; display: none; align-items: center; background-color: var(--comfy-input-bg); border: 1px solid var(--input-active-border); border-radius: 4px; padding: 0 8px; }
    .aiia-browser-path-prefix { color: var(--descrip-text); margin-right: 4px; white-space: nowrap; }
    .aiia-browser-path-input { flex-grow: 1; background: transparent; border: none; outline: none; color: var(--input-text); padding: 4px 0; font-family: monospace; }
    
    .aiia-browser-view-controls { display: flex; align-items: center; gap: 4px; }
    .aiia-browser-view-controls button, .aiia-browser-view-controls select, .aiia-browser-view-controls input { padding: 4px 8px; border: 1px solid var(--border-color); background: var(--comfy-input-bg); color: #E5E7EB; border-radius: 4px; cursor: pointer; vertical-align: middle; }
    .aiia-browser-view-controls button.active { border-color: var(--input-active-border); background: var(--comfy-menu-bg); }
    .aiia-browser-sort-controls { display: flex; align-items: center; gap: 4px; }
    .aiia-browser-sort-controls button { font-size: 12px; padding: 4px 6px; }
    
    /* --- Settings Panel Styles --- */
    .aiia-settings-container { position: relative; }
    .aiia-browser-settings-button { font-size: 16px; padding: 4px 6px; line-height: 1; }
    .aiia-settings-panel {
        display: none;
        position: absolute;
        top: 100%;
        left: 0;
        background-color: var(--comfy-menu-bg);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 8px 12px;
        z-index: 1003;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: #E5E7EB;
        font-size: 13px;
        min-width: 220px;
    }
    .aiia-settings-panel > div:first-of-type {
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 6px;
        margin-bottom: 4px;
    }
    .aiia-settings-label { 
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer; 
        white-space: nowrap; 
        padding: 6px 4px;
        border-radius: 3px;
    }
    .aiia-settings-label:hover {
        background-color: var(--comfy-hover-bg);
        color: var(--link-color);
    }
    
    .aiia-split-view-container { display: flex; flex-grow: 1; gap: 8px; overflow: hidden; position: relative; }
    .aiia-directory-panel { width: 220px; min-width: 150px; max-width: 50%; flex-shrink: 0; display: flex; flex-direction: column; border: 1px solid var(--border-color); resize: horizontal; overflow: hidden; background-color: var(--comfy-box-bg); }
    .aiia-directory-panel.hidden { display: none; }
    .aiia-directory-panel-header { padding: 8px; font-weight: bold; border-bottom: 1px solid var(--border-color); flex-shrink: 0; text-align: center; color: #E5E7EB; }
    .aiia-directory-list { overflow-y: auto; padding: 4px; }
    .aiia-directory-item { display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 4px; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #D1D5DB; }
    .aiia-directory-item:before { content: '📁'; font-size: 16px; }
    .aiia-directory-item:hover { background-color: var(--comfy-hover-bg); color: #F9FAFB; }
    .aiia-directory-item.focused, .aiia-item-icon.focused, .aiia-list-row.focused { outline: 2px solid var(--accent-color, #4a90e2) !important; outline-offset: -2px; background-color: var(--comfy-hover-bg); }
    
    .aiia-content-panel { flex-grow: 1; border: 1px solid var(--border-color); background-color: var(--comfy-box-bg); overflow: hidden; display: flex; }
    .aiia-browser-content { width: 100%; height: 100%; position: relative; overflow: hidden; display: flex; }
    
    .aiia-grid-wrapper::-webkit-scrollbar, .aiia-list-body::-webkit-scrollbar, .aiia-custom-tooltip::-webkit-scrollbar, .aiia-filmstrip-container::-webkit-scrollbar, .aiia-directory-list::-webkit-scrollbar { width: 16px; height: 16px; }
    .aiia-grid-wrapper::-webkit-scrollbar-track, .aiia-list-body::-webkit-scrollbar-track, .aiia-custom-tooltip::-webkit-scrollbar-track, .aiia-filmstrip-container::-webkit-scrollbar-track, .aiia-directory-list::-webkit-scrollbar-track { background: transparent; }
    .aiia-grid-wrapper::-webkit-scrollbar-thumb, .aiia-list-body::-webkit-scrollbar-thumb, .aiia-custom-tooltip::-webkit-scrollbar-thumb, .aiia-filmstrip-container::-webkit-scrollbar-thumb, .aiia-directory-list::-webkit-scrollbar-thumb { background-color: rgba(155, 155, 155, 0.5); border-radius: 8px; border: 4px solid transparent; background-clip: content-box; }
    .aiia-grid-wrapper:hover::-webkit-scrollbar-thumb, .aiia-list-container:hover .aiia-list-body::-webkit-scrollbar-thumb, .aiia-filmstrip-container:hover::-webkit-scrollbar-thumb, .aiia-directory-list:hover::-webkit-scrollbar-thumb { background-color: rgba(155, 155, 155, 0.8); }
    
    .aiia-grid-wrapper { height: 100%; width: 100%; overflow-y: scroll; overflow-x: hidden; position: relative; }
    .aiia-virtual-scroller { position: absolute; top: 0; left: 0; width: 1px; opacity: 0; pointer-events: none; }
    .aiia-grid-inner-wrapper { display: grid; grid-template-columns: repeat(auto-fill, minmax(var(--aiia-icon-size, 120px), 1fr)); gap: 10px; padding: 10px; position: absolute; top: 0; left: 0; right: 0; }
    
    .aiia-item-icon {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        border: 1px solid var(--border-color);
        border-radius: 5px;
        cursor: pointer;
        position: relative;
        background-color: var(--comfy-menu-bg);
        overflow: hidden;
    }
    .aiia-item-icon:hover .aiia-icon-button-overlay {
        opacity: 1;
        transform: translateY(0);
    }
    .aiia-icon-preview {
        position: relative;
        flex-grow: 1;
        min-height: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        overflow: hidden;
    }
    .aiia-icon-preview.media-placeholder { background-color: var(--comfy-box-bg); }
    .aiia-image-preview, .aiia-video-preview {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        width: calc(100% - 16px);
        height: calc(100% - 16px);
        border-radius: 4px;
    }
    .aiia-audio-container { display: flex; flex-direction: column; justify-content: center; align-items: center; width: 100%; height: 100%; gap: 8px; }
    .aiia-audio-container audio { width: 90%; }
    .aiia-audio-duration { font-size: 11px; color: var(--descrip-text); }
    .aiia-item-label {
        flex-shrink: 0;
        width: 100%;
        text-align: center;
        font-size: 12px;
        padding: 8px 4px;
        white-space: normal;
        word-break: break-all;
        color: #E5E7EB;
        line-height: 1.3;
        background-color: rgba(0,0,0,0.2);
    }
    .aiia-item-label.has-workflow {
        background-color: rgba(59, 130, 246, 0.25);
        border-top: 1px solid rgba(96, 165, 250, 0.3);
        color: #eff6ff;
    }
    
    .aiia-video-poster-container::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 30%;
        height: 30%;
        max-width: 48px;
        max-height: 48px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 50%;
        -webkit-mask-image: url('data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="black"%3E%3Cpath d="M8 5v14l11-7z"/%3E%3C/svg%3E');
        mask-image: url('data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="black"%3E%3Cpath d="M8 5v14l11-7z"/%3E%3C/svg%3E');
        -webkit-mask-size: 60%;
        mask-size: 60%;
        -webkit-mask-repeat: no-repeat;
        -webkit-mask-position: center;
        mask-position: center;
        transition: background-color 0.2s ease-in-out, transform 0.2s ease-in-out;
        pointer-events: none;
    }
    .aiia-item-icon:hover .aiia-video-poster-container::after {
        background-color: white;
        transform: translate(-50%, -50%) scale(1.1);
    }

    .aiia-custom-tooltip { position: absolute; display: none; padding: 12px; background-color: rgba(30, 30, 30, 0.95); border: 1px solid var(--border-color); color: white; border-radius: 8px; z-index: 1002; font-size: 13px; pointer-events: none; backdrop-filter: blur(4px); box-sizing: border-box; flex-direction: column; align-items: flex-start; }
    .aiia-tooltip-media { width: 100%; margin-bottom: 10px; max-width: var(--aiia-tooltip-media-max-size, 250px); }
    .aiia-tooltip-metadata { display: block; max-width: var(--aiia-tooltip-media-max-size, 250px); }
    .aiia-tooltip-media:not(:has(> :not([style*="display: none"]))) { display: none; margin-bottom: 0; }
    .aiia-tooltip-image, .aiia-tooltip-video { width: 100%; height: auto; object-fit: contain; display: none; border-radius: 4px; max-width: var(--aiia-tooltip-media-max-size, 250px); max-height: var(--aiia-tooltip-media-max-size, 250px); }
    .aiia-tooltip-row { display: grid; grid-template-columns: auto 1fr; gap: 8px; margin-top: 4px; }
    .aiia-tooltip-label { color: #aaa; text-align: right; white-space: nowrap; }
    .aiia-tooltip-value { color: #eee; font-weight: bold; white-space: normal; overflow-wrap: break-word; min-width: 0; }
    
    .aiia-tooltip-metadata audio {
        width: 100%;
        margin-bottom: 8px;
    }
    .aiia-tooltip-metadata audio::-webkit-media-controls-play-button,
    .aiia-tooltip-metadata audio::-webkit-media-controls-mute-button,
    .aiia-tooltip-metadata audio::-webkit-media-controls-volume-slider {
        display: none;
    }
    
    .aiia-list-container { display: flex; flex-direction: column; height: 100%; width: 100%; }
    .aiia-list-body { flex-grow: 1; overflow-y: scroll; position: relative; }
    .aiia-list-body-content { position: relative; width: 100%; height: 100%; }
    .aiia-list-header, .aiia-list-row { display: grid; grid-template-columns: 28px minmax(150px, 3fr) minmax(140px, 1.5fr) minmax(70px, 1fr) minmax(80px, 1fr) minmax(100px, 1fr) minmax(80px, 1fr) 110px; align-items: center; gap: 8px; padding: 0 8px; }
    .aiia-list-row { border-bottom: 1px solid var(--border-color); cursor: pointer; position: absolute; left: 0; right: 0; }
    .aiia-list-header { position: sticky; top: 0; z-index: 10; background: var(--comfy-box-bg); border-bottom: 2px solid var(--border-color); }
    .aiia-list-header-cell { padding: 8px 0; font-weight: bold; color: #F9FAFB; cursor: pointer; user-select: none; }
    .aiia-list-header-cell:hover { color: var(--link-color); }
    .aiia-list-cell { padding: 6px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #E5E7EB; }
    .aiia-list-cell-icon { font-size: 16px; text-align: center; }
    .aiia-list-cell-actions { display: flex; justify-content: center; align-items: center; gap: 4px; }

    .aiia-icon-button-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        padding: 8px;
        box-sizing: border-box;
        display: flex;
        justify-content: flex-end;
        gap: 6px;
        background: linear-gradient(to bottom, rgba(0,0,0,0.6), transparent);
        opacity: 0;
        transform: translateY(-20px);
        transition: opacity 0.2s ease-in-out, transform 0.2s ease-in-out;
        pointer-events: none;
    }
    .aiia-item-icon:hover .aiia-icon-button-overlay,
    .aiia-item-icon.actions-visible .aiia-icon-button-overlay {
        opacity: 1;
        transform: translateY(0);
        pointer-events: all;
    }
    .aiia-icon-button-overlay > button {
        pointer-events: all;
        position: static;
        padding: 4px 8px;
        font-size: 14px;
        line-height: 1;
        background-color: rgba(30, 30, 30, 0.8);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 4px;
        cursor: pointer;
        backdrop-filter: blur(2px);
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
    }
    .aiia-icon-button-overlay > button:hover {
        background-color: rgba(0, 123, 255, 0.9);
        transform: scale(1.05);
    }
    .aiia-icon-button-overlay > button:active {
        transform: scale(1);
    }

    /* Hide old, now-unused absolute buttons in icon view */
    .aiia-item-icon > .aiia-load-workflow-button,
    .aiia-item-icon > .aiia-download-button {
        display: none;
    }

    /* Keep original styles for list view */
    .aiia-list-row .aiia-load-workflow-button, .aiia-list-row .aiia-download-button {
        position: static;
        background-color: transparent;
        border: none;
        color: #E5E7EB;
        font-size: 18px;
    }
    .aiia-list-row .aiia-load-workflow-button:hover, .aiia-list-row .aiia-download-button:hover {
        color: var(--accent-color);
        background-color: transparent;
    }

    .aiia-browser-error, .aiia-browser-placeholder { padding: 20px; text-align: center; font-size: 16px; color: var(--descrip-text); width: 100%; }
    .aiia-empty-file-placeholder, .aiia-no-preview-placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; height: 100%; color: var(--descrip-text); }
    .aiia-empty-file-placeholder span:first-child, .aiia-no-preview-placeholder span:first-child { font-size: 24px; }
    .aiia-empty-file-placeholder span:last-child, .aiia-no-preview-placeholder span:last-child { font-size: 11px; margin-top: 4px; }
    
    .aiia-progress-container { display: none; flex-direction: column; gap: 4px; margin-bottom: 8px; }
    .aiia-progress-text { font-size: 12px; color: var(--descrip-text); }
    .aiia-progress-bar { width: 100%; height: 4px; background-color: var(--comfy-menu-bg); border-radius: 2px; }
    .aiia-progress-bar > div { width: 0%; height: 100%; background-color: var(--accent-color); border-radius: 2px; transition: width 0.2s ease-out; }

    .aiia-fullscreen-viewer { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(10, 10, 10, 0.9); backdrop-filter: blur(8px); z-index: 1001; display: none; flex-direction: column; }
    .aiia-fullscreen-main-content { display: flex; flex-grow: 1; align-items: center; position: relative; width: 100%; overflow: hidden; }
    .aiia-fullscreen-media-container { flex-grow: 1; display: flex; align-items: center; justify-content: center; padding: 20px; box-sizing: border-box; width: 100%; height: 100%; }
    .aiia-fullscreen-media-container > * { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px; }
    .aiia-fullscreen-audio-wrapper { display: flex; flex-direction: column; align-items: center; gap: 20px; color: white; }
    .aiia-fullscreen-audio-icon { font-size: 128px; }
    .aiia-fullscreen-close-button, .aiia-fullscreen-nav { background: rgba(0,0,0,0.4); color: white; border: none; border-radius: 50%; cursor: pointer; font-size: 24px; width: 44px; height: 44px; line-height: 44px; text-align: center; transition: all 0.2s; z-index: 12; }
    .aiia-fullscreen-close-button:hover, .aiia-fullscreen-nav:hover { background: rgba(0,0,0,0.7); }
    .aiia-fullscreen-close-button { position: absolute; top: 20px; right: 20px; }
    .aiia-fullscreen-nav { position: absolute; top: 50%; transform: translateY(-50%); }
    .aiia-fullscreen-nav.prev { left: 20px; }
    .aiia-fullscreen-nav.next { right: 20px; }
    .aiia-fullscreen-nav:disabled { opacity: 0.3; cursor: not-allowed; }
    .aiia-filmstrip-container { z-index: 11; flex-shrink: 0; background: linear-gradient(to top, rgba(0,0,0,0.7), transparent); padding: 15px; display: flex; gap: 10px; overflow-x: auto; width: 100%; box-sizing: border-box; }
    .aiia-filmstrip-thumb { height: 80px; width: 80px; flex-shrink: 0; background: #222; border-radius: 4px; cursor: pointer; border: 2px solid transparent; transition: border-color 0.2s, opacity 0.2s; overflow: hidden; display: flex; align-items: center; justify-content: center; }
    .aiia-filmstrip-thumb:hover { border-color: #555; }
    .aiia-filmstrip-thumb.active { border-color: var(--accent-color, #007BFF); }
    .aiia-filmstrip-thumb-content { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }
    .aiia-filmstrip-thumb img { width: 100%; height: 100%; object-fit: cover; }
    .aiia-filmstrip-thumb video { width: 100%; height: 100%; object-fit: contain; background-color: #000; }
    .aiia-filmstrip-placeholder, .aiia-filmstrip-placeholder-icon { font-size: 32px; color: #888; background-color: #2a2a2a; width:100%; height:100%; display:flex; align-items:center; justify-content:center; }
    .aiia-info-panel { position: absolute; bottom: 120px; right: 20px; width: auto; max-width: 350px; height: auto; background: rgba(15,15,15,0.8); padding: 10px 15px; box-sizing: border-box; color: #eee; border-radius: 6px; transition: all 0.2s ease-in-out; z-index: 11; pointer-events: none; backdrop-filter: blur(4px); }
    .aiia-info-panel:hover { pointer-events: auto; }
    .aiia-info-panel-row { display: grid; grid-template-columns: 80px 1fr; gap: 8px; margin-bottom: 8px; font-size: 13px; }
    .aiia-info-panel-row:last-child { margin-bottom: 0; }
    .aiia-info-panel-label { color: #aaa; text-align: right; }
    .aiia-info-panel-value { font-weight: 500; overflow-wrap: break-word; min-width: 0; }
`;

