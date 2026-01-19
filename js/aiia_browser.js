// 文件: aiia_browser.js (V82 - 虚拟滚动引擎重构)

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { $el, ComfyDialog } from "../../scripts/ui.js";
import { AIIAFullscreenViewer } from "./aiia_fullscreen_viewer.js";
import { browserStyles } from "./aiia_browser_styles.js";

const DEBUG = false;
const log = (...args) => DEBUG && console.log("[AIIA Browser Debug]", ...args);

const SETTINGS_PREFIX = "aiia-browser-settings:";
function getSetting(key, defaultValue) {
    try {
        const data = localStorage.getItem(SETTINGS_PREFIX + key);
        return data !== null ? JSON.parse(data) : defaultValue;
    } catch (e) {
        return defaultValue;
    }
}
function setSetting(key, value) {
    try {
        localStorage.setItem(SETTINGS_PREFIX + key, JSON.stringify(value));
    } catch (e) { console.error(`[AIIA] Error setting localStorage for key ${key}:`, e); }
}

const CACHE_PREFIX = "aiia-cache:v3:";
function getCacheKey(path, filename) { return `${CACHE_PREFIX}${path ? `${path}/` : ''}${filename}`; }
function getCache(key) { try { const data = localStorage.getItem(key); return data ? JSON.parse(data) : null; } catch (e) { console.error(`[AIIA] Error parsing cache for key ${key}:`, e); return null; } }
function setCache(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { console.error(`[AIIA] Error setting cache for key ${key}:`, e); } }

function getFileUrl(filename, path = "") { const subfolder = path || ""; return api.apiURL(`/view?type=output&subfolder=${encodeURIComponent(subfolder)}&filename=${encodeURIComponent(filename)}`); }
function getThumbnailUrl(filename, path = "", mtime = 0) { return `/api/aiia/v1/browser/thumbnail?path=${encodeURIComponent(path)}&filename=${encodeURIComponent(filename)}&mtime=${mtime}`; }
function getVideoPosterUrl(filename, path = "", mtime = 0) { return `/api/aiia/v1/browser/poster?path=${encodeURIComponent(path)}&filename=${encodeURIComponent(filename)}&mtime=${mtime}`; }

function getMimeType(filename) { if (typeof filename !== 'string') return 'unknown'; const extension = filename.split('.').pop().toLowerCase(); if (['png', 'jpg', 'jpeg', 'webp', 'gif'].includes(extension)) return 'image'; if (['mp4', 'webm', 'mov', 'avi'].includes(extension)) return 'video'; if (['mp3', 'wav', 'ogg'].includes(extension)) return 'audio'; return 'unknown'; }
function formatBytes(bytes, decimals = 2) { if (bytes === 0) return '0 Bytes'; const k = 1024; const dm = decimals < 0 ? 0 : decimals; const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB']; const i = Math.floor(Math.log(bytes) / Math.log(k)); return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]; }
function formatDate(timestamp) { const date = new Date(timestamp * 1000); return date.toLocaleString(); }
function formatDuration(seconds) { if (isNaN(seconds) || seconds < 0) return '—'; const min = Math.floor(seconds / 60); const sec = Math.floor(seconds % 60); return `${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`; }

class AIIABrowserDialog extends ComfyDialog {
    constructor() {
        super();
        this.currentPath = null;
        this.history = [];
        this.itemsData = [];
        this.itemsDataMap = new Map();
        this.isNavigating = false;
        this.lastActionClickTimestamp = 0;
        this.downloadLock = new Set();

        this.viewMode = getSetting('viewMode', 'icons');
        this.iconSize = getSetting('iconSize', 120);
        this.sortKey = getSetting('sortKey', 'name');
        this.sortDir = getSetting('sortDir', 'asc');
        this.hideFolders = getSetting('hideFolders', false);
        this.enableVideoPreviews = getSetting('enableVideoPreviews', false);

        this.metadataLoader = { controller: null, isLoading: false, path: null, isPaused: false };

        this.lastMouseEvent = null;
        this.isTooltipVisible = false;
        this.tooltipTimeout = null;
        this.isPointerDownOnSettings = false;

        this.fullscreenViewer = null;
        this.visibleItems = [];
        this.directoryItems = [];
        this.currentFocusIndex = -1;
        this.tooltipItemName = null;
        this.isKeyboardNavigating = false;
        this.lastFocus = { dir: -1, file: -1 };

        this.virtualList = { viewport: null, content: null, scroller: null, buffer: 10, lastRendered: { start: -1, end: -1 }, renderRaf: null };
        this.listView = { rowHeight: 31, activeNodes: new Map(), nodePool: [] };
        this.iconView = { resizeObserver: null, columns: 0, rowHeight: 0, activeNodes: new Map(), nodePool: [], mediaCache: new Map() };
        this.applyFocusRaf = null;
        this.iconLoadTimers = new Map();

        this.tooltipImage = $el("img.aiia-tooltip-image");
        this.tooltipVideo = $el("video.aiia-tooltip-video", { autoplay: true, muted: true, loop: true, controls: false, volume: 0.8 });
        this.tooltipAudio = $el("audio", { autoplay: true });
        this.tooltipMediaContainer = $el("div.aiia-tooltip-media", [this.tooltipImage, this.tooltipVideo]);
        this.tooltipMetadata = $el("div.aiia-tooltip-metadata");
        this.tooltipElement = $el("div.aiia-custom-tooltip", [this.tooltipMediaContainer, this.tooltipMetadata]);

        this.directoryList = $el("div.aiia-directory-list");
        this.directoryPanel = $el("div.aiia-directory-panel", [$el("div.aiia-directory-panel-header", { textContent: "Folders" }), this.directoryList]);
        this.contentArea = $el("div.aiia-browser-content");
        this.contentPanel = $el("div.aiia-content-panel", [this.contentArea]);

        const splitViewContainer = $el("div.aiia-split-view-container", [this.directoryPanel, this.contentPanel, this.tooltipElement]);

        this.iconViewObserver = new IntersectionObserver(this.handleIconIntersection.bind(this), { root: this.contentArea, rootMargin: '300px 0px 300px 0px' });

        this.titleElement = $el("span", { textContent: "AIIA Media Browser" });
        this.closeButton = $el("button.close", { textContent: "✖", title: "Close" });
        const titleBar = $el("div.aiia-browser-titlebar", [this.titleElement, this.closeButton]);
        this.backButton = $el("button.aiia-browser-nav-button", { textContent: "◀️", disabled: true, title: "Back" });
        this.upButton = $el("button.aiia-browser-nav-button", { textContent: "🔼", disabled: true, title: "Up to parent directory" });
        this.refreshButton = $el("button.aiia-browser-nav-button", { textContent: "🔄", title: "Refresh current directory" });
        this.breadcrumbs = $el("div.aiia-browser-breadcrumbs");
        this.pathInputContainer = $el("div.aiia-browser-path-input-container");
        this.pathPrefix = $el("span.aiia-browser-path-prefix", { textContent: "output / " });
        this.pathInput = $el("input.aiia-browser-path-input", { type: "text", dataset: { escapeToList: "true" } });
        this.pathInputContainer.append(this.pathPrefix, this.pathInput);
        const navContainer = $el("div.aiia-browser-nav-container", [this.backButton, this.upButton, this.refreshButton, this.breadcrumbs, this.pathInputContainer]);

        this.hideFoldersCheckbox = $el("input", { type: "checkbox", id: "aiia-hide-folders-cb", checked: this.hideFolders });
        const hideFoldersLabel = $el("label.aiia-settings-label", { htmlFor: "aiia-hide-folders-cb" }, [this.hideFoldersCheckbox, $el("span", { textContent: "Hide Folders" })]);

        this.videoPreviewsCheckbox = $el("input", { type: "checkbox", id: "aiia-video-previews-cb", checked: this.enableVideoPreviews });
        const videoPreviewsLabel = $el("label.aiia-settings-label", { htmlFor: "aiia-video-previews-cb" }, [this.videoPreviewsCheckbox, $el("span", { textContent: "Enable Video Previews" })]);

        this.settingsPanel = $el("div.aiia-settings-panel", { style: { display: "none" }, tabindex: -1 }, [
            $el("div", [$el("strong", { textContent: "Display" })]),
            hideFoldersLabel,
            videoPreviewsLabel,
        ]);
        this.settingsButton = $el("button.aiia-browser-settings-button", { textContent: "⚙️" });
        this.settingsContainer = $el("div.aiia-settings-container", [this.settingsButton, this.settingsPanel]);

        this.sortKeySelect = $el("select", [$el("option", { value: "name", textContent: "Sort by Name" }), $el("option", { value: "mtime", textContent: "Sort by Date" }), $el("option", { value: "type", textContent: "Sort by Type" }), $el("option", { value: "size", textContent: "Sort by Size" }), $el("option", { value: "dimensions", textContent: "Sort by Dimensions" }), $el("option", { value: "duration", textContent: "Sort by Duration" })]);
        this.sortDirButton = $el("button", { textContent: "▲" });
        const sortControls = $el("div.aiia-browser-sort-controls", [this.sortKeySelect, this.sortDirButton]);
        this.iconViewButton = $el("button", { textContent: "🖼️", dataset: { tooltipText: "Icon View" } });
        this.listViewButton = $el("button", { textContent: "📄", dataset: { tooltipText: "List View" } });
        this.sizeSlider = $el("input", { type: "range", min: 64, max: 256, value: this.iconSize, step: 8, dataset: { tooltipText: "Change icon size", escapeToList: "true" } });
        const viewControls = $el("div.aiia-browser-view-controls", [this.settingsContainer, sortControls, this.iconViewButton, this.listViewButton, this.sizeSlider]);

        this.progressBar = $el("div.aiia-progress-bar", [$el("div")]);
        this.progressText = $el("div.aiia-progress-text");
        this.progressContainer = $el("div.aiia-progress-container", [this.progressText, this.progressBar]);
        this.progressContainer.style.display = 'none';
        const headerControls = $el("div.aiia-browser-header-controls", [navContainer, viewControls]);
        const mainContainer = $el("div.aiia-browser-main-container", [headerControls, this.progressContainer, splitViewContainer]);
        this.element.replaceChildren(titleBar, mainContainer);
        this.element.classList.add("aiia-browser-dialog-root");
        this.element.setAttribute('tabindex', -1);
        this.bindEvents();
        this.applySettings();
        this.updateSortControls();
        this.resizeObserver = new ResizeObserver(() => { this.updateTooltipSizeLimits(); });
        this.resizeObserver.observe(this.contentPanel);
    }

    applySettings() {
        this.iconViewButton.classList.toggle('active', this.viewMode === 'icons');
        this.listViewButton.classList.toggle('active', this.viewMode === 'list');
        this.sizeSlider.style.display = this.viewMode === 'icons' ? 'inline-block' : 'none';
        document.documentElement.style.setProperty('--aiia-icon-size', `${this.iconSize}px`);
        this.directoryPanel.classList.toggle('hidden', this.hideFolders);
    }

    bindEvents() {
        this.closeButton.onclick = () => this.close();
        this.backButton.onclick = () => this.goBack();
        this.upButton.onclick = () => this.goUp();
        this.refreshButton.onclick = () => this.refreshCurrentDirectory();
        this.breadcrumbs.addEventListener('click', (e) => { this.showPathInput(); e.stopPropagation(); });
        this.pathInput.onkeydown = (e) => { if (e.key === 'Enter') { this.navigateTo(this.pathInput.value.trim()); this.hidePathInput(); } else if (e.key === 'Escape') { this.hidePathInput(); } };
        this.pathInput.onblur = () => this.hidePathInput();

        this.element.addEventListener('mousedown', (e) => {
            if (this.settingsContainer.contains(e.target)) {
                this.isPointerDownOnSettings = true;
            }
        });

        this.element.addEventListener('mouseup', (e) => {
            this.isPointerDownOnSettings = false;
        });

        this.element.addEventListener('click', (e) => {
            if (!this.isPointerDownOnSettings && !this.settingsContainer.contains(e.target)) {
                this.settingsPanel.style.display = 'none';
            }
            if (!e.target.closest('input, select, button, .aiia-item-icon, .aiia-list-row, .aiia-browser-breadcrumb-link, .aiia-directory-item, .aiia-settings-container')) {
                this.element.focus({ preventScroll: true });
            }
        });

        this.element.addEventListener('focusout', (e) => {
            if (!this.element.contains(e.relatedTarget)) this.hideTooltip();
        });

        this.element.addEventListener("dragover", (e) => { e.preventDefault(); });
        this.element.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); });

        this.settingsButton.onclick = (e) => {
            e.stopPropagation();
            const newDisplay = this.settingsPanel.style.display === 'block' ? 'none' : 'block';
            this.settingsPanel.style.display = newDisplay;
            if (newDisplay === 'block') {
                this.settingsPanel.focus();
            }
        };

        this.hideFoldersCheckbox.onchange = (e) => { this.hideFolders = e.target.checked; setSetting('hideFolders', this.hideFolders); this.render(); };
        this.videoPreviewsCheckbox.onchange = (e) => { this.enableVideoPreviews = e.target.checked; setSetting('enableVideoPreviews', this.enableVideoPreviews); this.render(); };

        this.sortKeySelect.onchange = (e) => this.handleSortChange(e.target.value, this.sortDir);
        this.sortDirButton.onclick = () => this.handleSortChange(this.sortKey, this.sortDir === 'asc' ? 'desc' : 'asc');
        this.iconViewButton.onclick = () => this.setViewMode('icons');
        this.listViewButton.onclick = () => this.setViewMode('list');
        this.sizeSlider.oninput = (e) => {
            this.iconSize = e.target.value;
            setSetting('iconSize', this.iconSize);
            document.documentElement.style.setProperty('--aiia-icon-size', `${this.iconSize}px`);
            if (this.viewMode === 'icons') {
                // A full re-render is the most robust way to handle the size change,
                // as it correctly re-initializes the virtual scroller and all its states,
                // mimicking a view switch.
                this.render();
            }
        };

        const splitViewContainer = this.element.querySelector('.aiia-split-view-container');
        splitViewContainer.addEventListener('mouseover', this.handleTooltipMouseOver);
        splitViewContainer.addEventListener('mouseout', this.handleTooltipMouseOut);
        splitViewContainer.addEventListener('mousemove', (e) => { if (this.isKeyboardNavigating) this.isKeyboardNavigating = false; this.lastMouseEvent = e; if (this.isTooltipVisible) this.updateTooltipPosition(e); });

        this.contentArea.addEventListener('click', (e) => {
            const button = e.target.closest('.aiia-load-workflow-button, .aiia-download-button');
            if (button) {
                e.stopPropagation();
                e.preventDefault();
                this.lastActionClickTimestamp = Date.now();
                const itemName = button.closest('[data-item-name]').dataset.itemName;

                if (button.classList.contains('aiia-load-workflow-button')) {
                    this.handleLoadWorkflow(itemName);
                } else if (button.classList.contains('aiia-download-button')) {
                    if (this.downloadLock.has(itemName)) {
                        console.log(`[AIIA] Download for ${itemName} is already in progress or recently triggered. Ignoring.`);
                        return;
                    }
                    this.downloadLock.add(itemName);
                    const url = getFileUrl(itemName, this.currentPath);
                    const link = $el('a', { href: url, download: itemName });
                    link.click();
                    // Remove lock after 2 seconds
                    setTimeout(() => {
                        this.downloadLock.delete(itemName);
                    }, 2000);
                }
            }
        });

        this.element.addEventListener('dblclick', (e) => {
            if (Date.now() - this.lastActionClickTimestamp < 300) {
                return;
            }
            const itemEl = e.target.closest('[data-item-name]');
            if (itemEl) {
                const item = this.itemsDataMap.get(itemEl.dataset.itemName);
                if (item && item.type === 'file' && item.size > 0) {
                    this.openFullscreen(item.name);
                }
            }
        });

        this.element.addEventListener('keydown', this.handleKeyDown.bind(this));
    }

    openFullscreen(itemName) { if (!this.fullscreenViewer) { this.fullscreenViewer = new AIIAFullscreenViewer(); } const filesOnly = this.itemsData.filter(i => i.type === 'file' && i.size > 0); this.sortItems(filesOnly); const currentIndex = filesOnly.findIndex(item => item.name === itemName); if (currentIndex !== -1) { this.fullscreenViewer.show(filesOnly, currentIndex, this.currentPath, () => { this.element.focus({ preventScroll: true }); }); } }

    async handleLoadWorkflow(itemName) {
        if (confirm(`Load workflow from "${itemName}"? This will replace your current workflow.`)) {
            try {
                const res = await fetch(window.location.origin + `/api/aiia/v1/browser/get_workflow?path=${encodeURIComponent(this.currentPath)}&filename=${encodeURIComponent(itemName)}`);
                if (!res.ok) {
                    throw new Error(`HTTP ${res.status}: ${await res.text()}`);
                }
                const data = await res.json();

                // The response can be in two formats: a direct workflow JSON,
                // or an object containing the workflow under a 'workflow' key.
                const workflowData = data.workflow || data;

                // The ComfyUI API can return a graph in two formats. We need to detect which one we have.
                // The API format is a flat object where keys are node IDs and values have `class_type`.
                // The standard format has a `nodes` array.
                const isApiFormat = workflowData && !workflowData.nodes && Object.values(workflowData).every(v => v.class_type);

                if (isApiFormat) {
                    // Handle API format: loadApiJson(apiData, fileName)
                    app.loadApiJson(workflowData, itemName);
                } else {
                    // Handle standard graph format: loadGraphData(graphData, clean, restore_view, workflowName)
                    app.loadGraphData(workflowData, true, true, itemName);
                }

                this.close();
            } catch (e) {
                alert(`Error loading workflow: ${e.message}`);
                console.error(e);
            }
        }
    }
    updateTooltipSizeLimits() { const rect = this.contentPanel.getBoundingClientRect(); const maxSize = Math.min(rect.width, rect.height) * 0.30; this.tooltipElement.style.setProperty('--aiia-tooltip-media-max-size', `${maxSize}px`); }
    updateSortControls() { this.sortKeySelect.value = this.sortKey; this.sortDirButton.textContent = this.sortDir === 'asc' ? '▲' : '▼'; this.sortDirButton.dataset.tooltipText = `Sort ${this.sortDir === 'asc' ? 'Descending' : 'Ascending'}`; }
    showPathInput() { this.breadcrumbs.style.display = 'none'; this.pathInputContainer.style.display = 'flex'; this.pathInput.value = this.currentPath; this.pathInput.focus(); this.pathInput.select(); }
    hidePathInput() { this.pathInputContainer.style.display = 'none'; this.breadcrumbs.style.display = 'flex'; if (!this.element.contains(document.activeElement)) { this.element.focus({ preventScroll: true }); } }

    hideTooltip() {
        clearTimeout(this.tooltipTimeout);
        if (this.isTooltipVisible) {
            this.tooltipElement.style.display = 'none';
            this.isTooltipVisible = false;
            this.tooltipItemName = null;

            this.tooltipVideo.pause();
            this.tooltipAudio.pause();
            this.tooltipImage.src = "";
            this.tooltipVideo.src = "";
            this.tooltipAudio.src = "";

            // Detach the audio element if it was added
            if (this.tooltipAudio.parentNode) {
                this.tooltipAudio.parentNode.removeChild(this.tooltipAudio);
            }

            if (this.metadataLoader.isPaused) {
                this.metadataLoader.isPaused = false;
            }
        }
    }

    setFocus(item, element, scrollIntoView = true, trigger = 'mouse') {
        if (this.applyFocusRaf) cancelAnimationFrame(this.applyFocusRaf);
        clearTimeout(this.tooltipTimeout);
        this.hideTooltip();

        const oldEl = this.element.querySelector('.focused');
        if (oldEl) {
            oldEl.classList.remove('focused');
            oldEl.classList.remove('actions-visible');
        }

        if (!item) { this.currentFocusIndex = -1; return; }

        this.currentFocusIndex = this.getAllNavigableItems().findIndex(i => i.name === item.name);
        if (this.currentFocusIndex > -1) {
            if (item.type === 'directory') this.lastFocus.dir = this.currentFocusIndex;
            else this.lastFocus.file = this.currentFocusIndex;
        }

        if (scrollIntoView) this.ensureFocusedItemVisible();

        const applyFocus = () => {
            let elToFocus = element || this.element.querySelector(`[data-item-name="${CSS.escape(item.name)}"]`);
            if (this.viewMode === 'list' && !elToFocus) {
                this.renderVisibleRows();
                elToFocus = this.element.querySelector(`[data-item-name="${CSS.escape(item.name)}"]`);
            }
            if (elToFocus) {
                const currentlyFocused = this.element.querySelector('.focused');
                if (currentlyFocused) currentlyFocused.classList.remove('focused');

                elToFocus.classList.add('focused');
                this.tooltipTimeout = setTimeout(() => {
                    const currentFocusedEl = this.element.querySelector('.focused');
                    if (currentFocusedEl && currentFocusedEl.dataset.itemName === item.name) {
                        const eventForPositioning = (trigger === 'keyboard' || !this.lastMouseEvent) ? { clientX: currentFocusedEl.getBoundingClientRect().right, clientY: currentFocusedEl.getBoundingClientRect().top } : this.lastMouseEvent;
                        this.updateTooltipContent(item, eventForPositioning);
                    }
                }, 200);
            }
            this.applyFocusRaf = null;
        };

        if ((this.viewMode === 'list' || !element) && trigger === 'keyboard') {
            this.applyFocusRaf = requestAnimationFrame(applyFocus);
        } else {
            applyFocus();
        }
    }

    handleTooltipMouseOver = (e) => { if (this.isKeyboardNavigating) return; const itemEl = e.target.closest('[data-item-name]'); if (itemEl) { const itemData = this.itemsDataMap.get(itemEl.dataset.itemName); if (itemData) this.setFocus(itemData, itemEl, false, 'mouse'); } }
    handleTooltipMouseOut = (e) => { if (!e.currentTarget.contains(e.relatedTarget)) this.setFocus(null, null); }

    updateTooltipPosition(event) { if (!event) return; const xOffset = 20, yOffset = 20; const tooltipRect = this.tooltipElement.getBoundingClientRect(); const containerRect = this.element.querySelector('.aiia-split-view-container').getBoundingClientRect(); const mouseX_inContainer = event.clientX - containerRect.left, mouseY_inContainer = event.clientY - containerRect.top; let targetX = mouseX_inContainer + xOffset; if (targetX + tooltipRect.width > containerRect.width) targetX = mouseX_inContainer - tooltipRect.width - xOffset; if (targetX < 0) targetX = 0; let targetY = mouseY_inContainer + yOffset; if (targetY + tooltipRect.height > containerRect.height) targetY = mouseY_inContainer - tooltipRect.height - yOffset; if (targetY < 0) targetY = 0; this.tooltipElement.style.left = `${targetX}px`; this.tooltipElement.style.top = `${targetY}px`; }

    showTooltip(event) { this.tooltipElement.style.display = 'flex'; this.tooltipElement.style.visibility = 'hidden'; requestAnimationFrame(() => { this.updateTooltipPosition(event); this.tooltipElement.style.visibility = 'visible'; this.isTooltipVisible = true; }); }
    updateTooltipContent(itemData, event) {
        this.tooltipItemName = itemData.name;
        this.tooltipMetadata.innerHTML = '';
        const createMetaRow = (label, value) => { if (value === undefined || value === null || value === '—' || value === '') return; const row = $el("div.aiia-tooltip-row", [$el("span.aiia-tooltip-label", { textContent: `${label}:` }), $el("span.aiia-tooltip-value", { textContent: value })]); this.tooltipMetadata.appendChild(row); return row.querySelector('.aiia-tooltip-value'); };
        createMetaRow("Name", itemData.name);

        const onMediaLoad = () => {
            if (this.metadataLoader.isPaused) this.metadataLoader.isPaused = false;
            if (this.tooltipItemName !== itemData.name) return;
            this.showTooltip(event);
            this.tooltipImage.onload = null;
            this.tooltipImage.onerror = null;
            this.tooltipVideo.onloadedmetadata = null;
            this.tooltipVideo.onerror = null;
            this.tooltipAudio.onloadedmetadata = null;
            this.tooltipAudio.onerror = null;
        };

        const onMediaError = (e) => {
            if (this.metadataLoader.isPaused) this.metadataLoader.isPaused = false;
            e.target.style.display = 'none'; // Hide the broken element
            onMediaLoad(); // Still show the tooltip with metadata
        };

        const mime = getMimeType(itemData.name);
        if (this.viewMode === 'list' && this.metadataLoader.isLoading && (mime === 'image' || mime === 'video')) this.metadataLoader.isPaused = true;

        // Reset media elements
        this.tooltipMediaContainer.style.display = 'block';
        this.tooltipImage.style.display = 'none';
        this.tooltipVideo.style.display = 'none';
        if (this.tooltipAudio.parentNode) this.tooltipAudio.parentNode.removeChild(this.tooltipAudio);
        this.tooltipAudio.removeAttribute('controls');


        if (itemData.type === 'directory') {
            this.tooltipMediaContainer.style.display = 'none';
            createMetaRow("Date", formatDate(itemData.mtime));
            createMetaRow("Type", "Folder");
            createMetaRow("Contains", `${itemData.item_count} items`);
            this.showTooltip(event);
        } else {
            if (itemData.size === 0) {
                this.tooltipMediaContainer.style.display = 'none';
                createMetaRow("Type", itemData.extension.replace('.', '').toUpperCase());
                createMetaRow("Size", formatBytes(0));
                createMetaRow("Status", "Empty File 🚫");
                this.showTooltip(event);
                return;
            }
            createMetaRow("Date", formatDate(itemData.mtime));
            createMetaRow("Type", itemData.extension.replace('.', '').toUpperCase());
            createMetaRow("Size", formatBytes(itemData.size));

            // Re-add the persistent elements to the container
            this.tooltipMediaContainer.replaceChildren(this.tooltipImage, this.tooltipVideo);

            if (mime === 'image') {
                this.tooltipImage.src = getFileUrl(itemData.name, this.currentPath);
                this.tooltipImage.style.display = 'block';
                this.tooltipImage.onload = onMediaLoad;
                this.tooltipImage.onerror = onMediaError;
            } else if (mime === 'video') {
                this.tooltipVideo.src = getFileUrl(itemData.name, this.currentPath);
                this.tooltipVideo.style.display = 'block';
                this.tooltipVideo.onloadedmetadata = onMediaLoad;
                this.tooltipVideo.onerror = onMediaError;
            } else if (mime === 'audio') {
                this.tooltipMediaContainer.style.display = 'none';
                this.tooltipAudio.src = getFileUrl(itemData.name, this.currentPath);
                this.tooltipAudio.setAttribute('controls', 'true');
                this.tooltipMetadata.insertBefore(this.tooltipAudio, this.tooltipMetadata.firstChild); // Prepend audio player
                this.tooltipAudio.onloadedmetadata = onMediaLoad;
                this.tooltipAudio.onerror = onMediaError;
            } else {
                this.tooltipMediaContainer.style.display = 'none';
                this.showTooltip(event);
            }

            const cachedMeta = getCache(getCacheKey(this.currentPath, itemData.name));
            if (cachedMeta && cachedMeta.metadata) {
                if (cachedMeta.metadata.width) createMetaRow("Dimensions", `${cachedMeta.metadata.width} x ${cachedMeta.metadata.height}`);
                if (cachedMeta.metadata.duration) createMetaRow("Duration", formatDuration(cachedMeta.metadata.duration));
                if (cachedMeta.metadata.has_workflow) createMetaRow("Workflow", "Yes");
            } else {
                if (itemData.width) createMetaRow("Dimensions", `${itemData.width} x ${itemData.height}`);
                if (itemData.duration) createMetaRow("Duration", formatDuration(itemData.duration));
                if (itemData.has_workflow) createMetaRow("Workflow", "Yes");
            }
        }
    }

    async navigateTo(path, isRefresh = false) {
        if (this.isNavigating) return;
        if (!isRefresh && this.currentPath === path) return;

        this.isNavigating = true;
        if (this.metadataLoader.isLoading) this.metadataLoader.controller.abort();

        if (!isRefresh) {
            this.lastFocus = { dir: -1, file: -1 };
            this.history.push(path);
        }
        this.currentPath = path;
        this.contentArea.innerHTML = '<div class="aiia-browser-placeholder">Loading...</div>';
        this.directoryList.innerHTML = '<div class="aiia-browser-placeholder" style="padding:10px;font-size:12px;">...</div>';

        try {
            const response = await api.fetchApi(`/aiia/v1/browser/list_items?path=${encodeURIComponent(path)}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);

            this.itemsData = await response.json().then(data => [...data.directories, ...data.files]);

            this.render();

            // Fetch metadata in the background after rendering placeholders
            this.startMetadataBatchLoading();

        } catch (error) {
            this.contentArea.innerHTML = `<div class="aiia-browser-error">Error: ${error.message}</div>`;
        } finally {
            this.isNavigating = false;
        }
    }

    refreshCurrentDirectory() {
        if (this.isNavigating) return;
        log(`Refreshing directory: ${this.currentPath}`);
        this.navigateTo(this.currentPath, true);
    }

    goBack() { if (this.history.length > 1) { this.history.pop(); const path = this.history.pop(); this.navigateTo(path); } }

    goUp() { if (this.currentPath) { const parentPath = this.currentPath.substring(0, this.currentPath.lastIndexOf('/')); this.navigateTo(parentPath); } }

    render() {
        this.backButton.disabled = this.history.length <= 1;
        this.upButton.disabled = !this.currentPath;
        if (this.virtualList.renderRaf) {
            cancelAnimationFrame(this.virtualList.renderRaf);
            this.virtualList.renderRaf = null;
        }
        this.iconViewObserver.disconnect();
        this.applySettings();
        this.updateSortControls();

        this.itemsDataMap.clear();
        for (const item of this.itemsData) this.itemsDataMap.set(item.name, item);

        this.updateBreadcrumbs(this.currentPath);

        const directories = this.itemsData.filter(i => i.type === 'directory');
        const files = this.itemsData.filter(i => i.type !== 'directory');

        if (this.hideFolders) {
            this.sortItems(files);
            this.directoryItems = [];
            this.renderDirectoryPanel([]);
            this.renderContentPanel(files);
        } else {
            this.sortItems(directories);
            this.sortItems(files);
            this.directoryItems = directories;
            this.renderDirectoryPanel(directories);
            this.renderContentPanel(files);
        }
    }
    sortItems(items) { const key = this.sortKey; const dir = this.sortDir === 'asc' ? 1 : -1; items.sort((a, b) => { let valA, valB; if (key === 'type') { valA = a.extension || ''; valB = b.extension || ''; } else if (key === 'dimensions') { valA = (a.width || 0) * (a.height || 0); valB = (b.width || 0) * (b.height || 0); } else if (key === 'duration') { valA = a.duration ?? -1; valB = b.duration ?? -1; } else { valA = a[key]; valB = b[key]; } let comparison = 0; if (typeof valA === 'string' && typeof valB === 'string') { let strA = valA, strB = valB; if (key === 'name') { strA = valA.toLowerCase(); strB = valB.toLowerCase(); } comparison = strA.localeCompare(strB, undefined, { numeric: true }); } else if (typeof valA === 'number' && typeof valB === 'number') { comparison = valA - valB; } return comparison * dir; }); }
    handleSortChange(newSortKey, newSortDir) { this.sortKey = newSortKey; this.sortDir = newSortDir; setSetting('sortKey', this.sortKey); setSetting('sortDir', this.sortDir); this.render(); }
    updateBreadcrumbs(path) { this.breadcrumbs.innerHTML = ''; const rootLink = $el('span.aiia-browser-breadcrumb-link', { textContent: 'output', dataset: { path: '' } }); rootLink.onclick = (e) => { e.stopPropagation(); this.navigateTo(''); }; this.breadcrumbs.appendChild(rootLink); if (path) { let cumulativePath = ''; path.split('/').filter(p => p).forEach(part => { cumulativePath += (cumulativePath ? '/' : '') + part; this.breadcrumbs.appendChild($el('span', { textContent: ' / ' })); const link = $el('span.aiia-browser-breadcrumb-link', { textContent: part, dataset: { path: cumulativePath } }); link.onclick = (e) => { e.stopPropagation(); this.navigateTo(link.dataset.path); }; this.breadcrumbs.appendChild(link); }); } this.hidePathInput(); }
    setViewMode(mode) {
        if (this.viewMode === mode) return;
        this.viewMode = mode;
        setSetting('viewMode', this.viewMode);
        this.refreshCurrentDirectory();
    }
    renderDirectoryPanel(directories) { this.directoryList.innerHTML = ""; if (directories.length === 0 && !this.hideFolders) { this.directoryList.appendChild($el("div.aiia-browser-placeholder", { style: { fontSize: "12px", padding: "10px" }, textContent: "No sub-folders." })); return; } directories.forEach(dir => { const dirEl = $el("div.aiia-directory-item", { dataset: { itemName: dir.name }, textContent: dir.name, onclick: () => this.navigateTo(this.currentPath ? `${this.currentPath}/${dir.name}` : dir.name) }); this.directoryList.appendChild(dirEl); }); }

    renderContentPanel(files) {
        this.virtualList.lastRendered = { start: -1, end: -1 };
        log("[Virtual List] State reset.");
        this.contentArea.innerHTML = '';
        this.visibleItems = files;
        this.setFocus(null, null);

        if (this.iconView.resizeObserver) {
            this.iconView.resizeObserver.disconnect();
            this.iconView.resizeObserver = null;
        }

        this.iconView.activeNodes.forEach(node => this.iconView.nodePool.push(node));
        this.iconView.activeNodes.clear();
        this.listView.activeNodes.forEach(node => this.listView.nodePool.push(node));
        this.listView.activeNodes.clear();

        if (this.viewMode === 'icons') {
            this.renderIconsView(files);
        } else {
            this.renderListView(files);
        }
    }

    renderIconsView(files) {
        if (files.length === 0) {
            this.contentArea.appendChild($el('div.aiia-browser-placeholder', { textContent: 'No files in this directory.' }));
            return;
        }

        // By wrapping the entire setup in a setTimeout, we push it to the next event
        // loop cycle. This ensures the browser has finished painting the dialog and the
        // container dimensions are stable before any layout calculations are performed.
        // This is a robust way to solve race conditions.
        setTimeout(() => {
            if (this.viewMode !== 'icons') return; // In case the view changed while waiting

            this.virtualList.lastRendered = { start: -1, end: -1 };

            const scroller = $el("div.aiia-virtual-scroller");
            this.virtualList.scroller = scroller;

            const content = $el("div.aiia-grid-inner-wrapper", { style: { position: 'relative' } });
            this.virtualList.content = content;

            const viewport = $el("div.aiia-grid-wrapper", [scroller, content]);
            this.virtualList.viewport = viewport;

            this.iconView.resizeObserver = new ResizeObserver(() => {
                this.updateIconViewGeometry(files);
                this.renderVisibleIcons(files, true);
            });
            this.iconView.resizeObserver.observe(viewport);

            viewport.onscroll = () => {
                if (this.virtualList.renderRaf) cancelAnimationFrame(this.virtualList.renderRaf);
                this.virtualList.renderRaf = requestAnimationFrame(() => this.renderVisibleIcons(files));
            };

            this.contentArea.replaceChildren(viewport); // Use replaceChildren to be safe
        }, 0);
    }

    updateIconViewGeometry(files) {
        if (!this.virtualList.viewport) return;
        const { clientWidth } = this.virtualList.viewport;

        // Defensive guard against erroneous zero-width calculations during layout shifts.
        if (clientWidth === 0) return;

        const iconSize = Number(this.iconSize);
        const minMargin = 10;
        this.iconView.columns = Math.max(1, Math.floor(clientWidth / (iconSize + minMargin)));

        this.iconView.rowHeight = iconSize + 40;

        const totalIconWidth = this.iconView.columns * iconSize;
        const totalMarginWidth = clientWidth - totalIconWidth;
        this.iconView.actualMargin = this.iconView.columns > 1 ? totalMarginWidth / (this.iconView.columns - 1) : 0;

        const totalRows = Math.ceil(files.length / this.iconView.columns);
        const totalHeight = totalRows * this.iconView.rowHeight;
        this.virtualList.scroller.style.height = `${totalHeight}px`;
    }

    renderVisibleIcons(files, force = false) {
        if (!this.virtualList.viewport || !files || !this.iconView.columns) return;

        const { scrollTop, clientHeight } = this.virtualList.viewport;
        const { rowHeight, columns, colWidth } = this.iconView;
        const { buffer } = this.virtualList;

        const totalRows = Math.ceil(files.length / columns);
        const firstVisibleRow = Math.max(0, Math.floor(scrollTop / rowHeight) - buffer);
        const lastVisibleRow = Math.min(totalRows - 1, Math.ceil((scrollTop + clientHeight) / rowHeight) + buffer);

        if (!force && firstVisibleRow === this.virtualList.lastRendered.start && lastVisibleRow === this.virtualList.lastRendered.end) {
            return;
        }

        const firstItem = firstVisibleRow * columns;
        const lastItem = Math.min(files.length - 1, ((lastVisibleRow + 1) * columns) - 1);

        const newVisibleNames = new Set();
        for (let i = firstItem; i <= lastItem; i++) {
            if (files[i]) newVisibleNames.add(files[i].name);
        }

        // Recycle nodes that are no longer visible
        for (const [name, node] of this.iconView.activeNodes.entries()) {
            if (!newVisibleNames.has(name)) {
                if (this.iconLoadTimers.has(name)) {
                    clearTimeout(this.iconLoadTimers.get(name));
                    this.iconLoadTimers.delete(name);
                }
                const mediaContainer = node.querySelector('.aiia-icon-preview');
                if (mediaContainer) this.iconViewObserver.unobserve(mediaContainer);
                node.remove();
                this.iconView.nodePool.push(node);
                this.iconView.activeNodes.delete(name);
            }
        }

        // Render new visible items
        for (let i = firstItem; i <= lastItem; i++) {
            const item = files[i];
            if (!item || this.iconView.activeNodes.has(item.name)) continue;

            const node = this.iconView.nodePool.pop() || (() => {
                const newNode = $el('div.aiia-item-icon', { style: { position: 'absolute' } });
                const preview = $el('div.aiia-icon-preview');
                const buttonOverlay = $el('div.aiia-icon-button-overlay');
                const label = $el('div.aiia-item-label');
                newNode.replaceChildren(preview, buttonOverlay, label);
                return newNode;
            })();

            // Cleanup and setup
            const preview = node.querySelector('.aiia-icon-preview');
            preview.innerHTML = '';
            preview.className = 'aiia-icon-preview';

            // Positioning
            const row = Math.floor(i / columns);
            const col = i % columns;
            node.style.top = `${row * rowHeight}px`;
            node.style.left = `${col * (Number(this.iconSize) + this.iconView.actualMargin)}px`;
            node.style.width = `${this.iconSize}px`;
            node.style.height = `${this.iconView.rowHeight}px`;

            // Content
            node.dataset.itemName = item.name;
            const label = node.querySelector('.aiia-item-label');
            label.textContent = item.name;
            label.classList.toggle('has-workflow', !!item.has_workflow);

            let cachedEntry = this.iconView.mediaCache.get(item.name);

            // Invalidate cache if mtime is different
            if (cachedEntry && cachedEntry.mtime !== item.mtime) {
                cachedEntry = null;
                this.iconView.mediaCache.delete(item.name);
            }

            if (cachedEntry) {
                const mediaElement = cachedEntry.element.cloneNode(true);
                preview.appendChild(mediaElement);
                if (mediaElement.classList.contains('aiia-video-poster-container-flag')) {
                    preview.classList.add("aiia-video-poster-container");
                }
            } else {
                preview.classList.add('media-placeholder');
                preview.dataset.filename = item.name;
                preview.dataset.path = this.currentPath;
                preview.dataset.mtime = item.mtime;
                this.iconViewObserver.observe(preview);
            }

            const buttonOverlay = node.querySelector('.aiia-icon-button-overlay');
            const downloadButton = $el('button.aiia-download-button', { textContent: '📥', title: 'Download File' });

            if (item.has_workflow) {
                const loadButton = $el('button.aiia-load-workflow-button', { textContent: '🚀', title: 'Load Workflow' });
                buttonOverlay.replaceChildren(loadButton, downloadButton);
            } else {
                buttonOverlay.replaceChildren(downloadButton);
            }

            this.virtualList.content.appendChild(node);
            this.iconView.activeNodes.set(item.name, node);
        }

        this.virtualList.lastRendered = { start: firstVisibleRow, end: lastVisibleRow };
    }

    renderListView(files) {
        if (files.length === 0) {
            this.contentArea.appendChild($el('div.aiia-browser-placeholder', { textContent: 'No files in this directory.' }));
            return;
        }

        const handleHeaderClick = (e) => {
            const cell = e.target.closest('.aiia-list-header-cell');
            if (!cell || (this.metadataLoader.isLoading && ['dimensions', 'duration'].includes(cell.dataset.sort))) return;
            const key = cell.dataset.sort;
            if (!key) return;
            this.handleSortChange(key, this.sortKey === key ? (this.sortDir === 'asc' ? 'desc' : 'asc') : 'asc');
        };

        const headers = [
            { key: 'name', text: 'Name' },
            { key: 'mtime', text: 'Date modified' },
            { key: 'type', text: 'Type' },
            { key: 'size', text: 'Size' },
            { key: 'dimensions', text: 'Dimensions' },
            { key: 'duration', text: 'Duration' }
        ];

        const headerContainer = $el("div.aiia-list-header", { onclick: handleHeaderClick });
        this.updateListHeader(headerContainer, headers);

        const totalHeight = files.length * this.listView.rowHeight;
        const scroller = $el("div.aiia-virtual-scroller", { style: { height: `${totalHeight}px` } });
        this.virtualList.scroller = scroller;

        const content = $el("div.aiia-list-body-content");
        this.virtualList.content = content;

        const viewport = $el("div.aiia-list-body", [scroller, content]);
        this.virtualList.viewport = viewport;

        // --- START: Fixed Pool Implementation ---
        this.listView.nodePool = [];
        this.listView.activeNodes.clear(); // Clear any previous state
        const poolSize = Math.ceil(viewport.clientHeight / this.listView.rowHeight) + (this.virtualList.buffer * 2);

        for (let i = 0; i < poolSize; i++) {
            const row = $el("div.aiia-list-row");
            row.replaceChildren(
                $el("div.aiia-list-cell.aiia-list-cell-icon"),
                $el("div.aiia-list-cell"),
                $el("div.aiia-list-cell"),
                $el("div.aiia-list-cell"),
                $el("div.aiia-list-cell"),
                $el("div.aiia-list-cell"),
                $el("div.aiia-list-cell"),
                $el("div.aiia-list-cell.aiia-list-cell-actions", [
                    $el('button.aiia-download-button', { textContent: '📥', title: 'Download File' })
                ])
            );
            this.listView.nodePool.push(row);
        }
        // --- END: Fixed Pool Implementation ---

        viewport.onscroll = () => {
            if (this.virtualList.renderRaf) cancelAnimationFrame(this.virtualList.renderRaf);
            this.virtualList.renderRaf = requestAnimationFrame(() => this.renderVisibleRows());
        };

        const container = $el("div.aiia-list-container", [headerContainer, viewport]);
        this.contentArea.innerHTML = '';
        this.contentArea.appendChild(container);
        this.renderVisibleRows();
    }

    updateListHeader(headerContainer, headers) {
        headerContainer.replaceChildren($el("div.aiia-list-header-cell"), ...headers.map(h => {
            let headerText = h.text;
            if (h.key === this.sortKey && !(this.metadataLoader.isLoading && ['dimensions', 'duration'].includes(h.key))) {
                headerText += this.sortDir === 'asc' ? ' ▲' : ' ▼';
            }
            const cell = $el("div.aiia-list-header-cell", { textContent: headerText, dataset: { sort: h.key } });
            if (this.metadataLoader.isLoading && ['dimensions', 'duration'].includes(h.key)) {
                cell.style.opacity = 0.5;
                cell.style.cursor = 'not-allowed';
            }
            return cell;
        }));
    }

    renderVisibleRows(force = false) {
        if (!this.virtualList.viewport || !this.visibleItems) return;

        const { scrollTop, clientHeight } = this.virtualList.viewport;
        const { rowHeight, nodePool, activeNodes } = this.listView;
        const { buffer } = this.virtualList;

        const totalRows = this.visibleItems.length;
        const firstVisibleRow = Math.max(0, Math.floor(scrollTop / rowHeight) - buffer);
        const lastVisibleRow = Math.min(totalRows - 1, Math.ceil((scrollTop + clientHeight) / rowHeight) + buffer);

        if (!force && firstVisibleRow === this.virtualList.lastRendered.start && lastVisibleRow === this.virtualList.lastRendered.end) {
            return;
        }

        const newActiveNodes = new Map();
        const fragment = document.createDocumentFragment();

        for (let i = firstVisibleRow; i <= lastVisibleRow; i++) {
            const item = this.visibleItems[i];
            if (!item) continue;

            let node = activeNodes.get(item.name);
            if (node) {
                activeNodes.delete(item.name);
            } else {
                node = nodePool.pop() || (() => {
                    const newNode = $el("div.aiia-list-row");
                    newNode.replaceChildren(
                        $el("div.aiia-list-cell.aiia-list-cell-icon"),
                        $el("div.aiia-list-cell"),
                        $el("div.aiia-list-cell"),
                        $el("div.aiia-list-cell"),
                        $el("div.aiia-list-cell"),
                        $el("div.aiia-list-cell"),
                        $el("div.aiia-list-cell"),
                        $el("div.aiia-list-cell.aiia-list-cell-actions", [
                            $el('button.aiia-download-button', { textContent: '📥', title: 'Download File' })
                        ])
                    );
                    return newNode;
                })();
            }

            node.dataset.itemName = item.name;
            node.style.position = 'relative';
            node.style.transform = '';

            const mime = getMimeType(item.name);
            let icon = '📄';
            if (mime === 'image') icon = '🖼️';
            else if (mime === 'video') icon = '🎬';
            else if (mime === 'audio') icon = '🎵';

            let dimText = item.width && item.height ? `${item.width} x ${item.height}` : '—';
            let durText = item.duration != null ? formatDuration(item.duration) : '—';

            const cells = node.children;
            cells[0].textContent = icon;
            cells[1].textContent = item.name;
            cells[2].textContent = formatDate(item.mtime);
            cells[3].textContent = item.extension.replace('.', '');
            cells[4].textContent = formatBytes(item.size);
            cells[5].textContent = dimText;
            cells[6].textContent = durText;

            const actionsCell = cells[7];
            actionsCell.innerHTML = ''; // Clear previous buttons

            const downloadButton = $el('button.aiia-download-button', { textContent: '📥', title: 'Download File' });

            if (item.has_workflow) {
                const loadButton = $el('button.aiia-load-workflow-button', { textContent: '🚀', title: 'Load Workflow' });
                actionsCell.append(loadButton, downloadButton);
            } else {
                actionsCell.appendChild(downloadButton);
            }

            const globalIndex = this.directoryItems.length + i;
            if (globalIndex === this.currentFocusIndex) node.classList.add('focused');
            else node.classList.remove('focused');

            fragment.appendChild(node);
            newActiveNodes.set(item.name, node);
        }

        for (const node of activeNodes.values()) {
            nodePool.push(node);
        }

        this.virtualList.content.replaceChildren(fragment);
        this.listView.activeNodes = newActiveNodes;

        this.virtualList.content.style.transform = `translateY(${firstVisibleRow * rowHeight}px)`;
        this.virtualList.lastRendered = { start: firstVisibleRow, end: lastVisibleRow };
    }
    getAllNavigableItems() { const directories = this.hideFolders ? [] : this.directoryItems; return [...directories, ...this.visibleItems]; }
    async startMetadataBatchLoading() {
        return new Promise(async (resolve) => {
            if (this.metadataLoader.isLoading) {
                resolve();
                return;
            }

            const filesToProcess = this.itemsData.filter(item => {
                const mime = getMimeType(item.name);
                let needsMeta = false;
                if (mime === 'image') {
                    needsMeta = !item.hasOwnProperty('width');
                } else if (mime === 'video' || mime === 'audio') {
                    needsMeta = !item.hasOwnProperty('duration');
                }
                return item.type === 'file' && needsMeta;
            });

            const filesToFetch = [];
            for (const item of filesToProcess) {
                const cacheKey = getCacheKey(this.currentPath, item.name);
                const cachedData = getCache(cacheKey);
                if (cachedData && cachedData.mtime === item.mtime && cachedData.size === item.size) {
                    const { mtime, size, ...rest } = cachedData.metadata;
                    Object.assign(item, rest);
                } else {
                    filesToFetch.push(item);
                }
            }

            if (filesToFetch.length === 0) {
                resolve();
                return;
            }

            this.metadataLoader.controller = new AbortController();
            this.metadataLoader.isLoading = true;
            this.metadataLoader.path = this.currentPath;

            const headerContainer = this.element.querySelector('.aiia-list-header');
            if (headerContainer) {
                const headers = [
                    { key: 'name', text: 'Name' },
                    { key: 'mtime', text: 'Date modified' },
                    { key: 'type', text: 'Type' },
                    { key: 'size', text: 'Size' },
                    { key: 'dimensions', text: 'Dimensions' },
                    { key: 'duration', text: 'Duration' }
                ];
                this.updateListHeader(headerContainer, headers);
            }

            const total = filesToFetch.length;
            let loaded = 0;
            this.updateProgressBar(loaded, total);

            const BATCH_SIZE = 50;
            for (let i = 0; i < filesToFetch.length; i += BATCH_SIZE) {
                if (this.metadataLoader.controller.signal.aborted) break;

                while (this.metadataLoader.isPaused) {
                    await new Promise(resolveWait => setTimeout(resolveWait, 200));
                    if (this.metadataLoader.controller.signal.aborted) break;
                }
                if (this.metadataLoader.controller.signal.aborted) break;

                const batch = filesToFetch.slice(i, i + BATCH_SIZE);
                const filenames = batch.map(f => f.name);

                try {
                    const response = await fetch(window.location.origin + '/api/aiia/v1/browser/get_batch_metadata', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ path: this.currentPath, filenames }),
                        signal: this.metadataLoader.controller.signal,
                    });

                    if (!response.ok) throw new Error(`HTTP ${response.status}`);

                    const data = await response.json();
                    for (const [filename, metadata] of Object.entries(data)) {
                        const f = this.itemsDataMap.get(filename);
                        if (f) {
                            const { mtime, size, ...rest } = metadata;
                            Object.assign(f, rest);
                            const cacheKey = getCacheKey(this.currentPath, f.name);
                            setCache(cacheKey, { mtime: f.mtime, size: f.size, metadata: rest });
                        }
                    }

                } catch (error) {
                    if (error.name !== 'AbortError') {
                        console.error('[AIIA] Error fetching batch metadata:', error);
                    }
                }

                loaded += batch.length;
                this.updateProgressBar(loaded, total);
            }
            this.updateProgressBar(loaded, total, true);
            resolve();
        });
    }
    updateProgressBar(loaded, total, finished = false) {
        if (!this.progressContainer) return;
        if (total === 0 || loaded >= total || finished) {
            this.progressContainer.style.display = 'none';
            if (finished) {
                this.metadataLoader.isLoading = false;
                const headerContainer = this.element.querySelector('.aiia-list-header');
                if (headerContainer) {
                    const headers = [{ key: 'name', text: 'Name' }, { key: 'mtime', text: 'Date modified' }, { key: 'type', text: 'Type' }, { key: 'size', text: 'Size' }, { key: 'dimensions', text: 'Dimensions' }, { key: 'duration', 'text': 'Duration' }];
                    this.updateListHeader(headerContainer, headers);
                }
                // Use setTimeout to push the final render to the next event loop.
                // This is the most robust way to ensure the UI correctly reflects all
                // loaded metadata after the initial asynchronous display.
                setTimeout(() => this.render(), 0);
            }
            return;
        }
        this.progressContainer.style.display = 'flex';
        const percent = (loaded / total) * 100;
        this.progressBar.firstChild.style.width = `${percent}%`;
        this.progressText.textContent = `Analyzing metadata... (${loaded}/${total})`;
    }

    handleIconIntersection(entries) {
        const LOAD_DELAY = 100;
        for (const entry of entries) {
            const placeholder = entry.target;
            const filename = placeholder.dataset.filename;
            if (!filename) continue;

            if (entry.isIntersecting) {
                if (this.iconLoadTimers.has(filename)) continue;

                const timer = setTimeout(() => {
                    this.iconLoadTimers.delete(filename);

                    const fileData = this.itemsDataMap.get(filename);
                    if (!fileData) return;

                    let cachedEntry = this.iconView.mediaCache.get(filename);
                    let mediaElement;

                    // If the cached entry exists but the modification time is different, invalidate it.
                    if (cachedEntry && cachedEntry.mtime !== fileData.mtime) {
                        cachedEntry = null;
                        this.iconView.mediaCache.delete(filename);
                    }

                    if (cachedEntry) {
                        mediaElement = cachedEntry.element;
                    } else {
                        const createPlaceholder = (icon, text) => $el(`div.${text === "Empty File" ? "aiia-empty-file-placeholder" : "aiia-no-preview-placeholder"}`, [$el("span", { textContent: icon }), $el("span", { textContent: text })]);

                        if (fileData.size === 0) {
                            mediaElement = createPlaceholder("🚫", "Empty File");
                        } else {
                            const path = placeholder.dataset.path;
                            const mtime = placeholder.dataset.mtime;
                            const mime = getMimeType(filename);

                            const handleMediaError = (e) => {
                                const newPlaceholder = createPlaceholder("🖼️🚫", "No Preview");
                                if (e.target.parentNode) e.target.parentNode.replaceChild(newPlaceholder, e.target);
                                this.iconView.mediaCache.set(filename, { element: newPlaceholder.cloneNode(true), mtime: fileData.mtime });
                            };

                            if (mime === 'image') {
                                const url = getThumbnailUrl(filename, path, mtime);
                                mediaElement = $el("img.aiia-image-preview", { src: url, onerror: handleMediaError });
                            } else if (mime === 'video') {
                                const url = this.enableVideoPreviews ? getFileUrl(filename, path) : getVideoPosterUrl(filename, path, mtime);
                                if (this.enableVideoPreviews) {
                                    mediaElement = $el("video.aiia-video-preview", { src: url, autoplay: true, muted: true, loop: true, playsinline: true, onerror: handleMediaError });
                                } else {
                                    mediaElement = $el("img.aiia-image-preview", { src: url, onerror: handleMediaError });
                                    mediaElement.classList.add("aiia-video-poster-container-flag");
                                }
                            } else if (mime === 'audio') {
                                mediaElement = createPlaceholder("🎵", fileData.duration ? formatDuration(fileData.duration) : 'Audio');
                            } else {
                                mediaElement = createPlaceholder("❓", "Unknown");
                            }
                        }
                        this.iconView.mediaCache.set(filename, { element: mediaElement.cloneNode(true), mtime: fileData.mtime });
                    }

                    if (placeholder.parentNode) {
                        placeholder.className = 'aiia-icon-preview';
                        if (mediaElement.classList.contains('aiia-video-poster-container-flag')) {
                            placeholder.classList.add("aiia-video-poster-container");
                        }
                        placeholder.replaceChildren(mediaElement);
                    }

                }, LOAD_DELAY);
                this.iconLoadTimers.set(filename, timer);
            } else {
                if (this.iconLoadTimers.has(filename)) {
                    clearTimeout(this.iconLoadTimers.get(filename));
                    this.iconLoadTimers.delete(filename);
                }
            }
        }
    }

    ensureFocusedItemVisible() { const item = this.getAllNavigableItems()[this.currentFocusIndex]; if (!item) return; if (item.type === 'directory') { const el = this.element.querySelector(`.aiia-directory-item[data-item-name="${CSS.escape(item.name)}"]`); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } else { if (this.viewMode === 'list' && this.virtualList.viewport) { const fileIndex = this.visibleItems.findIndex(f => f.name === item.name); if (fileIndex === -1) return; const { scrollTop, clientHeight } = this.virtualList.viewport; const { rowHeight } = this.listView; const itemTop = fileIndex * rowHeight, itemBottom = itemTop + rowHeight; if (itemTop < scrollTop) this.virtualList.viewport.scrollTop = itemTop; else if (itemBottom > scrollTop + clientHeight) this.virtualList.viewport.scrollTop = itemBottom - clientHeight; } else if (this.viewMode === 'icons' && this.virtualList.viewport) { const fileIndex = this.visibleItems.findIndex(f => f.name === item.name); if (fileIndex === -1) return; const { scrollTop, clientHeight } = this.virtualList.viewport; const { rowHeight, columns } = this.iconView; const itemRow = Math.floor(fileIndex / columns); const itemTop = itemRow * rowHeight; const itemBottom = itemTop + rowHeight; if (itemTop < scrollTop) { this.virtualList.viewport.scrollTop = itemTop; } else if (itemBottom > scrollTop + clientHeight) { this.virtualList.viewport.scrollTop = itemBottom - clientHeight; } } } }

    moveFocus(key) {
        const allNavigableItems = this.getAllNavigableItems();
        if (allNavigableItems.length === 0) return;

        let newIndex = this.currentFocusIndex;
        const originalIndex = newIndex;

        const dirCount = this.directoryItems.length;
        const fileCount = this.visibleItems.length;

        if (newIndex === -1) {
            newIndex = 0;
        } else {
            const isDir = newIndex < dirCount;
            const fileIndex = isDir ? -1 : newIndex - dirCount;

            let columns = 1;
            if (this.viewMode === 'icons') {
                columns = this.iconView.columns || 1;
            }

            switch (key) {
                case 'ArrowUp':
                    if (isDir) newIndex = (newIndex > 0) ? newIndex - 1 : dirCount - 1;
                    else {
                        if (this.viewMode === 'list') newIndex = (fileIndex > 0) ? newIndex - 1 : allNavigableItems.length - 1;
                        else {
                            if (fileIndex >= columns) newIndex -= columns;
                            else {
                                const lastRowCol = fileIndex % columns;
                                const lastRowFileCount = fileCount % columns || columns;
                                const lastRowFirstFileIndex = fileCount - lastRowFileCount;
                                newIndex = dirCount + lastRowFirstFileIndex + Math.min(lastRowCol, lastRowFileCount - 1);
                            }
                        }
                    }
                    break;

                case 'ArrowDown':
                    if (isDir) newIndex = (newIndex < dirCount - 1) ? newIndex + 1 : 0;
                    else {
                        if (this.viewMode === 'list') newIndex = (fileIndex < fileCount - 1) ? newIndex + 1 : dirCount;
                        else {
                            const targetIndex = newIndex + columns;
                            if (targetIndex < allNavigableItems.length) {
                                newIndex = targetIndex;
                            } else {
                                const currentRow = Math.floor(fileIndex / columns);
                                const totalRows = Math.ceil(fileCount / columns);
                                if (currentRow < totalRows - 1) {
                                    newIndex = allNavigableItems.length - 1;
                                } else {
                                    newIndex = dirCount + (fileIndex % columns);
                                    if (newIndex >= allNavigableItems.length) newIndex = allNavigableItems.length - 1;
                                }
                            }
                        }
                    }
                    break;

                case 'ArrowLeft':
                    if (isDir) newIndex = 0;
                    else {
                        if (this.viewMode === 'list') {
                            if (dirCount > 0) newIndex = this.lastFocus.dir !== -1 ? this.lastFocus.dir : 0;
                        } else {
                            if (fileIndex > 0) newIndex--;
                            else if (dirCount > 0) newIndex = this.lastFocus.dir !== -1 ? this.lastFocus.dir : 0;
                            else newIndex = allNavigableItems.length - 1;
                        }
                    }
                    break;

                case 'ArrowRight':
                    if (isDir) {
                        if (fileCount > 0) newIndex = this.lastFocus.file !== -1 && this.lastFocus.file >= dirCount ? this.lastFocus.file : dirCount;
                    } else {
                        if (this.viewMode === 'list') {
                            newIndex = allNavigableItems.length - 1;
                        } else {
                            if (fileIndex < fileCount - 1) newIndex++;
                            else newIndex = dirCount;
                        }
                    }
                    break;
            }
        }

        newIndex = Math.max(0, Math.min(allNavigableItems.length - 1, newIndex));

        if (newIndex !== originalIndex) {
            const newItem = allNavigableItems[newIndex];
            this.setFocus(newItem, null, true, 'keyboard');
        }
    }

    activateFocusedItem() { const item = this.getAllNavigableItems()[this.currentFocusIndex]; if (!item) return; if (item.type === 'directory') { const el = this.element.querySelector('.focused'); if (el) el.click(); } else { const itemEl = this.element.querySelector('.focused'); if (itemEl) itemEl.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true, view: window })); } }
    toggleTooltipForFocusedItem() {
        if (this.currentFocusIndex === -1) return;

        const allNavigableItems = this.getAllNavigableItems();
        const itemData = allNavigableItems[this.currentFocusIndex];
        const itemEl = this.element.querySelector('.focused');

        if (!itemEl) return;

        // Toggle the actions-visible class for keyboard users
        const isActionsVisible = itemEl.classList.toggle('actions-visible');

        if (this.isTooltipVisible && this.tooltipItemName === itemData.name) {
            this.hideTooltip();
            // If actions are not visible, it means we are closing everything, so remove focus.
            if (!isActionsVisible) {
                itemEl.classList.remove('focused');
                this.currentFocusIndex = -1;
            }
        } else {
            clearTimeout(this.tooltipTimeout);
            this.hideTooltip();
            const rect = itemEl.getBoundingClientRect();
            const fakeEvent = { clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2 };
            this.updateTooltipContent(itemData, fakeEvent);
        }
    }

    handleKeyDown(e) {
        const target = e.target;

        if (target.dataset.escapeToList) {
            if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                e.preventDefault();
                e.stopPropagation();

                if (target === this.pathInput) this.hidePathInput();

                const focusIndex = this.lastFocus.file !== -1 ? this.lastFocus.file : (this.directoryItems.length > 0 ? (this.lastFocus.dir !== -1 ? this.lastFocus.dir : 0) : -1);

                if (focusIndex !== -1) {
                    const item = this.getAllNavigableItems()[focusIndex];
                    if (item) this.setFocus(item, null, true, 'keyboard');
                } else if (this.getAllNavigableItems().length > 0) {
                    this.setFocus(this.getAllNavigableItems()[0], null, true, 'keyboard');
                }

                this.element.focus({ preventScroll: true });
                return;
            }

            if (target === this.sizeSlider && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
                e.preventDefault();
                e.stopPropagation();
                const step = parseInt(this.sizeSlider.step) || 1;
                const dir = e.key === 'ArrowLeft' ? -1 : 1;
                this.sizeSlider.value = parseInt(this.sizeSlider.value) + (step * dir);
                this.sizeSlider.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
                return;
            }
            return;
        }

        if ((target.tagName === 'INPUT' && target !== this.pathInput) || target.tagName === 'SELECT') return;

        const keyMap = { 'ArrowUp': 'moveFocus', 'ArrowDown': 'moveFocus', 'ArrowLeft': 'moveFocus', 'ArrowRight': 'moveFocus', 'Enter': 'activateFocusedItem', ' ': 'toggleTooltipForFocusedItem' };
        if (keyMap[e.key]) {
            e.preventDefault(); e.stopPropagation();
            this.isKeyboardNavigating = true;
            if (keyMap[e.key] === 'moveFocus') this.moveFocus(e.key);
            else this[keyMap[e.key]]();
        }
    }

    show() {
        this.currentPath = null;
        this.history = [];
        this.navigateTo("");
        super.show();
        this.updateTooltipSizeLimits();
        this.element.focus({ preventScroll: true });
    }
}

app.registerExtension({
    name: "Comfy.AIIA.Browser",
    setup() {
        // --- START: One-time cache cleanup ---
        try {
            const migrationKey = "aiia-cache-v3-migrated";
            if (!localStorage.getItem(migrationKey)) {
                console.log("[AIIA Browser] Checking for old cache entries to migrate...");
                const keysToRemove = [];
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    if (key.startsWith("aiia-cache:") && !key.startsWith("aiia-cache:v3:")) {
                        keysToRemove.push(key);
                    }
                }
                if (keysToRemove.length > 0) {
                    console.log(`[AIIA Browser] Removing ${keysToRemove.length} old cache entries...`);
                    keysToRemove.forEach(key => localStorage.removeItem(key));
                }
                localStorage.setItem(migrationKey, "true");
                console.log("[AIIA Browser] Cache migration check complete.");
            }
        } catch (e) {
            console.error("[AIIA Browser] Error during cache migration:", e);
        }
        // --- END: One-time cache cleanup ---

        document.head.appendChild($el("style", { textContent: browserStyles }));
        let browserDialog = null;
        const createDialog = () => { if (!browserDialog) browserDialog = new AIIABrowserDialog(); browserDialog.show(); };
        document.addEventListener('keydown', (e) => {
            if (!browserDialog || browserDialog.element.style.display === 'none') return;
            if (browserDialog.fullscreenViewer && browserDialog.fullscreenViewer.element.style.display !== 'none') return;
            if (!browserDialog.element.contains(document.activeElement)) {
                const isNavKey = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Enter', ' '].includes(e.key);
                if (isNavKey) {
                    e.preventDefault();
                    browserDialog.element.focus({ preventScroll: true });
                    if (browserDialog.currentFocusIndex === -1 && e.key !== ' ') {
                        const allItems = browserDialog.getAllNavigableItems();
                        if (allItems.length > 0) browserDialog.setFocus(allItems[0], browserDialog.element.querySelector('[data-item-name]'), true, 'keyboard');
                    } else {
                        browserDialog.handleKeyDown(e);
                    }
                }
            }
        });
        const menuButton = $el("button", { id: "aiia-browser-menu-button", textContent: "AIIA Browser", onclick: createDialog });
        app.menu.element.appendChild(menuButton);
    }
});