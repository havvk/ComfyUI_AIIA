// 文件: aiia_fullscreen_viewer.js (V2 - 增加滑动和滚动导航)

import { $el } from "../../scripts/ui.js";
import { api } from "../../scripts/api.js";

// Helper functions
function getFileUrl(filename, path = "") { const subfolder = path || ""; return api.apiURL(`/view?type=output&subfolder=${encodeURIComponent(subfolder)}&filename=${encodeURIComponent(filename)}`); }
function formatBytes(bytes, decimals = 2) { if (bytes === 0) return '0 Bytes'; const k = 1024; const dm = decimals < 0 ? 0 : decimals; const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB']; const i = Math.floor(Math.log(bytes) / Math.log(k)); return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]; }
function formatDate(timestamp) { const date = new Date(timestamp * 1000); return date.toLocaleString(); }
function formatDuration(seconds) { if (isNaN(seconds)) return '00:00'; const min = Math.floor(seconds / 60); const sec = Math.floor(seconds % 60); return `${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`; }
function getMimeType(filename) { if (typeof filename !== 'string') return 'unknown'; const extension = filename.split('.').pop().toLowerCase(); if (['png', 'jpg', 'jpeg', 'webp', 'gif'].includes(extension)) return 'image'; if (['mp4', 'webm', 'mov', 'avi'].includes(extension)) return 'video'; if (['mp3', 'wav', 'ogg'].includes(extension)) return 'audio'; return 'unknown'; }

export class AIIAFullscreenViewer {
    constructor() {
        this.items = [];
        this.currentIndex = -1;
        this.currentPath = "";
        this.onCloseCallback = null;
        this.filmstripObserver = null;
        this.loadMediaTimeout = null;

        // V2 优化: 增加滑动和滚动相关的状态
        this.longPressTimeout = null;
        this.longPressInterval = null;
        this.swipeState = {
            isDragging: false,
            startX: 0,
            threshold: 50, // 至少滑动50px才算有效
        };
        this.scrollDebounce = {
            timeout: null,
            delay: 150, // 滚动事件的防抖延迟
        };

        this.element = $el("div.aiia-fullscreen-viewer");
        this.element.setAttribute('tabindex', -1);

        this.mediaContainer = $el("div.aiia-fullscreen-media-container");
        this.closeButton = $el("button.aiia-fullscreen-close-button", { textContent: "✖" });
        this.prevButton = $el("button.aiia-fullscreen-nav.prev", { textContent: "❮" });
        this.nextButton = $el("button.aiia-fullscreen-nav.next", { textContent: "❯" });
        this.filmstripContainer = $el("div.aiia-filmstrip-container");
        this.infoPanel = $el("div.aiia-info-panel");

        const mainContent = $el("div.aiia-fullscreen-main-content", [this.prevButton, this.mediaContainer, this.nextButton]);
        this.element.append(mainContent, this.infoPanel, this.filmstripContainer, this.closeButton);
        document.body.appendChild(this.element);

        this.bindEvents();
    }

    bindEvents() {
        this.closeButton.onclick = () => this.hide();

        const setupNavButtonEvents = (button, direction) => {
            const clearTimers = () => {
                clearTimeout(this.longPressTimeout);
                clearInterval(this.longPressInterval);
                this.longPressTimeout = null;
                this.longPressInterval = null;
            };

            button.addEventListener("mousedown", (e) => {
                e.preventDefault();

                if (direction === 'prev') this.prev();
                else if (direction === 'next') this.next();

                this.longPressTimeout = setTimeout(() => {
                    this.longPressInterval = setInterval(() => {
                        if (direction === 'prev') this.prev();
                        else if (direction === 'next') this.next();
                    }, 150);
                }, 500);
            });

            button.addEventListener("mouseup", clearTimers);
            button.addEventListener("mouseleave", clearTimers);

            window.addEventListener("blur", clearTimers);
        };

        setupNavButtonEvents(this.prevButton, 'prev');
        setupNavButtonEvents(this.nextButton, 'next');

        // V2 优化: 增加滑动事件监听
        const handleSwipeStart = (e) => {
            if (e.target.tagName === 'VIDEO' || e.target.tagName === 'AUDIO') return; // 不在视频/音频控件上触发滑动
            e.preventDefault();
            this.swipeState.isDragging = true;
            this.swipeState.startX = e.clientX;
            this.mediaContainer.style.cursor = 'grabbing';
        };

        const handleSwipeEnd = (e) => {
            if (!this.swipeState.isDragging) return;
            this.swipeState.isDragging = false;
            this.mediaContainer.style.cursor = 'grab';

            const deltaX = e.clientX - this.swipeState.startX;
            if (Math.abs(deltaX) > this.swipeState.threshold) {
                if (deltaX > 0) {
                    this.prev();
                } else {
                    this.next();
                }
            }
        };

        this.mediaContainer.addEventListener('mousedown', handleSwipeStart);
        this.mediaContainer.addEventListener('mouseup', handleSwipeEnd);
        this.mediaContainer.addEventListener('mouseleave', () => {
            if (this.swipeState.isDragging) {
                this.swipeState.isDragging = false;
                this.mediaContainer.style.cursor = 'grab';
            }
        });

        // V2 优化: 增加滚轮事件监听
        this.element.addEventListener('wheel', (e) => {
            e.preventDefault();
            e.stopPropagation();

            if (this.scrollDebounce.timeout) return;

            this.scrollDebounce.timeout = setTimeout(() => {
                this.scrollDebounce.timeout = null;
            }, this.scrollDebounce.delay);

            if (e.deltaY < 0) {
                this.navigateVertical('up');
            } else if (e.deltaY > 0) {
                this.navigateVertical('down');
            }
        });

        this.handleKeyDown = (e) => {
            if (this.element.style.display === 'none') return;
            switch (e.key) {
                case "Escape": e.preventDefault(); e.stopPropagation(); this.hide(); break;
                case "ArrowLeft": e.preventDefault(); this.prev(); break;
                case "ArrowRight": e.preventDefault(); this.next(); break;
                // V2 优化: 增加上下方向键导航
                case "ArrowUp": e.preventDefault(); this.navigateVertical('up'); break;
                case "ArrowDown": e.preventDefault(); this.navigateVertical('down'); break;
                case " ": e.preventDefault(); this.infoPanel.style.display = this.infoPanel.style.display === 'none' ? '' : 'none'; break;
            }
        };
        this.element.addEventListener('keydown', this.handleKeyDown);
    }

    // V2 优化: 新增垂直导航方法
    navigateVertical(direction) {
        const thumb = this.filmstripContainer.querySelector('.aiia-filmstrip-thumb');
        if (!thumb) return;

        // 计算filmstrip一行可以显示多少个缩略图
        const thumbStyle = getComputedStyle(thumb);
        const containerStyle = getComputedStyle(this.filmstripContainer);
        const thumbWidth = thumb.offsetWidth + parseFloat(thumbStyle.marginLeft) + parseFloat(thumbStyle.marginRight);
        const gap = parseFloat(containerStyle.gap) || 10;
        const totalThumbWidth = thumbWidth + gap;
        const columns = Math.max(1, Math.floor(this.filmstripContainer.clientWidth / totalThumbWidth));

        let newIndex = this.currentIndex;
        if (direction === 'up') {
            newIndex -= columns;
        } else if (direction === 'down') {
            newIndex += columns;
        }

        // 确保新索引在有效范围内
        newIndex = Math.max(0, Math.min(this.items.length - 1, newIndex));

        if (newIndex !== this.currentIndex) {
            this.jumpTo(newIndex);
        }
    }

    show(items, currentIndex, currentPath, onCloseCallback = null) {
        this.items = items;
        this.currentIndex = currentIndex;
        this.currentPath = currentPath;
        this.onCloseCallback = onCloseCallback;
        if (this.currentIndex === -1) return;

        this.mediaContainer.style.cursor = 'grab'; // V2 优化: 设置初始鼠标手势

        this.renderFilmstrip();
        this.element.style.display = "flex";
        this.element.focus({ preventScroll: true });

        this.loadMedia();
    }

    hide() {
        if (this.onCloseCallback) {
            this.onCloseCallback();
        }
        this.element.style.display = "none";
        this.mediaContainer.innerHTML = "";
        this.filmstripContainer.innerHTML = "";
        if (this.filmstripObserver) this.filmstripObserver.disconnect();
        clearTimeout(this.loadMediaTimeout);
        clearTimeout(this.longPressTimeout);
        clearInterval(this.longPressInterval);
        clearTimeout(this.scrollDebounce.timeout); // V2 优化: 清除滚动计时器
    }

    renderFilmstrip() {
        this.filmstripContainer.innerHTML = "";
        if (this.filmstripObserver) this.filmstripObserver.disconnect();
        this.filmstripObserver = new IntersectionObserver(this.handleFilmstripIntersection.bind(this), {
            root: this.filmstripContainer,
            threshold: 0.1
        });

        this.items.forEach((item, index) => {
            const thumb = $el("div.aiia-filmstrip-thumb", {
                dataset: { index, filename: item.name },
                onclick: () => this.jumpTo(index)
            });
            const mime = getMimeType(item.name);
            const thumbContent = $el("div.aiia-filmstrip-thumb-content");
            thumb.appendChild(thumbContent);

            if (mime === 'video') {
                thumbContent.innerHTML = '<div class="aiia-filmstrip-placeholder-icon">🎬</div>';
                thumb.dataset.mediaType = 'video';
            } else if (mime === 'image') {
                thumbContent.innerHTML = '<div class="aiia-filmstrip-placeholder"></div>';
                thumb.dataset.mediaType = 'image';
            } else {
                thumbContent.innerHTML = '<div class="aiia-filmstrip-placeholder-icon">🎵</div>';
                thumb.dataset.mediaType = 'audio';
            }

            this.filmstripContainer.appendChild(thumb);
            if (mime === 'image' || mime === 'video') {
                this.filmstripObserver.observe(thumb);
            }
        });
    }

    handleFilmstripIntersection(entries) {
        for (const entry of entries) {
            if (entry.isIntersecting) {
                const thumb = entry.target;
                const mediaType = thumb.dataset.mediaType;
                const filename = thumb.dataset.filename;
                const thumbContent = thumb.querySelector('.aiia-filmstrip-thumb-content');
                const url = getFileUrl(filename, this.currentPath);
                let mediaElement;

                if (mediaType === 'image') {
                    mediaElement = $el("img", { src: url });
                } else if (mediaType === 'video') {
                    mediaElement = $el("video", {
                        src: url, autoplay: true, muted: true, loop: true, playsinline: true
                    });
                }

                if (mediaElement) {
                    thumbContent.replaceChildren(mediaElement);
                    this.filmstripObserver.unobserve(thumb);
                }
            }
        }
    }

    updateInfoPanel(item, extra = {}) {
        this.infoPanel.innerHTML = "";
        const createMetaRow = (label, value) => {
            if (value === undefined || value === null || value === '—' || value === '') return;
            this.infoPanel.appendChild($el("div.aiia-info-panel-row", [
                $el("span.aiia-info-panel-label", { textContent: label }),
                $el("span.aiia-info-panel-value", { textContent: value })
            ]));
        };

        createMetaRow("Name", item.name);
        createMetaRow("Date", formatDate(item.mtime));
        createMetaRow("Type", item.extension.replace('.', '').toUpperCase());
        createMetaRow("Size", formatBytes(item.size));

        const dimensions = extra.dimensions || (item.width && item.height ? `${item.width} x ${item.height}` : null);
        if (dimensions) createMetaRow("Dimensions", dimensions);

        if (extra.duration) createMetaRow("Duration", extra.duration);
    }

    loadMedia() {
        if (this.currentIndex < 0 || this.currentIndex >= this.items.length) return;
        const item = this.items[this.currentIndex];
        this.mediaContainer.innerHTML = "";

        const url = getFileUrl(item.name, this.currentPath);
        const mime = getMimeType(item.name);

        this.updateInfoPanel(item);

        let mediaElement;
        if (mime === 'image') {
            mediaElement = $el("img", { src: url });
        } else if (mime === 'video') {
            mediaElement = $el("video", { src: url, controls: true, autoplay: true, loop: true });
            mediaElement.onloadedmetadata = () => {
                const dimensions = `${mediaElement.videoWidth} x ${mediaElement.videoHeight}`;
                this.updateInfoPanel(item, { dimensions, duration: formatDuration(mediaElement.duration) });
            };
        } else if (mime === 'audio') {
            mediaElement = $el("div.aiia-fullscreen-audio-wrapper", [
                $el("div.aiia-fullscreen-audio-icon", { textContent: "🎵" }),
                $el("audio", {
                    src: url, controls: true, autoplay: true,
                    onloadedmetadata: (e) => this.updateInfoPanel(item, { duration: formatDuration(e.target.duration) })
                })
            ]);
        } else {
            mediaElement = $el("div", { textContent: `Unsupported file type: ${item.name}` });
        }

        this.mediaContainer.appendChild(mediaElement);
        this.updateNavButtons();
        this.updateFilmstripHighlight();
    }

    updateFilmstripHighlight() {
        const thumbs = this.filmstripContainer.children;
        for (let i = 0; i < thumbs.length; i++) {
            thumbs[i].classList.toggle("active", i === this.currentIndex);
        }
        const activeThumb = this.filmstripContainer.querySelector('.active');
        if (activeThumb) {
            activeThumb.scrollIntoView({ behavior: 'auto', block: 'nearest', inline: 'center' });
        }
    }

    jumpTo(index) {
        if (index >= 0 && index < this.items.length) {
            this.currentIndex = index;
            clearTimeout(this.loadMediaTimeout);
            this.loadMedia();
        }
    }

    navigate(direction) {
        let newIndex = this.currentIndex;
        if (direction === 'prev' && this.currentIndex > 0) newIndex--;
        else if (direction === 'next' && this.currentIndex < this.items.length - 1) newIndex++;
        else return;

        if (newIndex === this.currentIndex) return;
        this.currentIndex = newIndex;

        this.updateNavButtons();
        this.updateFilmstripHighlight();
        clearTimeout(this.loadMediaTimeout);
        this.loadMediaTimeout = setTimeout(() => this.loadMedia(), 150);
    }

    prev() { this.navigate('prev'); }
    next() { this.navigate('next'); }

    updateNavButtons() {
        this.prevButton.disabled = this.currentIndex <= 0;
        this.nextButton.disabled = this.currentIndex >= this.items.length - 1;
    }

    destroy() {
        this.element.removeEventListener('keydown', this.handleKeyDown);
        this.element.remove();
    }
}