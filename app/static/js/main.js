const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('canvas-container');
const roiListEl = document.getElementById('roiList');
const statusEl = document.getElementById('statusText');
const timelineSlider = document.getElementById('timelineSlider');
const currentFrameLabel = document.getElementById('currentFrame');
const totalFramesLabel = document.getElementById('totalFrames');

// State
let config = { rois: [] };
let videoInfo = { total_frames: 100, width: 1920, height: 1080, fps: 30 };
let currentFrameIndex = 0;
let image = new Image();
let isDrawing = false;
let currentPoints = [];
let selectedRoiIndex = -1;
let draggingPointIndex = -1;
let isPanning = false;
let startPanX = 0;
let startPanY = 0;

// View Transform
let transform = {
    scale: 1,
    x: 0,
    y: 0
};

// Tools: 'pointer', 'polygon', 'rectangle'
let currentTool = 'pointer';

// History
let history = [];
let historyIndex = -1;

// Initialization
async function init() {
    try {
        // Fetch Video Info
        const infoRes = await fetch('/video_info');
        videoInfo = await infoRes.json();

        timelineSlider.max = videoInfo.total_frames - 1;
        totalFramesLabel.innerText = videoInfo.total_frames;

        // Fetch Config
        const configRes = await fetch('/config');
        config = await configRes.json();
        if (!config.rois) config.rois = [];

        addToHistory();
        updateRoiList();
        loadFrame();

        // Auto-save
        setInterval(saveConfig, 30000);

        window.addEventListener('resize', fitCanvasToContainer);

    } catch (e) {
        console.error("Init failed", e);
        statusEl.innerText = "Error initializing application";
    }
}

// Canvas & View Logic
function fitCanvasToContainer() {
    // Initial fit, can be overridden by zoom
    canvas.width = videoInfo.width;
    canvas.height = videoInfo.height;
    // Don't reset view here on every resize to avoid jumping, 
    // but maybe we should if it's a major layout change?
    // Let's just render.
    render();
}

function setTransform(scale, x, y) {
    transform.scale = scale;
    transform.x = x;
    transform.y = y;
    canvas.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
}

function getFitScale() {
    const rect = container.getBoundingClientRect();
    const scaleX = rect.width / canvas.width;
    const scaleY = rect.height / canvas.height;
    return Math.min(scaleX, scaleY) * 0.9; // 90% fit
}

function zoom(delta, centerX, centerY) {
    const zoomFactor = 1.1;
    const newScale = delta > 0 ? transform.scale * zoomFactor : transform.scale / zoomFactor;

    // Limit zoom
    const fitScale = getFitScale();
    const minScale = fitScale / 2; // Limit zoom out to half the fit size (2x smaller than fit)

    if (newScale < minScale || newScale > 10) return;

    // Calculate new position to keep center stable
    const rect = container.getBoundingClientRect();
    const cx = centerX !== undefined ? centerX : rect.width / 2;
    const cy = centerY !== undefined ? centerY : rect.height / 2;

    const newX = cx - (cx - transform.x) * (newScale / transform.scale);
    const newY = cy - (cy - transform.y) * (newScale / transform.scale);

    setTransform(newScale, newX, newY);
}

function resetView() {
    // Fit to container
    const rect = container.getBoundingClientRect();

    // Use canvas dimensions (which should match image/video)
    const fitScale = getFitScale();

    // Center
    const x = (rect.width - canvas.width * fitScale) / 2 + 300;
    const y = (rect.height - canvas.height * fitScale) / 2 + 250;

    setTransform(fitScale, x, y);
}

// Frame Logic
function loadFrame() {
    image.src = `/frame?index=${currentFrameIndex}&t=${Date.now()}`;
    currentFrameLabel.innerText = currentFrameIndex;
    timelineSlider.value = currentFrameIndex;
}

image.onload = function () {
    // If first load or dimensions changed
    if (canvas.width !== image.width || canvas.height !== image.height) {
        canvas.width = image.width;
        canvas.height = image.height;
        resetView();
    }
    render();
};

function setFrame(index) {
    currentFrameIndex = parseInt(index);
    loadFrame();
}

// Drawing & Rendering
function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);

    // Draw ROIs
    config.rois.forEach((roi, index) => {
        drawPolygon(roi.points, roi.color || '#00ff00', index === selectedRoiIndex);
    });

    // Draw current drawing
    if (currentPoints.length > 0) {
        drawPolygon(currentPoints, '#ffff00', true, false);
    }
}

function drawPolygon(points, color, isSelected, isClosed = true) {
    if (points.length === 0) return;

    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
    }
    if (isClosed) ctx.closePath();

    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.globalAlpha = 0.2;
    ctx.fill();
    ctx.globalAlpha = 1.0;

    // Draw handles if selected
    if (isSelected) {
        ctx.fillStyle = '#fff';
        points.forEach(p => {
            const size = 8 / transform.scale; // Scale handles inversely so they stay constant size visually
            ctx.fillRect(p[0] - size / 2, p[1] - size / 2, size, size);
        });
    }
}

// Interaction
container.addEventListener('mousedown', (e) => {
    // Middle click or Alt+Click -> Pan
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
        startPan(e);
        return;
    }

    if (e.button !== 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / transform.scale;
    const y = (e.clientY - rect.top) / transform.scale;

    if (currentTool === 'pointer') {
        // Selection Logic
        if (selectedRoiIndex !== -1) {
            // Check handles
            const roi = config.rois[selectedRoiIndex];
            const handleSize = 10 / transform.scale;
            for (let i = 0; i < roi.points.length; i++) {
                const p = roi.points[i];
                if (Math.abs(p[0] - x) < handleSize && Math.abs(p[1] - y) < handleSize) {
                    draggingPointIndex = i;
                    return;
                }
            }
        }

        // Select ROI
        let found = false;
        for (let i = config.rois.length - 1; i >= 0; i--) {
            if (isPointInPoly([x, y], config.rois[i].points)) {
                selectedRoiIndex = i;
                found = true;
                break;
            }
        }

        if (!found) {
            selectedRoiIndex = -1;
            // If clicked on empty space in pointer mode, start panning
            startPan(e);
        }
        updateRoiList();
        render();

    } else if (currentTool === 'polygon') {
        isDrawing = true;
        currentPoints.push([x, y]);

        if (currentPoints.length === 4) {
            finishDrawing();
        } else {
            render();
        }

    } else if (currentTool === 'rectangle') {
        isDrawing = true;
        currentPoints = [[x, y]]; // Start point
    }
});

function startPan(e) {
    isPanning = true;
    startPanX = e.clientX - transform.x;
    startPanY = e.clientY - transform.y;
    container.classList.add('panning');
}

container.addEventListener('mousemove', (e) => {
    if (isPanning) {
        const newX = e.clientX - startPanX;
        const newY = e.clientY - startPanY;
        setTransform(transform.scale, newX, newY);
        return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / transform.scale;
    const y = (e.clientY - rect.top) / transform.scale;

    if (draggingPointIndex !== -1 && selectedRoiIndex !== -1) {
        config.rois[selectedRoiIndex].points[draggingPointIndex] = [x, y];
        render();
    } else if (isDrawing) {
        if (currentTool === 'rectangle') {
            // Update rectangle
            const start = currentPoints[0];
            // 4 points: start, top-right, bottom-right, bottom-left
            currentPoints = [
                start,
                [x, start[1]],
                [x, y],
                [start[0], y]
            ];
            render();
        } else if (currentTool === 'polygon') {
            render();
            // Elastic line
            if (currentPoints.length > 0) {
                const last = currentPoints[currentPoints.length - 1];
                ctx.beginPath();
                ctx.moveTo(last[0], last[1]);
                ctx.lineTo(x, y);
                ctx.strokeStyle = '#ffff00';
                ctx.lineWidth = 2 / transform.scale;
                ctx.stroke();
            }
        }
    }
});

container.addEventListener('mouseup', () => {
    if (isPanning) {
        isPanning = false;
        container.classList.remove('panning');
        return;
    }

    if (draggingPointIndex !== -1) {
        draggingPointIndex = -1;
        addToHistory();
        if (selectedRoiIndex !== -1) calculateHistogram(config.rois[selectedRoiIndex]);
    } else if (isDrawing && currentTool === 'rectangle') {
        // Finish rectangle
        finishDrawing();
    }
});

// For polygon, click on first point to close (optional now since we auto-close at 4)
container.addEventListener('dblclick', (e) => {
    if (currentTool === 'polygon' && isDrawing) {
        // finishDrawing(); 
    }
});

function finishDrawing() {
    // For polygon, we need exactly 4 points now based on user request "only 4"
    if (currentTool === 'polygon' && currentPoints.length !== 4) {
        // If called prematurely (e.g. dblclick), maybe just abort or keep it?
        // User said "deberia dejarme poner solo 4".
        // If we are here from the 4th click, it's fine.
        return;
    }

    if (currentPoints.length < 3) {
        currentPoints = [];
        isDrawing = false;
        render();
        return;
    }

    const newRoi = {
        label: "New ROI",
        group: "Default",
        price: "0",
        points: currentPoints,
        color: getRandomColor()
    };
    config.rois.push(newRoi);
    selectedRoiIndex = config.rois.length - 1;
    currentPoints = [];
    isDrawing = false;

    // Switch back to pointer after drawing
    setTool('pointer');

    addToHistory();
    updateRoiList();
    calculateHistogram(newRoi);
    render();
}

container.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    zoom(e.deltaY < 0 ? 1 : -1, x, y);
});

// Tool Switching
function setTool(tool) {
    currentTool = tool;
    document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`btn-${tool}`).classList.add('active');

    // Reset drawing state
    isDrawing = false;
    currentPoints = [];
    render();
}

// Helper Functions
function isPointInPoly(p, polygon) {
    let isInside = false;
    let minX = polygon[0][0], maxX = polygon[0][0];
    let minY = polygon[0][1], maxY = polygon[0][1];
    for (let n = 1; n < polygon.length; n++) {
        let q = polygon[n];
        minX = Math.min(q[0], minX);
        maxX = Math.max(q[0], maxX);
        minY = Math.min(q[1], minY);
        maxY = Math.max(q[1], maxY);
    }

    if (p[0] < minX || p[0] > maxX || p[1] < minY || p[1] > maxY) {
        return false;
    }

    let i = 0, j = polygon.length - 1;
    for (i, j; i < polygon.length; j = i++) {
        if ((polygon[i][1] > p[1]) != (polygon[j][1] > p[1]) &&
            p[0] < (polygon[j][0] - polygon[i][0]) * (p[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]) {
            isInside = !isInside;
        }
    }
    return isInside;
}

function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}

// History & Data
function addToHistory() {
    if (historyIndex < history.length - 1) {
        history = history.slice(0, historyIndex + 1);
    }
    history.push(JSON.parse(JSON.stringify(config)));
    historyIndex++;
}

function undo() {
    if (historyIndex > 0) {
        historyIndex--;
        config = JSON.parse(JSON.stringify(history[historyIndex]));
        render();
        updateRoiList();
        statusEl.innerText = "Undid last action";
    }
}

function redo() {
    if (historyIndex < history.length - 1) {
        historyIndex++;
        config = JSON.parse(JSON.stringify(history[historyIndex]));
        render();
        updateRoiList();
        statusEl.innerText = "Redid last action";
    }
}

function updateRoiList() {
    roiListEl.innerHTML = '';

    // Group ROIs
    const groups = {};
    config.rois.forEach((roi, index) => {
        const groupName = roi.group || 'Unassigned';
        if (!groups[groupName]) groups[groupName] = [];
        groups[groupName].push({ roi, index });
    });

    // Render Groups
    Object.keys(groups).sort().forEach(groupName => {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'roi-group mb-3';

        const groupHeader = document.createElement('div');
        groupHeader.className = 'text-uppercase text-muted small fw-bold mb-2 ps-1';
        groupHeader.innerText = groupName;
        groupDiv.appendChild(groupHeader);

        groups[groupName].forEach(item => {
            const { roi, index } = item;
            const div = document.createElement('div');
            div.className = 'roi-item' + (index === selectedRoiIndex ? ' selected' : '');
            div.onclick = () => {
                selectedRoiIndex = index;
                render();
                updateRoiList();
            };

            div.innerHTML = `
                <div class="d-flex align-items-center mb-2">
                    <div class="roi-color" style="background-color:${roi.color}"></div>
                    <input class="roi-input fw-bold" type="text" value="${roi.label}" 
                        onchange="updateLabel(${index}, this.value)" 
                        onclick="event.stopPropagation()" placeholder="Label">
                    <button class="btn btn-sm btn-link text-danger p-0 ms-2" 
                        onclick="deleteRoi(${index}); event.stopPropagation()"><i class="bi bi-x-lg"></i></button>
                </div>
                <div class="row g-2">
                    <div class="col-6">
                        <input class="form-control form-control-sm bg-dark text-light border-secondary" 
                            type="text" value="${roi.group || ''}" 
                            onchange="updateGroup(${index}, this.value)" 
                            onclick="event.stopPropagation()" placeholder="Group">
                    </div>
                    <div class="col-6">
                        <div class="input-group input-group-sm">
                            <span class="input-group-text bg-dark text-secondary border-secondary">$</span>
                            <input class="form-control bg-dark text-light border-secondary" 
                                type="number" value="${roi.price || 0}" 
                                onchange="updatePrice(${index}, this.value)" 
                                onclick="event.stopPropagation()" placeholder="Price">
                        </div>
                    </div>
                </div>
            `;
            groupDiv.appendChild(div);
        });

        roiListEl.appendChild(groupDiv);
    });
}

function updateLabel(index, newLabel) {
    config.rois[index].label = newLabel;
    addToHistory();
}

function updateGroup(index, newGroup) {
    config.rois[index].group = newGroup;
    addToHistory();
    updateRoiList(); // Re-render to move to new group
}

function updatePrice(index, newPrice) {
    config.rois[index].price = newPrice;
    addToHistory();
}

function deleteRoi(index) {
    if (confirm("Delete this ROI?")) {
        config.rois.splice(index, 1);
        selectedRoiIndex = -1;
        addToHistory();
        render();
        updateRoiList();
    }
}

function clearAll() {
    if (confirm("Are you sure you want to clear all ROIs?")) {
        config.rois = [];
        selectedRoiIndex = -1;
        addToHistory();
        render();
        updateRoiList();
    }
}

function saveConfig() {
    fetch('/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    })
        .then(res => res.json())
        .then(data => {
            statusEl.innerText = "Saved at " + new Date().toLocaleTimeString();
        });
}

function calculateHistogram(roi) {
    fetch('/histogram', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            points: roi.points,
            frame_index: currentFrameIndex,
            label: roi.label
        })
    })
        .then(res => res.json())
        .then(data => {
            roi.histogram = data.histogram;
            roi.image_path = data.image_path;
            console.log("Histogram updated for", roi.label);
        });
}

// Start
init();
