/**
 * Hospital Intelligent Monitor - JavaScript Controller
 * Handles real-time updates for Ward Status, Occupancy, and Alerts
 */

// Global State
let socket = null;
let isMonitoring = false;
let dbConnected = false;
let currentPersonCount = 0;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initSocket();
    updateDateTime();
    checkStatus(); // Sync UI with backend state
    setInterval(updateDateTime, 1000);

    // Auto-connect feedback
    setTimeout(() => {
        if (!socket.connected) {
            console.log('Connecting...');
        }
    }, 1000);
});

/**
 * Check Backend Status and Sync UI
 */
function checkStatus() {
    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            console.log('Backend Status:', data);
            if (data.is_monitoring) {
                isMonitoring = true;

                // If we are on the Live Feed page, restore the video
                const overlay = document.getElementById('video-overlay');
                const videoFeed = document.getElementById('video-feed');
                const startBtn = document.getElementById('start-btn');
                const stopBtn = document.getElementById('stop-btn');
                const cameraSelect = document.getElementById('camera-select');

                if (videoFeed) {
                    const source = cameraSelect ? cameraSelect.value : 0;
                    // Add random parameter to bypass cache
                    videoFeed.src = `/video_feed?source=${source}&t=${Date.now()}`;
                    if (overlay) overlay.style.display = 'none';
                }

                if (startBtn) startBtn.disabled = true;
                if (stopBtn) stopBtn.disabled = false;
            }
        })
        .catch(err => console.error('Error checking status:', err));
}

/**
 * Initialize WebSocket Connection
 */
function initSocket() {
    socket = io();

    socket.on('connect', () => {
        console.log('✅ Connected to Hospital System');
        updateStatusIndicator(true);
    });

    socket.on('disconnect', () => {
        console.log('❌ Disconnected from System');
        updateStatusIndicator(false);
    });

    // 1. Person Count Update (Event Driven)
    socket.on('person_count_update', (data) => {
        console.log('Person Count Update:', data);
        updateOccupancy(data.count, data.timestamp);
    });

    // 2. Stats Update (Every 3 Mins)
    socket.on('stats_update', (data) => {
        console.log('Stats Update:', data);
        updateWardStatus(data.summary);
    });

    // 3. Alert Update (Every 1 Min)
    socket.on('alert_update', (data) => {
        console.log('Alert Update:', data);
        updateAlerts(data.alerts);
    });

    // 4. DB Status
    socket.on('connected', (data) => {
        updateDbStatus(data.db_connected);
    });
}

/**
 * Update Occupancy Log
 */
function updateOccupancy(count, timestamp) {
    const countEl = document.getElementById('person-count');
    const logContainer = document.getElementById('occupancy-log');

    // Only update if elements exist (might be on Live Feed page)
    if (!countEl || !logContainer) return;

    // Prevent duplicate logs if count hasn't changed (though server should handle this)
    if (count === currentPersonCount) return;
    currentPersonCount = count;

    // Update current count display with animation
    countEl.style.transform = "scale(1.2)";
    countEl.style.color = "var(--primary)";
    setTimeout(() => {
        countEl.style.transform = "scale(1)";
        countEl.style.color = "var(--primary)";
    }, 300);

    countEl.textContent = count;

    // Add to Log
    const timeStr = new Date(timestamp * 1000).toLocaleTimeString();

    const logItem = document.createElement('div');
    logItem.className = 'log-item occupancy';
    logItem.innerHTML = `
        <div class="log-content">
            <span style="font-weight: 600; color: var(--text-main)">Occupancy Changed to ${count}</span>
            <span class="log-time">Ward A - General</span>
        </div>
        <span class="log-time">${timeStr}</span>
    `;

    prependLog(logContainer, logItem);
}

/**
 * Update Ward Status (3 Min Buffer)
 */
function updateWardStatus(summary) {
    if (!summary || summary.length === 0) return;

    const activityEl = document.getElementById('dominant-activity');
    const emotionEl = document.getElementById('dominant-emotion');

    if (!activityEl || !emotionEl) return;

    // Simple logic: Find most common activity/emotion across all persons
    const activities = summary.map(s => s.activity);
    const emotions = summary.map(s => s.emotion);

    const dominantActivity = getMostFrequent(activities);
    const dominantEmotion = getMostFrequent(emotions);

    const timeStr = new Date().toLocaleTimeString();

    // Update DOM
    if (dominantActivity) {
        activityEl.textContent = formatText(dominantActivity);
        const lastUpd = document.getElementById('last-update-activity');
        if (lastUpd) lastUpd.textContent = `Updated: ${timeStr}`;
    }

    if (dominantEmotion) {
        emotionEl.textContent = formatText(dominantEmotion);
        const lastUpdEmo = document.getElementById('last-update-emotion');
        if (lastUpdEmo) lastUpdEmo.textContent = `Updated: ${timeStr}`;

        // Color coding for emotion
        emotionEl.className = 'metric-value emotion'; // reset base class
        if (['happy', 'neutral'].includes(dominantEmotion)) {
            emotionEl.style.color = "var(--success)";
        } else if (['sad', 'distressed', 'fear'].includes(dominantEmotion)) {
            emotionEl.style.color = "var(--danger)";
        } else {
            emotionEl.style.color = "var(--text-main)";
        }
    }
}

/**
 * Update Alerts Log (1 Min Buffer)
 */
function updateAlerts(alerts) {
    if (!alerts || alerts.length === 0) return;

    const logContainer = document.getElementById('alerts-log');
    if (!logContainer) return;

    alerts.forEach(alert => {
        const timeStr = new Date(alert.timestamp * 1000).toLocaleTimeString();
        const severity = alert.severity || 'medium';

        const logItem = document.createElement('div');
        logItem.className = `log-item alert-${severity}`;

        // Icon based on alert type
        let icon = '⚠️';
        if (alert.type === 'fall_detected') icon = '🚨';
        else if (alert.type === 'distress') icon = '😰';

        logItem.innerHTML = `
            <div class="log-content">
                <span style="font-weight: 600; color: var(--text-main)">${icon} ${alert.message}</span>
                <span class="log-time">Person ${alert.person_id}</span>
            </div>
            <span class="log-time">${timeStr}</span>
        `;

        prependLog(logContainer, logItem);
    });
}

/**
 * Helper: Prepend log item and limit list size
 */
function prependLog(container, element) {
    // Remove empty state
    const emptyState = container.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    container.insertBefore(element, container.firstChild);

    // Limit to 20 items
    if (container.children.length > 20) {
        container.removeChild(container.lastChild);
    }
}

/**
 * Helper: Get most frequent item in array
 */
function getMostFrequent(arr) {
    if (arr.length === 0) return null;

    const frequency = {};
    let maxFreq = 0;
    let mostFrequent = arr[0];

    for (const item of arr) {
        frequency[item] = (frequency[item] || 0) + 1;
        if (frequency[item] > maxFreq) {
            maxFreq = frequency[item];
            mostFrequent = item;
        }
    }

    return mostFrequent;
}

/**
 * Helper: Format text (replace underscores, capitalize)
 */
function formatText(text) {
    if (!text) return '--';
    return text.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Start Monitoring
 */
window.startMonitoring = function () {
    const cameraSelect = document.getElementById('camera-select');
    if (!cameraSelect) return;

    const source = cameraSelect.value;

    console.log(`Starting monitoring: Source ${source}`);

    const overlay = document.getElementById('video-overlay');
    const videoFeed = document.getElementById('video-feed');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');

    // UI Updates
    if (overlay) overlay.style.display = 'none';
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;

    // Set Video Source
    // Add randomness to prevent caching
    if (videoFeed) videoFeed.src = `/video_feed?source=${source}&t=${Date.now()}`;

    isMonitoring = true;

    // API Call
    fetch('/api/start', { method: 'POST' })
        .then(res => res.json())
        .then(data => console.log(data))
        .catch(err => console.error(err));
}

/**
 * Stop Monitoring
 */
window.stopMonitoring = function () {
    console.log('Stopping monitoring');

    const overlay = document.getElementById('video-overlay');
    const videoFeed = document.getElementById('video-feed');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');

    // UI Updates
    if (overlay) overlay.style.display = 'flex';
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;

    // Kill Video Source
    if (videoFeed) videoFeed.src = "";

    isMonitoring = false;

    // API Call
    fetch('/api/stop', { method: 'POST' })
        .then(res => res.json())
        .then(data => console.log(data))
        .catch(err => console.error(err));
}

/**
 * Change Camera
 */
window.changeCamera = function () {
    const cameraSelect = document.getElementById('camera-select');
    const videoFeed = document.getElementById('video-feed');

    if (isMonitoring) {
        videoFeed.src = `/video_feed?source=${cameraSelect.value}&t=${Date.now()}`;
    }
}

/**
 * Clear Alerts
 */
window.clearAlerts = function () {
    document.getElementById('alerts-log').innerHTML = '<div class="empty-state">No active alerts</div>';
}

/**
 * Update Time Display
 */
function updateDateTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour12: false });
    const dateString = now.toLocaleDateString();

    const el = document.getElementById('datetime');
    if (el) el.textContent = `${dateString} ${timeString}`;
}

/**
 * Update DB Status Indicator
 */
function updateDbStatus(connected) {
    const dot = document.getElementById('db-status-dot');
    const text = document.getElementById('db-status-text');

    if (!dot || !text) return;

    if (connected) {
        dot.className = 'status-dot connected';
        text.textContent = 'Database Online';
    } else {
        dot.className = 'status-dot';
        text.textContent = 'Database Offline';
    }
}

/**
 * Update System Status Indicator (Sidebar)
 */
function updateStatusIndicator(connected) {
    const systemStatusDot = document.querySelector('.system-status .status-item .status-dot');
    const systemStatusText = document.querySelector('.system-status .status-item span:last-child');

    if (systemStatusDot && systemStatusText) {
        if (connected) {
            systemStatusDot.className = 'status-dot connected';
            systemStatusText.textContent = 'System Online';
        } else {
            systemStatusDot.className = 'status-dot';
            systemStatusText.textContent = 'System Disconnected';
        }
    }
}
