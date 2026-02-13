/**
 * Video Insights Dashboard
 * Focused on video-captured information (movement & events)
 */

// Reuse most behaviour from main dashboard: socket, start/stop, HUD, etc.
let socket = null;
let isMonitoring = false;
let dbConnected = false;
let movementHistoryChart = null;

// Stored movement points from /api/movements
const maxHistoryPoints = 100;
let movementHistory = [];

document.addEventListener('DOMContentLoaded', () => {
    initSocket();
    initMovementHistoryChart();
    updateDateTime();
    setInterval(updateDateTime, 1000);

    // Auto-start monitoring after short delay
    setTimeout(() => {
        startMonitoring();
    }, 1500);

    // Initial load of captured movement/events
    loadCapturedMovements();
    setInterval(loadCapturedMovements, 5000);
});

function initSocket() {
    socket = io();

    socket.on('connect', () => {
        updateConnectionStatus('Connected', 'connected');
    });

    socket.on('disconnect', () => {
        updateConnectionStatus('Disconnected', '');
    });

    socket.on('connected', data => {
        dbConnected = data.db_connected || false;
        updateDbStatus();
    });

    socket.on('status_update', data => {
        handleStatusUpdate(data);
    });
}

function updateConnectionStatus(text, className) {
    const badge = document.getElementById('connection-status');
    if (!badge) return;
    badge.textContent = text;
    badge.className = 'status-badge ' + className;
}

function updateDateTime() {
    const now = new Date();
    const options = {
        weekday: 'short',
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    };
    const el = document.getElementById('datetime');
    if (el) el.textContent = now.toLocaleDateString('en-US', options);
}

function updateDbStatus() {
    const dbIndicator = document.getElementById('db-status');
    if (!dbIndicator) return;

    if (dbConnected) {
        dbIndicator.textContent = '🗄️ MongoDB Connected';
        dbIndicator.className = 'status-badge connected';
    } else {
        dbIndicator.textContent = '⚠️ DB Offline';
        dbIndicator.className = 'status-badge';
    }
}

function startMonitoring() {
    const cameraSelect = document.getElementById('camera-select');
    const source = cameraSelect ? cameraSelect.value : 0;

    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const overlay = document.getElementById('video-overlay');
    const liveIndicator = document.getElementById('live-indicator');

    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
    if (overlay) overlay.classList.add('hidden');
    if (liveIndicator) liveIndicator.classList.add('active');

    const videoFeed = document.getElementById('video-feed');
    if (videoFeed) {
        videoFeed.src = `/video_feed?source=${source}&t=${Date.now()}`;
    }

    isMonitoring = true;
    updateConnectionStatus('🎥 Live', 'monitoring');

    fetch('/api/start', { method: 'POST' }).catch(() => {});
}

function stopMonitoring() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const overlay = document.getElementById('video-overlay');
    const liveIndicator = document.getElementById('live-indicator');

    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    if (overlay) overlay.classList.remove('hidden');
    if (liveIndicator) liveIndicator.classList.remove('active');

    const videoFeed = document.getElementById('video-feed');
    if (videoFeed) {
        videoFeed.src = '';
    }

    isMonitoring = false;
    updateConnectionStatus('Stopped', '');

    fetch('/api/stop', { method: 'POST' }).catch(() => {});
}

function changeCamera() {
    if (!isMonitoring) return;
    const cameraSelect = document.getElementById('camera-select');
    const videoFeed = document.getElementById('video-feed');
    if (cameraSelect && videoFeed) {
        videoFeed.src = `/video_feed?source=${cameraSelect.value}&t=${Date.now()}`;
    }
}

function handleStatusUpdate(data) {
    const personCount = document.getElementById('person-count');
    const frameNum = document.getElementById('frame-num');
    const alertCount = document.getElementById('alert-count');

    if (personCount) personCount.textContent = data.statuses.length;
    if (frameNum) frameNum.textContent = data.frame_num || 0;
    if (alertCount) alertCount.textContent = data.alert_count || 0;

    // HUD elements
    const hudPerson = document.getElementById('video-person-count');
    const hudAlerts = document.getElementById('video-alert-count');
    const hudActivity = document.getElementById('video-main-activity');
    const hudEmotion = document.getElementById('video-main-emotion');
    const hudEmotionPill = document.getElementById('video-main-emotion-pill');
    const hudFall = document.getElementById('video-fall-status');
    const hudFallPill = document.getElementById('video-fall-status-pill');

    if (hudPerson) hudPerson.textContent = data.statuses.length;
    if (hudAlerts) hudAlerts.textContent = data.alert_count || 0;

    if (data.statuses && data.statuses.length > 0) {
        const primary = data.statuses[0];
        const activity = primary.activity?.type || primary.activity || 'Idle';
        const emotion = primary.emotion?.type || primary.emotion || 'Unknown';
        const fallDetected = primary.fall_detection?.fall_detected || false;
        const fallStatus = primary.fall_detection?.status || (fallDetected ? 'Fall detected' : 'Normal');

        if (hudActivity) hudActivity.textContent = activity.replace('_', ' ');
        if (hudEmotion) hudEmotion.textContent = emotion;

        if (hudEmotionPill) {
            hudEmotionPill.classList.remove('positive', 'negative', 'danger');
            const emoClass = getEmotionClass(emotion);
            if (emoClass === 'positive') hudEmotionPill.classList.add('positive');
            else if (emoClass === 'negative') hudEmotionPill.classList.add('negative');
        }

        if (hudFall) hudFall.textContent = fallStatus;
        if (hudFallPill) {
            if (fallDetected) {
                hudFallPill.classList.add('danger', 'negative');
            } else {
                hudFallPill.classList.remove('danger', 'negative');
                hudFallPill.classList.add('positive');
            }
        }
    } else {
        if (hudActivity) hudActivity.textContent = 'Idle';
        if (hudEmotion) hudEmotion.textContent = '—';
        if (hudEmotionPill) hudEmotionPill.classList.remove('positive', 'negative', 'danger');
        if (hudFall) hudFall.textContent = 'Normal';
        if (hudFallPill) {
            hudFallPill.classList.remove('negative');
            hudFallPill.classList.add('positive');
        }
    }
}

function getEmotionClass(emotion) {
    const positive = ['happy', 'surprised', 'surprise'];
    const negative = ['sad', 'angry', 'fearful', 'fear', 'distressed', 'disgust'];
    const e = (emotion || '').toLowerCase();
    if (positive.includes(e)) return 'positive';
    if (negative.includes(e)) return 'negative';
    return 'neutral';
}

async function loadCapturedMovements() {
    try {
        const res = await fetch('/api/movements');
        const data = await res.json();
        if (!Array.isArray(data)) return;

        // Sort by timestamp ascending for chart
        data.sort((a, b) => {
            const ta = a.timestamp || 0;
            const tb = b.timestamp || 0;
            return ta - tb;
        });

        movementHistory = data.slice(-maxHistoryPoints);
        renderCapturedEventsList();
        updateMovementHistoryChart();
    } catch (e) {
        console.error('Failed to load movements', e);
    }
}

function renderCapturedEventsList() {
    const container = document.getElementById('captured-events-list');
    if (!container) return;

    if (!movementHistory.length) {
        container.innerHTML = '<div class="empty-state"><span>No captured events yet</span></div>';
        return;
    }

    container.innerHTML = movementHistory
        .slice()
        .reverse()
        .map(m => {
            const t = (m.timestamp && typeof m.timestamp === 'number')
                ? new Date(m.timestamp * 1000).toLocaleTimeString()
                : '';
            const person = m.person_id ?? m.personId ?? '-';
            const activity = m.activity || 'unknown';
            const emotion = m.emotion || 'unknown';
            const movementScore = m.movement_score ?? m.movementScore ?? 0;
            const severityClass = activity === 'lying_down' || activity === 'inactive' ? 'high' : 'medium';

            return `
                <div class="alert-item ${severityClass}">
                    <div class="alert-icon">🎞️</div>
                    <div class="alert-content">
                        <div class="alert-message">
                            Person ${person} • ${activity.replace('_', ' ')} • ${emotion}
                        </div>
                        <div class="alert-time">
                            Movement score: ${movementScore.toFixed(1)} • ${t}
                        </div>
                    </div>
                </div>
            `;
        })
        .join('');
}

function initMovementHistoryChart() {
    const ctx = document.getElementById('movement-history-chart');
    if (!ctx) return;

    movementHistoryChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Movement Score',
                data: [],
                borderColor: '#0d9488',
                backgroundColor: 'rgba(13, 148, 136, 0.15)',
                fill: true,
                tension: 0.35,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font: { size: 10 }
                    }
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    min: 0,
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

function updateMovementHistoryChart() {
    if (!movementHistoryChart) return;

    const labels = [];
    const data = [];

    movementHistory.forEach(m => {
        const movementScore = m.movement_score ?? m.movementScore ?? 0;
        data.push(movementScore);
        labels.push('');
    });

    movementHistoryChart.data.labels = labels;
    movementHistoryChart.data.datasets[0].data = data;
    movementHistoryChart.update('none');
}

