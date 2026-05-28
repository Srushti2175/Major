/**
 * Audio-only dashboard
 * Polls /api/audio_status and renders audio emotion + keywords
 */

let audioPollInterval = null;

document.addEventListener('DOMContentLoaded', () => {
    updateDateTime();
    setInterval(updateDateTime, 1000);
    startAudioPolling();
});

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
    if (el) {
        el.textContent = now.toLocaleDateString('en-US', options);
    }
}

function startAudioPolling() {
    fetchAndRenderAudioStatus();
    audioPollInterval = setInterval(fetchAndRenderAudioStatus, 4000);
}

async function fetchAndRenderAudioStatus() {
    try {
        const res = await fetch('/api/audio_status');
        const data = await res.json();
        renderAudioStatus(data);
    } catch (e) {
        console.error('Failed to fetch audio status', e);
    }
}

function renderAudioStatus(data) {
    const emotionEl = document.getElementById('audio-emotion');
    const confEl = document.getElementById('audio-emotion-confidence');
    const kwEl = document.getElementById('audio-keywords');
    const alertEl = document.getElementById('audio-alert-level');
    const badge = document.getElementById('audio-status-badge');
    const eventsList = document.getElementById('audio-events-list');

    if (emotionEl) emotionEl.textContent = (data.emotion || '--').toUpperCase();
    if (confEl) confEl.textContent = `${Math.round((data.emotion_confidence || 0) * 100)}%`;

    if (kwEl) {
        const keywords = data.keywords_detected || [];
        kwEl.innerHTML = keywords.length
            ? keywords.map(k => `<li>${k}</li>`).join('')
            : '<li>No keywords yet</li>';
    }

    if (alertEl) {
        alertEl.textContent = (data.alert_level || 'normal').toUpperCase();
        alertEl.className = 'audio-alert-level ' + (data.alert_level || 'normal');
    }

    if (badge) {
        badge.textContent = data.needs_attention ? 'Needs attention' : 'Listening...';
        badge.className = 'status-badge ' + (data.needs_attention ? 'monitoring' : 'connected');
    }

    if (eventsList) {
        const time = new Date((data.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString();
        const item = document.createElement('div');
        item.className = 'alert-item ' + (data.needs_attention ? 'high' : 'low');
        item.innerHTML = `
            <div class="alert-icon"></div>
            <div class="alert-content">
                <div class="alert-message">
                    Sound: ${(data.emotion || 'unknown')}  Level: ${(data.alert_level || 'normal')}
                </div>
                <div class="alert-time">${time}</div>
            </div>
        `;

        const empty = eventsList.querySelector('.empty-state');
        if (empty) empty.remove();
        eventsList.insertBefore(item, eventsList.firstChild);

        while (eventsList.children.length > 20) {
            eventsList.removeChild(eventsList.lastChild);
        }
    }
}

function startMonitoring() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
    fetch('/api/start', { method: 'POST' }).catch(() => {});
}

function stopMonitoring() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    fetch('/api/stop', { method: 'POST' }).catch(() => {});
}
