/**
 * Elderly Care AI Dashboard - JavaScript
 * Real-time monitoring interface with WebSocket updates
 * MongoDB integration for persistent storage
 */

// Global state
let socket = null;
let isMonitoring = false;
let activityChart = null;
let emotionChart = null;
let movementChart = null;
let dbConnected = false;

// Activity counts for summary
let activityCounts = {
    standing: 0,
    sitting: 0,
    walking: 0,
    lying_down: 0,
    inactive: 0
};
let fallCount = 0;

// Chart data storage
const maxDataPoints = 50;
let activityData = [];
let emotionData = [];
let movementData = [];

// Emotion history storage
const maxEmotionHistory = 30;
let emotionHistoryList = [];

// Dashboard summary refresh (for "every 3 min" concept)
const SUMMARY_REFRESH_SECONDS = 180;
let summaryCountdown = SUMMARY_REFRESH_SECONDS;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initSocket();
    initCharts();
    updateDateTime();
    setInterval(updateDateTime, 1000);
    loadEmotionHistory();
    startSummaryCountdown();
    
    // Auto-start monitoring with laptop camera after a short delay
    setTimeout(() => {
        console.log('🎥 Auto-starting webcam monitoring...');
        startMonitoring();
    }, 1500);
});

/**
 * Load emotion history from MongoDB on page load
 */
async function loadEmotionHistory() {
    try {
        const response = await fetch('/api/emotions');
        const emotions = await response.json();
        
        if (emotions && emotions.length > 0) {
            // Add unique emotions to history (avoid duplicates)
            emotions.forEach(e => {
                const exists = emotionHistoryList.some(
                    h => h.personId === e.personId && 
                         h.emotion === e.emotion && 
                         Math.abs(h.timestamp - e.timestamp) < 5
                );
                if (!exists) {
                    emotionHistoryList.push(e);
                }
            });
            
            // Sort by timestamp descending
            emotionHistoryList.sort((a, b) => b.timestamp - a.timestamp);
            
            // Limit to max
            if (emotionHistoryList.length > maxEmotionHistory) {
                emotionHistoryList = emotionHistoryList.slice(0, maxEmotionHistory);
            }
            
            renderEmotionHistory();
        }
    } catch (error) {
        console.error('Failed to load emotion history:', error);
    }
}

/**
 * Initialize WebSocket connection
 */
function initSocket() {
    socket = io();
    
    socket.on('connect', () => {
        console.log('Connected to server');
        updateConnectionStatus('Connected', 'connected');
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateConnectionStatus('Disconnected', '');
    });
    
    socket.on('connected', (data) => {
        console.log(data.status);
        dbConnected = data.db_connected || false;
        updateDbStatus();
    });
    
    socket.on('status_update', (data) => {
        handleStatusUpdate(data);
    });
    
    socket.on('new_alert', (alert) => {
        addAlert(alert);
        playAlertSound(alert.severity);
    });
}

/**
 * Update connection status badge
 */
function updateConnectionStatus(text, className) {
    const badge = document.getElementById('connection-status');
    badge.textContent = text;
    badge.className = 'status-badge ' + className;
}

/**
 * Update date/time display
 */
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
    document.getElementById('datetime').textContent = now.toLocaleDateString('en-US', options);
}

/**
 * Summary countdown + periodic backend summary fetch
 */
function startSummaryCountdown() {
    // Initial paint
    renderSummaryCountdown();
    
    setInterval(() => {
        summaryCountdown -= 1;
        if (summaryCountdown <= 0) {
            summaryCountdown = SUMMARY_REFRESH_SECONDS;
            fetchSummarySnapshot();
        }
        renderSummaryCountdown();
    }, 1000);
}

function renderSummaryCountdown() {
    const el = document.getElementById('next-summary-countdown');
    if (!el) return;
    const mins = Math.floor(summaryCountdown / 60);
    const secs = summaryCountdown % 60;
    el.textContent = `${mins.toString().padStart(2, '0')}:${secs
        .toString()
        .padStart(2, '0')}`;
}

async function fetchSummarySnapshot() {
    try {
        const res = await fetch('/api/summary');
        await res.json();
        // We could surface more info here later (e.g. mini badges),
        // for now it simply confirms the backend snapshot is refreshed.
    } catch (e) {
        console.error('Failed to refresh summary snapshot', e);
    }
}

/**
 * Smooth scroll helper for sidebar navigation
 */
function scrollToSection(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Start monitoring - captures live video from laptop camera
 */
function startMonitoring() {
    const cameraSelect = document.getElementById('camera-select');
    const source = cameraSelect.value;
    
    console.log(`📷 Starting monitoring with camera source: ${source}`);
    
    // Update UI
    document.getElementById('start-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;
    const videoOverlay = document.getElementById('video-overlay');
    if (videoOverlay) videoOverlay.classList.add('hidden');
    
    // Show live indicator
    const liveIndicator = document.getElementById('live-indicator');
    if (liveIndicator) {
        liveIndicator.classList.add('active');
    }
    
    // Start video feed from laptop camera
    const videoFeed = document.getElementById('video-feed');
    videoFeed.src = `/video_feed?source=${source}&t=${Date.now()}`;  // Add timestamp to prevent caching
    
    isMonitoring = true;
    updateConnectionStatus('🎥 Live', 'monitoring');
    
    // Notify server
    fetch('/api/start', { method: 'POST' })
        .then(res => res.json())
        .then(data => console.log('Start response:', data))
        .catch(err => console.error('Start error:', err));
}

/**
 * Stop monitoring - releases camera
 */
function stopMonitoring() {
    console.log('🛑 Stopping monitoring...');
    
    // Update UI
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    const videoOverlay = document.getElementById('video-overlay');
    if (videoOverlay) videoOverlay.classList.remove('hidden');
    
    // Hide live indicator
    const liveIndicator = document.getElementById('live-indicator');
    if (liveIndicator) {
        liveIndicator.classList.remove('active');
    }
    
    // Stop video feed
    document.getElementById('video-feed').src = '';
    
    isMonitoring = false;
    updateConnectionStatus('Stopped', '');
    
    // Notify server to release camera
    fetch('/api/stop', { method: 'POST' })
        .then(res => res.json())
        .then(data => console.log('Stop response:', data))
        .catch(err => console.error('Stop error:', err));
}

/**
 * Change camera source
 */
function changeCamera() {
    if (isMonitoring) {
        const cameraSelect = document.getElementById('camera-select');
        document.getElementById('video-feed').src = `/video_feed?source=${cameraSelect.value}`;
    }
}

/**
 * Handle status update from server
 */
function handleStatusUpdate(data) {
    // Update stats
    document.getElementById('person-count').textContent = data.statuses.length;
    document.getElementById('frame-num').textContent = data.frame_num || 0;
    document.getElementById('alert-count').textContent = data.alert_count || 0;
    
    // Mirror key stats into video HUD
    const hudPerson = document.getElementById('video-person-count');
    const hudAlerts = document.getElementById('video-alert-count');
    const hudActivity = document.getElementById('video-main-activity');
    const hudEmotion = document.getElementById('video-main-emotion');
    const hudEmotionPill = document.getElementById('video-main-emotion-pill');
    const hudFall = document.getElementById('video-fall-status');
    const hudFallPill = document.getElementById('video-fall-status-pill');
    
    if (hudPerson) hudPerson.textContent = data.statuses.length;
    if (hudAlerts) hudAlerts.textContent = data.alert_count || 0;
    
    // Update DB status
    if (data.db_connected !== undefined) {
        dbConnected = data.db_connected;
        updateDbStatus();
    }
    
    // Derive a “primary” person for HUD (first in list)
    if (data.statuses && data.statuses.length > 0) {
        const primary = data.statuses[0];
        const activity = primary.activity?.type || primary.activity || 'Idle';
        const emotion = primary.emotion?.type || primary.emotion || 'Unknown';
        const fallDetected = primary.fall_detection?.fall_detected || false;
        const fallStatus = primary.fall_detection?.status || (fallDetected ? 'Fall detected' : 'Normal');
        
        if (hudActivity) {
            hudActivity.textContent = activity.replace('_', ' ');
        }
        if (hudEmotion) {
            hudEmotion.textContent = emotion;
        }
        if (hudEmotionPill) {
            hudEmotionPill.classList.remove('positive', 'negative', 'danger');
            const emoClass = getEmotionClass(emotion);
            if (emoClass === 'positive') hudEmotionPill.classList.add('positive');
            else if (emoClass === 'negative') hudEmotionPill.classList.add('negative');
        }
        if (hudFall) {
            hudFall.textContent = fallStatus;
        }
        if (hudFallPill) {
            if (fallDetected) {
                hudFallPill.classList.add('danger', 'negative');
            } else {
                hudFallPill.classList.remove('danger', 'negative');
                hudFallPill.classList.add('positive');
            }
        }
    } else {
        // Reset HUD when no statuses
        if (hudActivity) hudActivity.textContent = 'Idle';
        if (hudEmotion) hudEmotion.textContent = '—';
        if (hudEmotionPill) hudEmotionPill.classList.remove('positive', 'negative', 'danger');
        if (hudFall) hudFall.textContent = 'Normal';
        if (hudFallPill) {
            hudFallPill.classList.remove('negative');
            hudFallPill.classList.add('positive');
        }
    }
    
    // Update person cards
    updatePersonCards(data.statuses);
    
    // Update emotion history
    if (data.statuses.length > 0) {
        updateEmotionHistory(data.statuses);
    }
    
    // Update charts
    if (data.statuses.length > 0) {
        updateChartData(data.statuses);
    }
    
    // Update summary counts
    updateSummaryCounts(data.statuses);
}

/**
 * Update database connection status indicator
 */
function updateDbStatus() {
    const dbIndicator = document.getElementById('db-status');
    if (dbIndicator) {
        if (dbConnected) {
            dbIndicator.textContent = '🗄️ MongoDB Connected';
            dbIndicator.className = 'status-badge connected';
        } else {
            dbIndicator.textContent = '⚠️ DB Offline';
            dbIndicator.className = 'status-badge';
        }
    }
}

/**
 * Update emotion history list
 */
function updateEmotionHistory(statuses) {
    const container = document.getElementById('emotion-history-list');
    if (!container) return;
    
    statuses.forEach(status => {
        const emotion = status.emotion?.type || status.emotion || 'unknown';
        const confidence = status.emotion?.confidence || 0;
        const personId = status.person_id;
        const timestamp = status.timestamp || Date.now() / 1000;
        
        // Only add if emotion is detected (not unknown)
        if (emotion !== 'unknown' && emotion !== '') {
            // Check if this is a new emotion (different from last one for this person)
            const lastEntry = emotionHistoryList.find(e => e.personId === personId);
            if (!lastEntry || lastEntry.emotion !== emotion) {
                emotionHistoryList.unshift({
                    personId: personId,
                    emotion: emotion,
                    confidence: confidence,
                    timestamp: timestamp
                });
                
                // Limit history size
                if (emotionHistoryList.length > maxEmotionHistory) {
                    emotionHistoryList.pop();
                }
            }
        }
    });
    
    // Render emotion history
    renderEmotionHistory();
}

/**
 * Render emotion history list
 */
function renderEmotionHistory() {
    const container = document.getElementById('emotion-history-list');
    if (!container) return;
    
    if (emotionHistoryList.length === 0) {
        container.innerHTML = '<div class="empty-state"><span>No emotions detected yet</span></div>';
        return;
    }
    
    let tableHTML = `
        <table class="emotion-table" style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
            <thead>
                <tr style="text-align: left; border-bottom: 2px solid #00b7b5; color: #005461;">
                    <th style="padding: 8px;">Time</th>
                    <th style="padding: 8px;">ID</th>
                    <th style="padding: 8px;">Emotion</th>
                    <th style="padding: 8px;">Confidence</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    tableHTML += emotionHistoryList.map(entry => {
        const time = new Date(entry.timestamp * 1000).toLocaleTimeString();
        const emotionIcon = getEmotionIcon(entry.emotion);
        const emotionClass = getEmotionClass(entry.emotion);
        const confidencePercent = Math.round(entry.confidence * 100);
        
        return `
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 8px; color: #5a9ea8;">${time}</td>
                <td style="padding: 8px; font-weight: bold; color: #005461;">${entry.personId}</td>
                <td style="padding: 8px;" class="${emotionClass}">
                    <span style="margin-right: 5px;">${emotionIcon}</span>
                    <span style="text-transform: capitalize;">${entry.emotion}</span>
                </td>
                <td style="padding: 8px;">${confidencePercent}%</td>
            </tr>
        `;
    }).join('');
    
    tableHTML += `</tbody></table>`;
    container.innerHTML = tableHTML;
}

/**
 * Get emoji icon for emotion
 */
function getEmotionIcon(emotion) {
    const icons = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fearful': '😨',
        'fear': '😨',
        'surprised': '😲',
        'surprise': '😲',
        'neutral': '😐',
        'tired': '😴',
        'distressed': '😰',
        'disgust': '🤢'
    };
    return icons[emotion.toLowerCase()] || '🙂';
}

/**
 * Get CSS class for emotion styling
 */
function getEmotionClass(emotion) {
    const positiveEmotions = ['happy', 'surprised', 'surprise'];
    const negativeEmotions = ['sad', 'angry', 'fearful', 'fear', 'distressed', 'disgust'];
    
    const emotionLower = emotion.toLowerCase();
    
    if (positiveEmotions.includes(emotionLower)) {
        return 'positive';
    } else if (negativeEmotions.includes(emotionLower)) {
        return 'negative';
    }
    return 'neutral';
}

/**
 * Clear emotion history
 */
function clearEmotionHistory() {
    emotionHistoryList = [];
    renderEmotionHistory();
}

/**
 * Update person status cards
 */
function updatePersonCards(statuses) {
    const container = document.getElementById('person-cards');
    
    if (statuses.length === 0) {
        container.innerHTML = '<div class="empty-state"><span>No persons detected</span></div>';
        return;
    }
    
    let tableHTML = `
        <table class="person-table" style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
            <thead>
                <tr style="text-align: left; border-bottom: 2px solid #00b7b5; color: #005461;">
                    <th style="padding: 8px;">ID</th>
                    <th style="padding: 8px;">Activity</th>
                    <th style="padding: 8px;">Emotion</th>
                    <th style="padding: 8px;">Move Score</th>
                    <th style="padding: 8px;">Duration</th>
                    <th style="padding: 8px;">Fall Status</th>
                </tr>
            </thead>
            <tbody>
    `;

    tableHTML += statuses.map(status => {
        const activity = status.activity?.type || status.activity || 'unknown';
        const emotion = status.emotion?.type || status.emotion || 'unknown';
        const fallDetected = status.fall_detection?.fall_detected || false;
        const isDistressed = status.emotion?.is_distressed || false;
        
        const emotionClass = getEmotionClass(emotion);
        const movementScore = status.activity?.movement_score || 0;
        const duration = status.activity?.duration_seconds || 0;
        
        let rowStyle = "border-bottom: 1px solid #eee;";
        if (fallDetected) rowStyle += " background-color: rgba(220, 38, 38, 0.1);";
        else if (isDistressed) rowStyle += " background-color: rgba(217, 119, 6, 0.1);";

        return `
            <tr style="${rowStyle}">
                <td style="padding: 8px; font-weight: bold; color: #005461;">${status.person_id}</td>
                <td style="padding: 8px;"><span class="activity-badge ${activity}">${activity.replace('_', ' ')}</span></td>
                <td style="padding: 8px;" class="${emotionClass}">${emotion}</td>
                <td style="padding: 8px;">${movementScore.toFixed(1)}</td>
                <td style="padding: 8px;">${formatDuration(duration)}</td>
                <td style="padding: 8px;" class="${fallDetected ? 'negative' : 'positive'}">
                    ${status.fall_detection?.status || 'normal'}
                </td>
            </tr>
        `;
    }).join('');

    tableHTML += `</tbody></table>`;
    container.innerHTML = tableHTML;
}

/**
 * Get emotion CSS class
 */
function getEmotionClass(emotion) {
    const positive = ['happy'];
    const negative = ['sad', 'angry', 'fearful', 'distressed'];
    
    if (positive.includes(emotion)) return 'positive';
    if (negative.includes(emotion)) return 'negative';
    return 'neutral';
}

/**
 * Format duration in seconds to readable string
 */
function formatDuration(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
}

/**
 * Update summary counts
 */
function updateSummaryCounts(statuses) {
    // Reset counts
    const counts = { standing: 0, sitting: 0, walking: 0, lying_down: 0, inactive: 0 };
    let falls = 0;
    
    statuses.forEach(status => {
        const activity = status.activity?.type || status.activity || 'unknown';
        if (counts.hasOwnProperty(activity)) {
            counts[activity]++;
        }
        if (status.fall_detection?.fall_detected) {
            falls++;
        }
    });
    
    // Update UI
    document.getElementById('standing-count').textContent = counts.standing;
    document.getElementById('sitting-count').textContent = counts.sitting;
    document.getElementById('walking-count').textContent = counts.walking;
    document.getElementById('lying-count').textContent = counts.lying_down;
    document.getElementById('inactive-count').textContent = counts.inactive;
    document.getElementById('fall-count').textContent = falls;
}

/**
 * Add alert to alerts panel
 */
function addAlert(alert) {
    const container = document.getElementById('alerts-list');
    
    let tbody = container.querySelector('tbody');
    
    if (!tbody) {
        container.innerHTML = `
            <table class="alerts-table" style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                <thead>
                    <tr style="text-align: left; border-bottom: 2px solid #00b7b5; color: #005461;">
                        <th style="padding: 8px;">Time</th>
                        <th style="padding: 8px;">Type</th>
                        <th style="padding: 8px;">Message</th>
                        <th style="padding: 8px;">Severity</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        `;
        tbody = container.querySelector('tbody');
    }
    
    const alertIcon = getAlertIcon(alert.type);
    const time = new Date(alert.timestamp * 1000).toLocaleTimeString();
    
    const tr = document.createElement('tr');
    tr.style.borderBottom = "1px solid #eee";
    
    let severityStyle = "";
    if (alert.severity === 'critical') severityStyle = "color: #dc2626; font-weight: bold;";
    else if (alert.severity === 'high') severityStyle = "color: #d97706; font-weight: bold;";
    
    tr.innerHTML = `
        <td style="padding: 8px; color: #5a9ea8;">${time}</td>
        <td style="padding: 8px; text-transform: capitalize;">${alertIcon} ${alert.type.replace('_', ' ')}</td>
        <td style="padding: 8px;">${alert.message}</td>
        <td style="padding: 8px; text-transform: uppercase; ${severityStyle}">${alert.severity}</td>
    `;
    
    tbody.insertBefore(tr, tbody.firstChild);
    
    // Limit alerts displayed
    while (tbody.children.length > 20) {
        tbody.removeChild(tbody.lastChild);
    }
}

/**
 * Get alert icon based on type
 */
function getAlertIcon(type) {
    const icons = {
        'fall_detected': '🚨',
        'inactivity': '⏰',
        'distress': '😰',
        'negative_emotion': '😢',
        'left_camera_view': '📍'
    };
    return icons[type] || '⚠️';
}

/**
 * Clear all alerts
 */
function clearAlerts() {
    const container = document.getElementById('alerts-list');
    container.innerHTML = '<div class="empty-state"><span>No alerts</span></div>';
    
    fetch('/api/alerts/clear', { method: 'POST' });
    document.getElementById('alert-count').textContent = '0';
}

/**
 * Play alert sound
 */
function playAlertSound(severity) {
    // Simple beep for alerts (can be enhanced with actual audio files)
    if (severity === 'critical' || severity === 'high') {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = severity === 'critical' ? 880 : 660;
            oscillator.type = 'sine';
            gainNode.gain.value = 0.3;
            
            oscillator.start();
            oscillator.stop(audioContext.currentTime + 0.3);
        } catch (e) {
            console.log('Audio not available');
        }
    }
}

/**
 * Initialize Chart.js charts
 */
function initCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top',
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
                grid: {
                    color: 'rgba(148, 163, 184, 0.1)'
                },
                ticks: {
                    color: '#94a3b8'
                }
            }
        }
    };
    
    // Activity Chart
    const activityCtx = document.getElementById('activity-chart').getContext('2d');
    activityChart = new Chart(activityCtx, {
        type: 'bar',
        data: {
            labels: ['Standing', 'Sitting', 'Walking', 'Lying', 'Inactive'],
            datasets: [{
                label: 'Current Activity',
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.7)',
                    'rgba(59, 130, 246, 0.7)',
                    'rgba(245, 158, 11, 0.7)',
                    'rgba(249, 115, 22, 0.7)',
                    'rgba(239, 68, 68, 0.7)'
                ],
                borderColor: [
                    '#10b981',
                    '#3b82f6',
                    '#f59e0b',
                    '#f97316',
                    '#ef4444'
                ],
                borderWidth: 1
            }]
        },
        options: chartOptions
    });
    
    // Emotion Chart
    const emotionCtx = document.getElementById('emotion-chart').getContext('2d');
    emotionChart = new Chart(emotionCtx, {
        type: 'doughnut',
        data: {
            labels: ['Happy', 'Neutral', 'Sad', 'Angry', 'Fearful', 'Tired'],
            datasets: [{
                data: [1, 1, 0, 0, 0, 0],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(148, 163, 184, 0.8)',
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(168, 85, 247, 0.8)',
                    'rgba(107, 114, 128, 0.8)'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#94a3b8',
                        font: { size: 10 },
                        boxWidth: 12
                    }
                }
            }
        }
    });
    
    // Movement Chart
    const movementCtx = document.getElementById('movement-chart').getContext('2d');
    movementChart = new Chart(movementCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Movement Score',
                data: [],
                borderColor: '#0d9488',
                backgroundColor: 'rgba(13, 148, 136, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            ...chartOptions,
            scales: {
                ...chartOptions.scales,
                y: {
                    ...chartOptions.scales.y,
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

/**
 * Update chart data
 */
function updateChartData(statuses) {
    if (!statuses || statuses.length === 0) return;
    
    // Update activity chart
    const activityCounts = { standing: 0, sitting: 0, walking: 0, lying_down: 0, inactive: 0 };
    const emotionCounts = { happy: 0, neutral: 0, sad: 0, angry: 0, fearful: 0, tired: 0 };
    let totalMovement = 0;
    
    statuses.forEach(status => {
        const activity = status.activity?.type || status.activity || 'unknown';
        const emotion = status.emotion?.type || status.emotion || 'neutral';
        const movement = status.activity?.movement_score || 0;
        
        if (activityCounts.hasOwnProperty(activity)) {
            activityCounts[activity]++;
        }
        if (emotionCounts.hasOwnProperty(emotion)) {
            emotionCounts[emotion]++;
        }
        totalMovement += movement;
    });
    
    // Update activity chart
    activityChart.data.datasets[0].data = [
        activityCounts.standing,
        activityCounts.sitting,
        activityCounts.walking,
        activityCounts.lying_down,
        activityCounts.inactive
    ];
    activityChart.update('none');
    
    // Update emotion chart
    emotionChart.data.datasets[0].data = [
        emotionCounts.happy,
        emotionCounts.neutral,
        emotionCounts.sad,
        emotionCounts.angry,
        emotionCounts.fearful,
        emotionCounts.tired
    ];
    emotionChart.update('none');
    
    // Update movement chart
    const avgMovement = totalMovement / statuses.length;
    const labels = movementChart.data.labels;
    const data = movementChart.data.datasets[0].data;
    
    labels.push('');
    data.push(avgMovement);
    
    if (labels.length > maxDataPoints) {
        labels.shift();
        data.shift();
    }
    
    movementChart.update('none');
}

/**
 * Fetch and display alerts
 */
async function fetchAlerts() {
    try {
        const response = await fetch('/api/alerts');
        const alerts = await response.json();
        
        alerts.forEach(alert => addAlert(alert));
    } catch (e) {
        console.error('Failed to fetch alerts:', e);
    }
}

// Load existing alerts on page load
fetchAlerts();

