const connDot = document.getElementById("conn-dot");
const connLabel = document.getElementById("conn-label");
const armLabel = document.getElementById("arm-label");
const armButton = document.getElementById("arm-button");
const testButton = document.getElementById("test-button");
const reconnectButton = document.getElementById("reconnect-button");
const alertBanner = document.getElementById("alert-banner");
const bannerTitle = document.getElementById("banner-title");
const bannerCopy = document.getElementById("banner-copy");
const stateValue = document.getElementById("state-value");
const fatigueValue = document.getElementById("fatigue-value");
const erraticValue = document.getElementById("erratic-value");
const sourceValue = document.getElementById("source-value");
const updatedValue = document.getElementById("updated-value");
const eventLog = document.getElementById("event-log");

const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = `${wsProtocol}//${window.location.host}/ws/mobile`;

const VIBRATE_COOLDOWN_MS = 10000;
const HEARTBEAT_MS = 20000;

let ws = null;
let isArmed = false;
let heartbeatTimer = null;
let lastAlertActive = false;
let lastVibrateAt = 0;

function fmtPercent(value) {
  return `${((Number(value) || 0) * 100).toFixed(1)}%`;
}

function fmtTime(ts) {
  if (!ts) {
    return "No events yet";
  }
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function setConnectionState(connected) {
  connDot.classList.toggle("connected", connected);
  connLabel.textContent = connected ? "Connected" : "Disconnected";
}

function setArmState(armed) {
  isArmed = armed;
  armLabel.textContent = armed ? "Armed for vibration" : "Not armed";
  armButton.textContent = armed ? "Alerts Armed" : "Arm Alerts";
}

function logMessage(text) {
  eventLog.textContent = text;
}

function updateBanner(message) {
  const active = Boolean(message.alert);
  alertBanner.dataset.alert = String(active);
  stateValue.textContent = message.state || "NORMAL";
  fatigueValue.textContent = fmtPercent(message.fatigue_score);
  erraticValue.textContent = fmtPercent(message.erratic_score);
  sourceValue.textContent = message.source || "unknown";
  updatedValue.textContent = fmtTime(message.timestamp);

  if (active) {
    bannerTitle.textContent = `${message.state || "HIGH"} alert`;
    bannerCopy.textContent =
      message.message ||
      "Fatigue or erratic-driving thresholds were crossed. Warn the driver immediately.";
  } else {
    bannerTitle.textContent = "Monitoring normally";
    bannerCopy.textContent = "No active driver alert right now.";
  }

  logMessage(JSON.stringify(message, null, 2));
}

function maybeVibrate(message) {
  if (!isArmed || !message.alert || !("vibrate" in navigator)) {
    return;
  }

  const now = Date.now();
  const pattern = Array.isArray(message.pattern) && message.pattern.length
    ? message.pattern
    : [300, 120, 300];
  const risingEdge = !lastAlertActive;
  const cooldownExpired = now - lastVibrateAt >= VIBRATE_COOLDOWN_MS;

  if (risingEdge || cooldownExpired) {
    navigator.vibrate(pattern);
    lastVibrateAt = now;
  }
}

function handleIncoming(raw) {
  let message;
  try {
    message = JSON.parse(raw);
  } catch {
    return;
  }

  if (message.type === "pong" || message.type === "connection") {
    return;
  }

  updateBanner(message);
  maybeVibrate(message);
  lastAlertActive = Boolean(message.alert);
}

function stopHeartbeat() {
  if (heartbeatTimer) {
    window.clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  }
}

function startHeartbeat() {
  stopHeartbeat();
  heartbeatTimer = window.setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send("ping");
    }
  }, HEARTBEAT_MS);
}

function connect() {
  stopHeartbeat();
  if (ws) {
    ws.close();
  }

  setConnectionState(false);
  connLabel.textContent = "Connecting";
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    setConnectionState(true);
    ws.send("phone-ready");
    startHeartbeat();
  };

  ws.onmessage = (event) => {
    handleIncoming(event.data);
  };

  ws.onclose = () => {
    setConnectionState(false);
    stopHeartbeat();
  };

  ws.onerror = () => {
    setConnectionState(false);
  };
}

armButton.addEventListener("click", async () => {
  setArmState(true);
  if (document.visibilityState === "visible" && "vibrate" in navigator) {
    navigator.vibrate(1);
  }

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    connect();
  }
});

testButton.addEventListener("click", () => {
  if ("vibrate" in navigator) {
    navigator.vibrate([120, 60, 120]);
  }
});

reconnectButton.addEventListener("click", () => {
  connect();
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/phone/sw.js", { scope: "/phone/" }).catch(() => {});
  });
}

connect();
