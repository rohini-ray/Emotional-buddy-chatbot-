// ===============================
// Emotional Buddy AI - script.js
// Text + Voice + Face Emotion
// ===============================

const API_BASE = "http://127.0.0.1:5000";

let userId = null;
let userName = "";
let mediaRecorder = null;
let audioChunks = [];

// Elements
const modal = document.getElementById("modal");
const startBtn = document.getElementById("startBtn");
const nameInput = document.getElementById("nameInput");

const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const voiceBtn = document.getElementById("voiceBtn");

const messages = document.getElementById("messages");
const whoChip = document.getElementById("whoChip");

const themeBtn = document.getElementById("themeBtn");
const modalThemeBtn = document.getElementById("modalThemeBtn");

const recordStatus = document.getElementById("recordStatus");

const bootTime = document.getElementById("bootTime");
if (bootTime) {
    bootTime.textContent = new Date().toLocaleTimeString();
}

// ===============================
// Theme Toggle
// ===============================

function setTheme(theme) {

    if (theme === "light") {

        document.body.classList.add("light");

        // 🌙 Click to switch to Dark Mode
        themeBtn.innerHTML = "🌙";
        modalThemeBtn.innerHTML = "🌙";

    } else {

        document.body.classList.remove("light");

        // ☀️ Click to switch to Light Mode
        themeBtn.innerHTML = "☀️";
        modalThemeBtn.innerHTML = "☀️";

    }

    localStorage.setItem("theme", theme);
}


function toggleTheme() {

    const currentTheme =
        document.body.classList.contains("light")
        ? "light"
        : "dark";


    if (currentTheme === "dark") {

        // 🌞 Switch to Light Theme
        setTheme("light");

    } else {

        // 🌙 Switch to Dark Theme
        setTheme("dark");

    }
}


// Load saved theme
const savedTheme =
    localStorage.getItem("theme") || "dark";

setTheme(savedTheme);


// Theme buttons
themeBtn.onclick = toggleTheme;
modalThemeBtn.onclick = toggleTheme;

// ===============================
// Message UI
// ===============================
function addMessage(text, sender = "bot") {
    const row = document.createElement("div");
    row.className = `msgRow ${sender}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = sender === "bot" ? "🤗" : "🧑";

    const wrap = document.createElement("div");
    wrap.className = "bubbleWrap";

    const bubble = document.createElement("div");
    bubble.className = `bubble ${sender}`;
    bubble.textContent = text;

    wrap.appendChild(bubble);

    row.appendChild(avatar);
    row.appendChild(wrap);

    messages.appendChild(row);

    messages.scrollTop = messages.scrollHeight;
}

// ===============================
// Register User
// ===============================
startBtn.addEventListener("click", async () => {

    userName = nameInput.value.trim();

    if (!userName) {
        alert("Please enter your name.");
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/register`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                name: userName
            })
        });

        const data = await response.json();

        userId = data.user_id;

        whoChip.textContent = userName;

        modal.style.display = "none";

        userInput.disabled = false;
        sendBtn.disabled = false;
        voiceBtn.disabled = false;

        addMessage(`Hello ${userName}! How are you feeling today?`);

    } catch (error) {
        console.error(error);
        alert("Failed to connect to backend.");
    }
});

// ===============================
// Send Text Message
// ===============================
async function sendMessage() {

    const message = userInput.value.trim();

    if (!message) return;

    addMessage(message, "user");

    userInput.value = "";

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                user_id: userId,
                user_name: userName,
                message: message
            })
        });

        const data = await response.json();

        addMessage(data.response || "No response");

    } catch (error) {
        console.error(error);
        addMessage("Server error.");
    }
}

sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keydown", (e) => {

    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }

});

// ===============================
// Voice Recording
// ===============================
voiceBtn.addEventListener("click", async () => {

    if (mediaRecorder && mediaRecorder.state === "recording") {

        mediaRecorder.stop();

        voiceBtn.textContent = "🎙️";

        recordStatus.classList.add("hidden");

        return;
    }

    try {

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

        mediaRecorder = new MediaRecorder(stream);

        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {

            const blob = new Blob(audioChunks, {
                type: "audio/webm"
            });

            const formData = new FormData();

            formData.append("audio", blob, "voice.webm");
            formData.append("user_id", userId);
            formData.append("user_name", userName);

            try {

                addMessage("🎙️ Voice message sent", "user");

                const response = await fetch(
                    `${API_BASE}/api/analyze/voice`,
                    {
                        method: "POST",
                        body: formData
                    }
                );

                const data = await response.json();

                addMessage(data.response || "Voice processed.");

            } catch (error) {

                console.error(error);

                addMessage("Voice analysis failed.");

            }
        };

        mediaRecorder.start();

        voiceBtn.textContent = "⏹️";

        recordStatus.classList.remove("hidden");

    } catch (error) {

        console.error(error);

        alert("Microphone permission denied.");

    }
});

// ===============================
// Face Emotion Recognition
// Webcam Capture
// ===============================
const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const camera = document.getElementById("camera");
const canvas = document.getElementById("captureCanvas");

let stream = null;

startCameraBtn.addEventListener("click", async () => {
    try {

        stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        camera.srcObject = stream;

    } catch (err) {

        console.error(err);
        alert("Camera access denied.");

    }
});

captureBtn.addEventListener("click", async () => {

    if (!camera.srcObject) {
        alert("Start camera first.");
        return;
    }

    const ctx = canvas.getContext("2d");

    canvas.width = camera.videoWidth;
    canvas.height = camera.videoHeight;

    ctx.drawImage(
        camera,
        0,
        0,
        canvas.width,
        canvas.height
    );

    canvas.toBlob(async (blob) => {

        const formData = new FormData();

        formData.append("image", blob, "face.jpg");
        formData.append("user_id", userId);
        formData.append("user_name", userName);

        try {

            addMessage("📷 Face captured", "user");

            const response = await fetch(
                `${API_BASE}/api/analyze/face`,
                {
                    method: "POST",
                    body: formData
                }
            );

            const data = await response.json();

            console.log("Face Response:", data);

            addMessage(data.response || "Face analyzed.");

            console.log(data.emotion_debug);

            if (data.emotion_debug) {

                const badge =
                    document.getElementById("emotionBadge");

                const confidence =
                    document.getElementById("emotionConfidence");

                if (badge) {
                    badge.textContent =
                        data.emotion_debug.label;
                }

                if (confidence) {
                    confidence.textContent =
                        (
                            data.emotion_debug.confidence_model * 100
                        ).toFixed(1) + "%";
                }

                const history =
                    document.getElementById("emotionHistory");

                if (history) {

                    const item =
                        document.createElement("div");

                    item.className = "history-item";

                    item.innerHTML = `
                        <strong>${data.emotion_debug.label}</strong>
                        <br>
                        ${new Date().toLocaleTimeString()}
                    `;

                    history.prepend(item);
                }
            }

        } catch (err) {

            console.error(err);

            addMessage(
                "Face analysis failed.",
                "bot"
            );

        }

    }, "image/jpeg");

});

// ===============================
// Auto Focus
// ===============================
nameInput.focus();