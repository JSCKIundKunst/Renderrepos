let fullGeneratedHtml = "";// Schutz vor versehentlichem Reload
let promptButtonsInitialized = false;

window.onload = () => {
  initStartScreen();
  preventReload();
  initNavigationTabs();
};

function initStartScreen() {
  switchTab('start');

  const startImage = document.getElementById("startImage");
  if (startImage) {
    startImage.addEventListener("click", showMainUI);
  }
}

function preventReload() {
  window.addEventListener("beforeunload", function (e) {
    e.preventDefault();
    e.returnValue = "";
    console.warn("🚨 Seite wollte sich neu laden!");
  });
}

function initNavigationTabs() {
  document.querySelectorAll("nav.tabs button").forEach(button => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;
      switchTab(tab);
    });
  });
}

function showMainUI() {
  document.getElementById("mainTitle").style.display = "block";
  document.querySelector("nav.tabs").style.display = "flex";
  switchTab('generator');

  if (!promptButtonsInitialized) {
    document.querySelectorAll(".prompt-buttons button").forEach(button => {
      button.addEventListener("click", async () => {
        const prompt = button.dataset.prompt || "Generiere eine satirische Spam-Mail";
        await delayedGenerate(prompt);
      });
    });
    document.getElementById("sendBtn").addEventListener("click", sendEmail);
    promptButtonsInitialized = true;
  }
}

function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(tab => {
    tab.style.display = "none";
  });
  const selected = document.getElementById(tabId);
  if (selected) selected.style.display = "block";
}

async function delayedGenerate(prompt) {
  const video = document.getElementById("loadingGif");
  video.style.display = "block";
  hideUIWhileLoading(); // UI-Elemente ausblenden

  const output = document.getElementById("output");
  output.innerHTML = "";
  const delay = Math.floor(Math.random() * 1000) + 2000; // 2000–4000ms
  await new Promise(resolve => setTimeout(resolve, delay));

  await generateWithPrompt(prompt);
}

async function generateWithPrompt(prompt) {
  console.log("🔘 Prompt gestartet:", prompt);

  const video = document.getElementById("loadingGif");
  video.style.display = "block";
  const output = document.getElementById("output");
  output.innerHTML = "";


  try {
    const response = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt })
    });

    const text = await response.text();
    let data;

    try {
      data = JSON.parse(text);
    } catch (err) {
      console.error("❌ JSON-Fehler:", err);
      alert("Antwort vom Server war kein valides JSON.");
      return;
    }

    let html = data.response;
    fullGeneratedHtml = html;

    const bodyMatch = html.match(/<body\s*(style="[^"]*")?[^>]*>([\s\S]*?)<\/body>/i);
    let bodyStyle = "";
    let bodyContent = html;

    if (bodyMatch) {
      bodyStyle = bodyMatch[1] || "";
      bodyStyle = bodyStyle.replace(/^style="|\"$/g, "");
      bodyContent = bodyMatch[2];
    }

    output.setAttribute("style", `border:1px solid #ccc; padding:1em; margin-top:10px; ${bodyStyle}`);
    output.innerHTML = bodyContent;

    output.querySelectorAll("a[href='#']").forEach(link => {
      link.addEventListener("click", e => {
        e.preventDefault();
        console.warn("⛔️ Klick auf <a href='#'> blockiert!");
      });
    });

  } catch (error) {
    console.error("❌ Fehler bei der Generierung:", error);
    alert("Fehler bei der Generierung.");
  } finally {
    video.style.display = "none";
    showUIAfterLoading();
  }
}

async function sendEmail() {
  const email = document.getElementById("emailInput").value;

  if (!email || !fullGeneratedHtml) {
    alert("Bitte E-Mail-Adresse eingeben und erst eine Mail generieren.");
    return;
  }

  try {
    const response = await fetch("/send_email", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ to: email, html: fullGeneratedHtml })
    });

    const result = await response.json();
    alert(result.message || "Mail gesendet! bitte schaue in deinen Spam-Ordner");
    document.getElementById("emailInput").value = "";
  } catch (error) {
    console.error("Fehler beim Senden der Mail:", error);
    alert("Fehler beim Mailversand.");
  }
}
function hideUIWhileLoading() {
  document.querySelectorAll(".prompt-buttons, #emailInput, #sendBtn, h2,#output").forEach(el => {
    el.classList.add("hidden-during-loading");
  });
}

function showUIAfterLoading() {
  document.querySelectorAll(".prompt-buttons, #emailInput, #sendBtn, h2,#output ").forEach(el => {
    el.classList.remove("hidden-during-loading");
  });
}
