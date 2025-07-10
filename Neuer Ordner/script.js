window.onload = () => {
  switchTab('start');
  const startImage = document.getElementById("startImage");
  if (startImage) {
    startImage.addEventListener("click", showMainUI);
  }
};

// Schutz vor versehentlichem Reload
window.addEventListener("beforeunload", function (e) {
  e.preventDefault();
  e.returnValue = "";
  console.warn("🚨 Seite wollte sich neu laden!");
});

function showMainUI() {
  document.getElementById("mainTitle").style.display = "block";
  document.querySelector("nav.tabs").style.display = "flex";
  switchTab('generator');
}

let fullGeneratedHtml = "";

function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(tab => {
    tab.style.display = "none";
  });
  const selected = document.getElementById(tabId);
  if (selected) selected.style.display = "block";
}

async function generateWithPrompt(prompt) {
  console.log("🔘 Prompt gestartet:", prompt);

  const video = document.getElementById("loadingGif");
  const output = document.getElementById("output");
  

  output.innerHTML = "";
  
  video.style.display = "block";

  try {
    const response = await fetch("http://localhost:8000/generate", {
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

    // ⛏️ Nur <body>-Inhalt extrahieren (wie früher)
    const bodyMatch = html.match(/<body\s*(style="[^"]*")?[^>]*>([\s\S]*?)<\/body>/i);
    let bodyStyle = "";
    let bodyContent = html;

    if (bodyMatch) {
      bodyStyle = bodyMatch[1] || "";
      bodyStyle = bodyStyle.replace(/^style="|\"$/g, "");
      bodyContent = bodyMatch[2];
    }

    // 🧱 HTML-Inhalt einfügen wie früher
    output.setAttribute("style", `border:1px solid #ccc; padding:1em; margin-top:10px; ${bodyStyle}`);
    output.innerHTML = bodyContent;

    
    

    // ⛔ Prevent reloads from <a href="#">
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
  }
}

async function sendEmail() {
  const email = document.getElementById("emailInput").value;

  if (!email || !fullGeneratedHtml) {
    alert("Bitte E-Mail-Adresse eingeben und erst eine Mail generieren.");
    return;
  }

  try {
    const response = await fetch("http://localhost:8000/send_email", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ to: email, html: fullGeneratedHtml })
    });

    const result = await response.json();
    alert(result.message || "Mail gesendet!");

  } catch (error) {
    console.error("Fehler beim Senden der Mail:", error);
    alert("Fehler beim Mailversand.");
  }
}
