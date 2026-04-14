let sessionId = null;

async function api(path, method = "GET", body = null) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }
  return res.json();
}

function render(state) {
  document.getElementById("board").textContent = state.pretty;
  const winner = state.done ? `Winner: ${state.winner}` : "Game in progress";
  document.getElementById("status").textContent = `Current player: ${state.current_player} | ${winner}`;

  const actions = document.getElementById("actions");
  actions.innerHTML = "";
  state.legal_actions.forEach((action) => {
    const btn = document.createElement("button");
    btn.textContent = `Move ${action}`;
    btn.onclick = async () => {
      const next = await api("/api/move", "POST", { session_id: sessionId, action });
      render(next);
    };
    actions.appendChild(btn);
  });
}

async function init() {
  const list = await api("/api/games");
  const select = document.getElementById("game");
  list.games.forEach((g) => {
    const opt = document.createElement("option");
    opt.value = g;
    opt.textContent = g;
    select.appendChild(opt);
  });

  document.getElementById("start").onclick = async () => {
    let parsed = {};
    try {
      parsed = JSON.parse(document.getElementById("params").value || "{}");
    } catch {
      alert("Params must be valid JSON");
      return;
    }
    const state = await api("/api/new_game", "POST", {
      game: document.getElementById("game").value,
      params: parsed,
      human_player: Number(document.getElementById("human_player").value),
    });
    sessionId = state.session_id;
    render(state);
  };
}

init();
