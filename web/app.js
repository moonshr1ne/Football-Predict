const prediction = document.querySelector("#prediction");

document.querySelector("#predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await runPrediction(true);
});

document.querySelector("#auto-check").addEventListener("click", async () => {
  const response = await fetch("/api/auto-check", {
    method: "POST",
  });
  const data = await response.json();
  if (!response.ok) {
    document.querySelector("#review").textContent = data.error || "Ошибка автопроверки";
    return;
  }
  document.querySelector("#review").textContent =
    `Проверено: ${data.checked}, обучено: ${data.learned}, ожидают: ${data.pending}, ошибок: ${data.errors}.`;
});

async function getJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Ошибка");
  return data;
}

function renderPrediction(data) {
  const warnings = data.warnings.map((item) => `<p class="warn">${escapeHtml(item)}</p>`).join("");
  prediction.innerHTML = `
    <article class="card">
      <h3>Исход</h3>
      <div class="metric">${data.market_pick}</div>
      <p class="sub">уверенность ${(data.confidence * 100).toFixed(1)}%${data.match_date ? `, найдено ${data.match_date}` : ""}</p>
    </article>
    <article class="card">
      <h3>Угловые</h3>
      <div class="metric">${data.predicted_corners.toFixed(2)}</div>
      <p class="sub">ожидаемый тотал</p>
    </article>
    <article class="card">
      <h3>Точные счета</h3>
      <div class="pill-row">${data.exact_scores.map((score) => `<span class="pill">${score}</span>`).join("")}</div>
    </article>
    <article class="card">
      <h3>Вероятности</h3>
      <p>П1 ${(data.probabilities["П1"] * 100).toFixed(1)}%</p>
      <p>X ${(data.probabilities.X * 100).toFixed(1)}%</p>
      <p>П2 ${(data.probabilities["П2"] * 100).toFixed(1)}%</p>
    </article>
    <article class="card wide">
      <h3>${escapeHtml(data.home_team)}: последние матчи</h3>
      ${statsTable(data.home_stats)}
    </article>
    <article class="card wide">
      <h3>${escapeHtml(data.away_team)}: последние матчи</h3>
      ${statsTable(data.away_stats)}
    </article>
    <article class="card wide">
      <h3>Контекст</h3>
      ${fixtureBlock(data.fixture)}
      <p>Турнир: ${escapeHtml(data.match_context?.competition || "FIFA World Cup")}</p>
      <p>Важность: ${Number(data.match_context?.importance ?? 1).toFixed(2)}, базовая сила состава: ${Number(data.match_context?.lineup_strength_floor ?? 0.92).toFixed(2)}</p>
      <p>${escapeHtml(data.home_team)}: ${contextLine(data.home_context)}</p>
      <p>${escapeHtml(data.away_team)}: ${contextLine(data.away_context)}</p>
    </article>
    <article class="card wide">
      <h3>${escapeHtml(data.home_team)}: тактика</h3>
      ${tacticsBlock(data.home_tactics)}
    </article>
    <article class="card wide">
      <h3>${escapeHtml(data.away_team)}: тактика</h3>
      ${tacticsBlock(data.away_tactics)}
    </article>
    <article class="card wide">
      <h3>Тактическая пара</h3>
      <p>${escapeHtml(data.tactical_matchup?.summary || "")}</p>
      <p class="sub">${escapeHtml(data.tactical_matchup?.home_route || "")}</p>
      <p class="sub">${escapeHtml(data.tactical_matchup?.away_route || "")}</p>
    </article>
    <article class="card wide">
      <h3>Качество данных</h3>
      ${warnings || "<p class='muted'>Предупреждений нет.</p>"}
    </article>
  `;
}

function statsTable(stats) {
  return `
    <table>
      <tr><th>Матчи</th><td>${stats.sample_size}</td><th>Форма</th><td>${stats.wins}-${stats.draws}-${stats.losses}</td></tr>
      <tr><th>Голы</th><td>${stats.avg_goals_for} / ${stats.avg_goals_against}</td><th>Очки</th><td>${stats.points_per_match}</td></tr>
      <tr><th>Угловые</th><td>${stats.avg_total_corners ?? "нет"}</td><th>Сухие</th><td>${stats.clean_sheets}</td></tr>
    </table>
  `;
}

function contextLine(context) {
  const injuries = context.injuries?.length ? `${context.injuries.length} травм/рисков` : "травм не внесено";
  const motivation = context.motivation?.level ?? 0.5;
  const lineup = context.lineup_strength ?? "World Cup base";
  return `мотивация ${motivation}, состав ${lineup}, ${injuries}`;
}

function fixtureBlock(fixture) {
  if (!fixture) {
    return "<p class='warn'>Дата матча не найдена автоматически. Проверьте, есть ли такая пара в расписании ЧМ.</p>";
  }
  const status = fixture.completed ? "завершен" : (fixture.status_detail || fixture.status || "запланирован");
  const kickoff = fixture.kickoff ? ` · ${escapeHtml(fixture.kickoff)}` : "";
  return `<p><strong>Матч найден:</strong> ${escapeHtml(fixture.date)} · ${escapeHtml(status)}${kickoff}</p>`;
}

function tacticsBlock(tactics) {
  return `
    <p><strong>${escapeHtml(tactics.formation || "unknown")}</strong> · ${escapeHtml(tactics.style || "balanced")}</p>
    <p class="sub">${escapeHtml(tactics.primary_attack || "mixed attack")} · ${escapeHtml(tactics.defensive_block || "mid")} block</p>
    <table>
      <tr><th>Владение</th><td>${percent(tactics.possession_intent)}</td><th>Прессинг</th><td>${percent(tactics.pressing)}</td></tr>
      <tr><th>Защита</th><td>${percent(tactics.defensive_solidity)}</td><th>Темп</th><td>${percent(tactics.tempo)}</td></tr>
      <tr><th>Фланги</th><td>${percent(tactics.attack_width)}</td><th>Стандарты</th><td>${percent(tactics.set_piece_threat)}</td></tr>
    </table>
  `;
}

function percent(value) {
  return `${Math.round(Number(value ?? 0.5) * 100)}%`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

async function runPrediction(remember) {
  const matchup = document.querySelector("#matchup").value;
  const homeVenue = document.querySelector("#home-venue").checked;
  const data = await getJson(`/api/predict?matchup=${encodeURIComponent(matchup)}&home_venue=${homeVenue}&remember=${remember}`);
  renderPrediction(data);
}

runPrediction(false);
