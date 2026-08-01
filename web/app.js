const predictionRoot = document.querySelector("#prediction");
const loading = document.querySelector("#loading");
const errorBox = document.querySelector("#error");
const opponentInput = document.querySelector("#opponent");
let requestId = 0;

document.querySelector("#predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await runPrediction(true);
});

document.querySelector("#auto-check").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  button.classList.add("spinning");
  setSystem("Обновляю результаты и модель…", true);
  try {
    const response = await fetch("/api/auto-check", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Ошибка обновления");
    const reviewed = data.review?.reviewed ?? 0;
    setSystem(`База обновлена · проверено прогнозов: ${reviewed}`, false);
    await runPrediction(false);
  } catch (error) {
    showError(error.message);
    setSystem("Обновление не выполнено", false);
  } finally {
    button.disabled = false;
    button.classList.remove("spinning");
  }
});

async function runPrediction(remember) {
  const current = ++requestId;
  hideError();
  loading.hidden = false;
  predictionRoot.classList.add("is-loading");
  try {
    const opponent = opponentInput.value.trim();
    const response = await fetch(`/api/predict?opponent=${encodeURIComponent(opponent)}&remember=${remember}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Не удалось построить прогноз");
    if (current !== requestId) return;
    render(data);
    const freshness = data.sync?.skipped ? "данные актуальны" : "данные обновлены";
    setSystem(`${freshness} · модель обучается автоматически`, false);
  } catch (error) {
    if (current === requestId) showError(error.message);
  } finally {
    if (current === requestId) {
      loading.hidden = true;
      predictionRoot.classList.remove("is-loading");
    }
  }
}

function render(data) {
  if (data.prediction_available === false) {
    predictionRoot.innerHTML = renderUnavailable(data);
    return;
  }
  const fixture = data.fixture || {};
  const isBarcaHome = data.barcelona_side === "home";
  const opponentLogo = isBarcaHome ? fixture.away_logo : fixture.home_logo;
  const kickoff = formatKickoff(fixture.kickoff || data.match_date);
  const probs = data.outcome?.probabilities || {};
  const goals = data.goals || {};
  const backtest = data.data_quality?.backtest || {};
  const recommendation = data.recommended_bet || {};

  predictionRoot.innerHTML = `
    <section class="fixture-band">
      <div class="content-width fixture-layout">
        <div class="fixture-meta">
          <span class="competition-tag">${escapeHtml(data.competition || "")}</span>
          <span>${escapeHtml(data.stage || "Матч турнира")}</span>
          <span>${kickoff}</span>
          <span>${isBarcaHome ? "Барселона дома" : "Барселона в гостях"}</span>
        </div>
        <div class="teams-line">
          ${teamMark("Barcelona", "https://a.espncdn.com/i/teamlogos/soccer/500/83.png")}
          <div class="score-core">
            <span class="score-label">ПРОГНОЗ</span>
            <strong>${escapeHtml(data.exact_score?.score || "—")}</strong>
            <span>${percent(data.exact_score?.probability)} точный счет</span>
          </div>
          ${teamMark(data.opponent, opponentLogo)}
        </div>
        <p class="snapshot-note">${escapeHtml(data.prediction_snapshot?.message || "")}</p>
      </div>
    </section>

    <section class="content-width primary-grid">
      <article class="metric-panel outcome-panel">
        <div class="section-heading"><span>Основной исход</span><small>вероятности 1X2</small></div>
        <h2>${escapeHtml(data.outcome?.label || "—")}</h2>
        <div class="confidence-number">${percent(data.outcome?.confidence)}</div>
        ${probabilityRow("Победа Барселоны", probs.barcelona_win)}
        ${probabilityRow("Ничья", probs.draw)}
        ${probabilityRow(`Победа ${data.opponent}`, probs.opponent_win)}
        ${probabilityRow("Барселона не проиграет", probs.barcelona_not_lose, true)}
      </article>

      <article class="metric-panel">
        <div class="section-heading"><span>Голы</span><small>модель распределения счета</small></div>
        <div class="big-value">${number(goals.total_expected, 2)}</div>
        <p class="value-caption">ожидаемый тотал · ${number(goals.barcelona_expected, 2)} : ${number(goals.opponent_expected, 2)} xG</p>
        <div class="market-grid">
          ${marketCell("ТБ 1.5", goals.probabilities?.over_1_5)}
          ${marketCell("ТМ 1.5", goals.probabilities?.under_1_5)}
          ${marketCell("ТБ 2.5", goals.probabilities?.over_2_5)}
          ${marketCell("ТМ 2.5", goals.probabilities?.under_2_5)}
          ${marketCell("ТБ 3.5", goals.probabilities?.over_3_5)}
          ${marketCell("ТМ 3.5", goals.probabilities?.under_3_5)}
        </div>
      </article>

      ${statPanel("Угловые", data.corners, "xC")}
      ${statPanel("Фолы", data.fouls, "xF")}
    </section>

    <section class="recommendation-band">
      <div class="content-width recommendation-layout">
        <div>
          <span class="section-kicker">РЕКОМЕНДУЕМАЯ СТАВКА</span>
          <h2>${recommendation.eligible ? escapeHtml(recommendation.label) : "Нет ставки с порогом 75%"}</h2>
          <p>${escapeHtml(recommendation.note || "")}</p>
        </div>
        <div class="recommendation-probability ${recommendation.eligible ? "eligible" : ""}">
          <strong>${percent(recommendation.model_probability)}</strong>
          <span>оценка модели</span>
        </div>
      </div>
    </section>

    <section class="content-width comparison-section">
      <div class="section-title-row">
        <div><span class="section-kicker">ТАКТИЧЕСКИЙ МАТЧ-АП</span><h2>Форма и структура игры</h2></div>
        <div class="elo-line"><span>Elo ${data.strength?.barcelona_elo ?? "—"}</span><strong>${signed(data.strength?.difference)}</strong><span>${data.strength?.opponent_elo ?? "—"} Elo</span></div>
      </div>
      <div class="team-comparison">
        ${teamProfile(data.barcelona_profile, true)}
        ${teamProfile(data.opponent_profile, false)}
      </div>
    </section>

    <section class="content-width history-grid">
      ${recentTable(data.barcelona_profile)}
      ${recentTable(data.opponent_profile)}
    </section>

    <section class="detail-band">
      <div class="content-width detail-layout">
        <div class="lineup-grid">
          ${lineupPanel("Барселона", data.lineups?.barcelona)}
          ${lineupPanel(data.opponent, data.lineups?.opponent)}
        </div>
        <div class="support-grid">
          ${refereePanel(data.referee)}
          ${h2hPanel(data.h2h, data.opponent)}
        </div>
      </div>
    </section>

    <section class="content-width validation-section">
      <div class="section-title-row">
        <div><span class="section-kicker">WALK-FORWARD</span><h2>Честная проверка модели</h2></div>
        <span class="audit-chip">без результата прогнозируемого матча</span>
      </div>
      <div class="validation-grid">
        ${validationMetric("Исход", backtest.outcome_accuracy, "accuracy")}
        ${validationMetric("Точный счет", backtest.exact_score_accuracy, "accuracy")}
        ${validationMetric("Ошибка голов", backtest.goal_total_mae, "mae")}
        ${validationMetric("Ошибка угловых", backtest.corner_mae, "mae")}
        ${validationMetric("Ошибка фолов", backtest.foul_mae, "mae")}
      </div>
      <p class="audit-note">${escapeHtml(backtest.evaluation || data.data_quality?.leakage_guard || "")}</p>
    </section>

    ${resultBand(data)}
  `;
}

function renderUnavailable(data) {
  const fixture = data.fixture || {};
  const summary = data.result_summary || {};
  return `
    <section class="fixture-band">
      <div class="content-width unavailable">
        <span class="competition-tag">${escapeHtml(fixture.competition || "Матч")}</span>
        <h2>Barcelona — ${escapeHtml(data.opponent || "соперник")}</h2>
        <div class="actual-score">${escapeHtml(summary.actual || "—")}</div>
        <p>${escapeHtml(data.message || "Прогноз недоступен.")}</p>
        <p class="snapshot-note">${escapeHtml(data.fixture_status?.label || "")}</p>
      </div>
    </section>`;
}

function teamMark(name, logo) {
  return `<div class="team-mark">
    ${logo ? `<img src="${escapeAttribute(logo)}" alt="" width="74" height="74">` : `<span class="logo-fallback">${escapeHtml(name?.slice(0, 2) || "FC")}</span>`}
    <h2>${escapeHtml(name || "—")}</h2>
  </div>`;
}

function probabilityRow(label, value, accent = false) {
  const width = Math.max(0, Math.min(100, Number(value || 0) * 100));
  return `<div class="probability-row ${accent ? "accent" : ""}">
    <div><span>${escapeHtml(label)}</span><strong>${percent(value)}</strong></div>
    <div class="track"><i style="width:${width.toFixed(1)}%"></i></div>
  </div>`;
}

function marketCell(label, value) {
  return `<div><span>${escapeHtml(label)}</span><strong>${percent(value)}</strong></div>`;
}

function statPanel(title, payload = {}, prefix) {
  const interval = payload.interval_70 || [];
  const markets = Object.entries(payload.markets || {});
  return `<article class="metric-panel compact-stat">
    <div class="section-heading"><span>${escapeHtml(title)}</span><small>последние 10 + соперник</small></div>
    <div class="big-value">${payload.point ?? "—"}</div>
    <p class="value-caption">${prefix} ${number(payload.expected, 2)} · интервал ${number(interval[0], 1)}–${number(interval[1], 1)}</p>
    <div class="market-grid two">${markets.map(([key, value]) => marketCell(marketName(key), value)).join("")}</div>
  </article>`;
}

function teamProfile(profile = {}, barca) {
  const metrics = profile.metrics || {};
  return `<article class="team-profile ${barca ? "barca" : "opponent"}">
    <div class="profile-head">
      <div><span>${barca ? "BARÇA" : "СОПЕРНИК"}</span><h3>${escapeHtml(profile.team || "—")}</h3></div>
      <div class="formation"><strong>${escapeHtml(profile.formation || "—")}</strong><small>${profile.formation_source === "confirmed_lineup" ? "подтверждено" : "по последним матчам"}</small></div>
    </div>
    <p class="style-line">${escapeHtml(profile.style || "")}</p>
    ${profileMetric("Владение", metrics.possession)}
    ${profileMetric("Атака", metrics.attack)}
    ${profileMetric("Защита", metrics.defense)}
    ${profileMetric("Прессинг", metrics.pressing)}
    ${profileMetric("Фланги", metrics.width)}
    ${profileMetric("Темп", metrics.tempo)}
  </article>`;
}

function profileMetric(label, value) {
  const width = Math.max(0, Math.min(100, Number(value || 0)));
  return `<div class="profile-metric"><span>${escapeHtml(label)}</span><div class="track"><i style="width:${width}%"></i></div><strong>${Math.round(width)}%</strong></div>`;
}

function recentTable(profile = {}) {
  const rows = (profile.recent_matches || []).map((match) => `<tr>
    <td>${formatDate(match.date)}</td>
    <td>${escapeHtml(match.opponent || "—")}</td>
    <td><span class="result ${match.result || ""}">${escapeHtml(match.result || "—")}</span></td>
    <td class="score-cell">${escapeHtml(match.score || "—")}</td>
    <td>${escapeHtml(match.formation || "—")}</td>
  </tr>`).join("");
  return `<article class="history-table">
    <div class="table-head"><h3>${escapeHtml(profile.team || "Команда")} · последние ${profile.sample_size || 0}</h3><span>${number(profile.averages?.goals_for, 2)} забито / ${number(profile.averages?.goals_against, 2)} пропущено</span></div>
    <div class="table-scroll"><table><thead><tr><th>Дата</th><th>Соперник</th><th>Ф</th><th>Счет</th><th>Схема</th></tr></thead><tbody>${rows || `<tr><td colspan="5">Нет данных</td></tr>`}</tbody></table></div>
  </article>`;
}

function lineupPanel(title, lineup = {}) {
  const official = Boolean(lineup.official_available ?? lineup.confirmed);
  const display = lineup.display_lineup || (official ? lineup.official_lineup : lineup.predicted_lineup) || {};
  const players = display.players || [];
  const sampleMatches = lineup.predicted_lineup?.sample_matches || 0;
  const squad = lineup.squad_context || {};
  const signings = squad.recent_signings || [];
  const filtered = squad.filtered_departures || [];
  const playerRows = players.map((player) => {
    const alternatives = (player.alternatives || []).slice(0, 2);
    const alternativeText = alternatives.length
      ? `Альтернатива: ${alternatives.map((item) => `${escapeHtml(item.name)} ${percent(item.probability)}`).join(" · ")}`
      : "";
    return `<li>
      <span class="formation-slot" aria-label="Позиция в схеме">${escapeHtml(player.formation_place || "—")}</span>
      <div class="player-identity">
        <strong>${escapeHtml(player.name || "Игрок не определен")}</strong>
        <span>${escapeHtml(player.position || "Позиция не определена")}</span>
        ${alternativeText ? `<small>${alternativeText}</small>` : ""}
      </div>
      <span class="player-probability">${official ? "старт" : percent(player.probability)}</span>
    </li>`;
  }).join("");
  const emptyMessage = official
    ? "В протоколе пока недостаточно данных о позициях игроков."
    : "Недостаточно загруженных протоколов, чтобы надежно спрогнозировать 11 игроков.";
  const metaValue = official ? "ESPN" : percent(display.confidence || 0);
  const metaLabel = official ? "Источник" : "Уверенность XI";
  const squadInfo = squad.applied ? `<div class="squad-context">
    <div><strong>Текущая первая команда</strong><span>${squad.active_players || 0} игроков · официальный ростер FC Barcelona</span></div>
    ${signings.length ? `<div><strong>Новые подписания</strong><span>${signings.map(escapeHtml).join(" · ")}</span></div>` : ""}
    ${filtered.length ? `<div><strong>Вне текущего ростера</strong><span>${filtered.map(escapeHtml).join(" · ")}</span></div>` : ""}
  </div>` : "";
  return `<article class="detail-panel lineup-panel">
    <div class="detail-title"><h3>${official ? "Официальный состав" : "Прогноз состава"} · ${escapeHtml(title || "")}</h3><span class="status-label ${official ? "confirmed" : "pending"}">${official ? "официальный" : "официального нет"}</span></div>
    <p class="lineup-message">${escapeHtml(lineup.message || (official ? "Официальный состав опубликован." : "Официального состава пока нет. Ниже показан прогноз модели."))}</p>
    <div class="lineup-meta"><span>Схема</span><strong>${escapeHtml(display.formation || lineup.formation || "не определена")}</strong><span>${metaLabel}</span><strong>${metaValue}</strong><span>Матчей в выборке</span><strong>${sampleMatches}</strong></div>
    ${squadInfo}
    <ol class="squad-list">${playerRows || `<li class="empty-lineup">${emptyMessage}</li>`}</ol>
    ${!official && lineup.predicted_lineup?.selection_context ? `<small class="selection-context">${escapeHtml(lineup.predicted_lineup.selection_context)}</small>` : ""}
    ${!official && lineup.predicted_lineup?.method ? `<small class="lineup-method">${escapeHtml(lineup.predicted_lineup.method)}</small>` : ""}
  </article>`;
}

function refereePanel(referee = {}) {
  return `<article class="detail-panel">
    <div class="detail-title"><h3>Судья</h3><span class="status-label">${referee.matches || 0} матчей</span></div>
    <div class="detail-big">${escapeHtml(referee.name || "Не назначен")}</div>
    <p>${referee.avg_fouls == null ? "Среднее появится после назначения." : `${number(referee.avg_fouls, 2)} фола в среднем`}</p>
    <small>${escapeHtml(referee.message || "")}</small>
  </article>`;
}

function h2hPanel(h2h = {}, opponent) {
  const matches = (h2h.matches || []).slice(0, 5).map((item) => `<li><span>${formatDate(item.date)}</span><strong>${escapeHtml(item.score)}</strong><small>вес ${number(item.weight, 2)}</small></li>`).join("");
  return `<article class="detail-panel">
    <div class="detail-title"><h3>Очные · ${escapeHtml(opponent || "")}</h3><span class="status-label">${h2h.sample || 0} матчей</span></div>
    <ul class="h2h-list">${matches || "<li>Нет матчей в базе</li>"}</ul>
    <small>${escapeHtml(h2h.weighting || "")}</small>
  </article>`;
}

function validationMetric(label, value, kind) {
  const displayed = value == null ? "—" : kind === "accuracy" ? percent(value) : number(value, 2);
  return `<div><span>${escapeHtml(label)}</span><strong>${displayed}</strong><small>${kind === "accuracy" ? "доля попаданий" : "MAE"}</small></div>`;
}

function resultBand(data) {
  const summary = data.result_summary || {};
  const status = data.fixture_status || {};
  const actual = summary.actual || (status.state === "live" ? "матч идет" : "матч еще не начался");
  return `<section class="result-band"><div class="content-width result-layout">
    <div><span>ПРЕДМАТЧЕВЫЙ ПРОГНОЗ</span><strong>${escapeHtml(summary.prediction || data.exact_score?.score || "—")}</strong></div>
    <div class="result-divider"></div>
    <div><span>ФАКТИЧЕСКИЙ СЧЕТ</span><strong>${escapeHtml(actual)}</strong></div>
    <p>${escapeHtml(status.label || "")}${status.detail ? ` · ${escapeHtml(status.detail)}` : ""}</p>
  </div></section>`;
}

async function loadOpponents() {
  try {
    const response = await fetch("/api/opponents");
    const data = await response.json();
    if (!response.ok) return;
    document.querySelector("#opponents").innerHTML = (data.opponents || [])
      .map((item) => `<option value="${escapeAttribute(item.name)}"></option>`)
      .join("");
  } catch (_) {
    // Search remains usable if suggestions are temporarily unavailable.
  }
}

function setSystem(message, busy) {
  document.querySelector("#system-label").textContent = message;
  document.querySelector(".system-state").classList.toggle("busy", Boolean(busy));
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function formatKickoff(value) {
  if (!value) return "Дата уточняется";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return escapeHtml(value);
  return new Intl.DateTimeFormat("ru-RU", { dateStyle: "long", timeStyle: "short", timeZone: "Europe/Moscow" }).format(date) + " МСК";
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(`${value}T12:00:00Z`);
  return new Intl.DateTimeFormat("ru-RU", { day: "2-digit", month: "2-digit", year: "2-digit" }).format(date);
}

function marketName(key) {
  return key.replace("over_", "ТБ ").replace("under_", "ТМ ").replaceAll("_", ".");
}

function number(value, digits = 1) {
  return value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits);
}

function percent(value) {
  return value == null || Number.isNaN(Number(value)) ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
}

function signed(value) {
  if (value == null) return "—";
  return `${Number(value) >= 0 ? "+" : ""}${Number(value)}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

loadOpponents();
runPrediction(false);
