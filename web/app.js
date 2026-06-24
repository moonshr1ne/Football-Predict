const prediction = document.querySelector("#prediction");
let activePredictionRequest = 0;

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
      <h3>Голы</h3>
      ${goalTotalBlock(data.goal_total)}
    </article>
    <article class="card">
      <h3>Точные счета</h3>
      <div class="pill-row">${scorePills(data)}</div>
    </article>
    <article class="card">
      <h3>Вероятности</h3>
      ${marketTable(data)}
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
      ${dataQualityBlock(data.data_quality)}
      ${warnings || "<p class='muted'>Предупреждений нет.</p>"}
    </article>
    <article class="card wide">
      <h3>Прогноз и факт</h3>
      ${resultSummaryBlock(data)}
    </article>
  `;
}

function scorePills(data) {
  const items = data.exact_score_probabilities?.length
    ? data.exact_score_probabilities
    : (data.exact_scores || []).map((score) => ({ score, probability: null }));
  return items
    .map((item) => {
      const probabilityText =
        item.probability == null ? "" : `<span class="pill-prob">${probability(item.probability)}</span>`;
      return `<span class="pill">${escapeHtml(item.score)}${probabilityText}</span>`;
    })
    .join("");
}

function marketTable(data) {
  const markets = data.markets?.length
    ? data.markets
    : [
        { code: "П1", label: `Победа ${data.home_team}`, probability: data.probabilities?.["П1"] },
        { code: "X", label: "Ничья", probability: data.probabilities?.X },
        { code: "П2", label: `Победа ${data.away_team}`, probability: data.probabilities?.["П2"] },
      ];
  return `
    <table class="compact">
      ${markets
        .map(
          (market) => `
            <tr>
              <th>${escapeHtml(market.code)}</th>
              <td>${escapeHtml(market.label)}</td>
              <td>${probability(market.probability)}</td>
            </tr>
          `
        )
        .join("")}
    </table>
  `;
}

function goalTotalBlock(goalTotal) {
  if (!goalTotal) {
    return "<p class='muted'>нет расчета</p>";
  }
  const probabilities = goalTotal.probabilities || {};
  const likely = (goalTotal.most_likely_totals || [])
    .map((item) => `${escapeHtml(item.goals)} (${probability(item.probability)})`)
    .join(", ");
  return `
    <div class="metric">${Number(goalTotal.expected ?? 0).toFixed(2)}</div>
    <p class="sub">${escapeHtml(goalTotal.label || "тотал")}</p>
    <table class="compact">
      <tr><th>ТБ 2.5</th><td>${probability(probabilities.over_2_5)}</td><th>ТБ 3.5</th><td>${probability(probabilities.over_3_5)}</td></tr>
      <tr><th>ТБ 4.5</th><td>${probability(probabilities.over_4_5)}</td><th>Чаще всего</th><td>${likely || "нет"}</td></tr>
    </table>
  `;
}

function statsTable(stats) {
  return `
    <table>
      <tr><th>Матчи</th><td>${stats.sample_size}</td><th>Форма</th><td>${stats.wins}-${stats.draws}-${stats.losses}</td></tr>
      <tr><th>Голы</th><td>${stats.avg_goals_for} / ${stats.avg_goals_against}</td><th>Очки</th><td>${stats.points_per_match}</td></tr>
      <tr><th>Угловые</th><td>${stats.avg_total_corners ?? "нет"}</td><th>Сухие</th><td>${stats.clean_sheets}</td></tr>
      <tr><th>Владение</th><td>${stats.avg_possession ?? "нет"}%</td><th>Удары</th><td>${stats.avg_shots_for ?? "нет"} / ${stats.avg_shots_against ?? "нет"}</td></tr>
      <tr><th>В створ</th><td>${stats.avg_shots_on_target_for ?? "нет"}</td><th>Источник</th><td>${stats.recent?.[0]?.source || "local"}</td></tr>
      <tr><th>Угл. выборка</th><td>${stats.corner_samples}</td><th>Влад/удары</th><td>${stats.possession_samples}/${stats.shot_samples}</td></tr>
    </table>
  `;
}

function dataQualityBlock(quality) {
  if (!quality) {
    return "<p class='muted'>Сводка качества пока не рассчитана.</p>";
  }
  const backtest = quality.backtest || {};
  const targets = backtest.targets || {};
  const targetStatus = backtest.target_status || {};
  return `
    <table>
      <tr><th>Общая база</th><td>${percent(quality.score)}</td><th>Участники ЧМ</th><td>${quality.participants || 0}</td></tr>
      <tr><th>Матчи команд</th><td>${quality.home_matches || 0} / ${quality.away_matches || 0}</td><th>Богатые матчи</th><td>${quality.home_rich_matches || 0} / ${quality.away_rich_matches || 0}</td></tr>
      <tr><th>Бэктест</th><td>${backtest.matches || 0} матчей</td><th>Исходы</th><td>${backtest.outcome_accuracy == null ? "нет" : percent(backtest.outcome_accuracy)}</td></tr>
      <tr><th>Точные счета</th><td>${backtest.exact_score_accuracy == null ? "нет" : percent(backtest.exact_score_accuracy)}</td><th>Ошибка угл.</th><td>${backtest.corner_mae ?? "нет"}</td></tr>
      <tr><th>Цель исходов</th><td>${targetCell(backtest.outcome_accuracy, targets.outcome_accuracy, targetStatus.outcome_accuracy, "higher")}</td><th>Цель счетов</th><td>${targetCell(backtest.exact_score_accuracy, targets.exact_score_accuracy, targetStatus.exact_score_accuracy, "higher")}</td></tr>
      <tr><th>Цель угловых</th><td>${targetCell(backtest.corner_mae, targets.corner_mae, targetStatus.corner_mae, "lower", false)}</td><th>Угл. ±1</th><td>${backtest.corner_within_one_rate == null ? "нет" : probability(backtest.corner_within_one_rate)}</td></tr>
    </table>
  `;
}

function targetCell(actual, target, met, direction, asPercent = true) {
  if (actual == null || target == null) {
    return "нет";
  }
  const actualText = asPercent ? probability(actual) : Number(actual).toFixed(2);
  const targetText = asPercent ? probability(target) : Number(target).toFixed(2);
  const sign = direction === "lower" ? "≤" : "≥";
  const mark = met ? "достигнута" : direction === "lower" ? "выше цели" : "ниже цели";
  return `${actualText} / ${sign} ${targetText} · ${mark}`;
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
  const status = fixture.completed
    ? "завершен"
    : fixture.in_progress
      ? "матч идет"
      : (fixture.status_detail || fixture.status || "запланирован");
  const kickoff = fixture.kickoff ? ` · ${escapeHtml(fixture.kickoff)}` : "";
  return `<p><strong>Матч найден:</strong> ${escapeHtml(fixture.date)} · ${escapeHtml(status)}${kickoff}</p>`;
}

function resultSummaryBlock(data) {
  const summary = data.result_summary || {};
  const predicted = summary.predicted || {};
  const predictedScores = scoreListText(predicted.scores || data.exact_score_probabilities || []);
  const predictedLine = `<p><strong>Предикт:</strong> ${escapeHtml(
    predicted.outcome_label || data.market_pick
  )}; счета ${predictedScores}; голы ${Number(predicted.goal_total?.expected ?? data.goal_total?.expected ?? 0).toFixed(2)}; угловые ${Number(predicted.corners ?? data.predicted_corners).toFixed(2)}</p>`;

  if (summary.status === "completed" && summary.actual) {
    const cornerText = summary.actual.corners == null ? "угловые: нет данных" : `угловые ${summary.actual.corners}`;
    const cornerError = summary.corner_error == null ? "" : `, ошибка угловых ${Math.abs(summary.corner_error).toFixed(2)}`;
    return `
      ${predictedLine}
      <p><strong>Факт:</strong> ${escapeHtml(summary.actual.outcome_label)}; счет ${escapeHtml(summary.actual.score)}, ${cornerText}${cornerError}</p>
      <p class="sub">Исход: ${summary.outcome_hit ? "угадан" : "мимо"}, точный счет: ${summary.score_hit ? "угадан" : "мимо"}.</p>
    `;
  }

  if (summary.status === "live") {
    const score = summary.actual?.score ? ` Текущий счет ${escapeHtml(summary.actual.score)}.` : "";
    return `
      ${predictedLine}
      <p><strong>Факт:</strong> матч сейчас идет.${score} Финальный счет еще не известен.</p>
    `;
  }

  if (summary.status === "scheduled") {
    return `
      ${predictedLine}
      <p><strong>Факт:</strong> матч еще не начался.</p>
    `;
  }

  return `
    ${predictedLine}
    <p><strong>Факт:</strong> матч не найден в расписании, настоящий счет пока неизвестен.</p>
  `;
}

function scoreListText(items) {
  if (!items.length) {
    return "нет";
  }
  return items
    .map((item) => {
      const score = typeof item === "string" ? item : item.score;
      const probabilityValue = typeof item === "string" ? null : item.probability;
      return probabilityValue == null ? escapeHtml(score) : `${escapeHtml(score)} (${probability(probabilityValue)})`;
    })
    .join(", ");
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

function probability(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "нет";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
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
  const requestId = ++activePredictionRequest;
  const matchup = document.querySelector("#matchup").value;
  const homeVenue = document.querySelector("#home-venue").checked;
  prediction.innerHTML = `
    <article class="card">
      <h3>Обновляю прогноз</h3>
      <p class="muted">Собираю матчи, тактику и дату.</p>
    </article>
  `;
  try {
    const data = await getJson(`/api/predict?matchup=${encodeURIComponent(matchup)}&home_venue=${homeVenue}&remember=${remember}`);
    if (requestId === activePredictionRequest) {
      renderPrediction(data);
    }
  } catch (error) {
    if (requestId === activePredictionRequest) {
      prediction.innerHTML = `
        <article class="card">
          <h3>Не смог построить прогноз</h3>
          <p class="warn">${escapeHtml(error.message)}</p>
        </article>
      `;
    }
  }
}

runPrediction(false);
