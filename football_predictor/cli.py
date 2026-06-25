from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date

from .aliases import parse_matchup
from .context_tools import update_team_context, update_team_tactics
from .autocheck import AutoChecker
from .data_store import DataStore
from .learning import OnlineLearner
from .predictor import MatchPredictor
from .providers import EspnWorldCupProvider, ProviderError
from .server import run_server
from .sync import WorldCupDataSync


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nfp", description="Прогнозы матчей сборных по футболу.")
    subparsers = parser.add_subparsers(dest="command")

    predict = subparsers.add_parser("predict", help="Сделать прогноз: nfp predict \"Англия, Гана\"")
    predict.add_argument("matchup")
    predict.add_argument("--date", help="Дата матча YYYY-MM-DD. Если указана, прогноз попадет в очередь автопроверки.")
    predict.add_argument("--no-auto-date", action="store_true", help="Не искать дату матча автоматически.")
    predict.add_argument("--home-venue", action="store_true", help="Считать, что первая команда играет дома.")
    predict.add_argument("--json", action="store_true")

    result = subparsers.add_parser("result", help="Внести итог матча и обновить модель.")
    result.add_argument("matchup")
    result.add_argument("--date", default=date.today().isoformat())
    result.add_argument("--score", required=True, help="Например: 2-1")
    result.add_argument("--corners", type=float, help="Общее количество угловых.")
    result.add_argument("--home-corners", type=float)
    result.add_argument("--away-corners", type=float)
    result.add_argument("--home-venue", action="store_true")
    result.add_argument("--competition", default="")
    result.add_argument("--stage", default="")
    result.add_argument("--json", action="store_true")

    check = subparsers.add_parser("check", help="Проверить результат через API-Football и обучиться.")
    check.add_argument("matchup")
    check.add_argument("--date")
    check.add_argument("--home-venue", action="store_true")
    check.add_argument("--json", action="store_true")

    auto_check = subparsers.add_parser("auto-check", help="Проверить все прогнозы в очереди через API-Football.")
    auto_check.add_argument("--limit", type=int, default=25)
    auto_check.add_argument("--json", action="store_true")

    sync = subparsers.add_parser("sync-world-cup", help="Загрузить прошедшие матчи ЧМ-2026 и обновить тактические профили.")
    sync.add_argument("--json", action="store_true")

    train = subparsers.add_parser("train", help="Переобучить модель на всей накопленной истории матчей.")
    train.add_argument("--epochs", type=int, default=80, help="Сколько раз прогнать историю, по умолчанию 80.")
    train.add_argument("--json", action="store_true")

    watch = subparsers.add_parser("watch", help="Постоянно проверять очередь прогнозов и обучаться.")
    watch.add_argument("--interval", type=int, default=3600, help="Пауза между проверками в секундах.")
    watch.add_argument("--limit", type=int, default=25)

    context = subparsers.add_parser("context", help="Добавить травмы, мотивацию и заметки по сборной.")
    context.add_argument("team")
    context.add_argument("--motivation", type=float, help="0..1")
    context.add_argument("--lineup-strength", type=float, help="0..1, ожидаемая сила состава")
    context.add_argument("--note")
    context.add_argument("--injury", help='Формат: "Игрок:out:0.6"')
    context.add_argument("--clear-injuries", action="store_true")
    context.add_argument("--json", action="store_true")

    tactics = subparsers.add_parser("tactics", help="Обновить схему и стиль игры сборной.")
    tactics.add_argument("team")
    tactics.add_argument("--formation")
    tactics.add_argument("--style")
    tactics.add_argument("--build-up")
    tactics.add_argument("--primary-attack")
    tactics.add_argument("--defensive-block")
    tactics.add_argument("--possession", type=float, dest="possession_intent")
    tactics.add_argument("--pressing", type=float)
    tactics.add_argument("--line-height", type=float)
    tactics.add_argument("--defense", type=float, dest="defensive_solidity")
    tactics.add_argument("--width", type=float, dest="attack_width")
    tactics.add_argument("--central", type=float, dest="central_progression")
    tactics.add_argument("--directness", type=float)
    tactics.add_argument("--creation", type=float, dest="chance_creation")
    tactics.add_argument("--transition-attack", type=float)
    tactics.add_argument("--transition-defense", type=float)
    tactics.add_argument("--set-pieces", type=float, dest="set_piece_threat")
    tactics.add_argument("--tempo", type=float)
    tactics.add_argument("--note")
    tactics.add_argument("--json", action="store_true")

    serve = subparsers.add_parser("serve", help="Запустить локальный веб-интерфейс.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        args.command = "predict"
        args.matchup = "Англия, Гана"
        args.date = None
        args.no_auto_date = False
        args.home_venue = False
        args.json = False

    store = DataStore()

    if args.command == "predict":
        home_team, away_team = parse_matchup(args.matchup, store.resolver)
        sync_info = WorldCupDataSync(store).sync_all()
        fixture = None
        lookup_warnings = []
        if sync_info.get("imported"):
            lookup_warnings.append(_sync_prediction_warning(sync_info))
        match_date = args.date
        if not match_date and not args.no_auto_date:
            fixture, fixture_warnings = _find_fixture_for_prediction(home_team, away_team)
            lookup_warnings.extend(fixture_warnings)
            match_date = fixture.get("date") if fixture else None
        prediction = MatchPredictor(store).predict(
            home_team,
            away_team,
            neutral=not args.home_venue,
            match_date=match_date,
            fixture=fixture,
            extra_warnings=lookup_warnings,
        )
        if args.json:
            print(json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(prediction.short_text())
            if match_date:
                print(f"Дата матча найдена: {match_date}. Прогноз добавлен в очередь автопроверки.")
            print(_details_text(prediction.to_dict()))
        return 0

    if args.command == "result":
        home_team, away_team = parse_matchup(args.matchup, store.resolver)
        home_goals, away_goals = _parse_score(args.score)
        baseline = store.latest_prediction(home_team, away_team, match_date=args.date, status="pending")
        review = OnlineLearner(store).record_result(
            home_team=home_team,
            away_team=away_team,
            date=args.date,
            home_goals=home_goals,
            away_goals=away_goals,
            corners_total=args.corners,
            home_corners=args.home_corners,
            away_corners=args.away_corners,
            competition=args.competition,
            stage=args.stage,
            neutral=not args.home_venue,
            baseline_prediction=baseline,
        )
        if baseline and baseline.get("prediction_id"):
            store.update_prediction(
                baseline["prediction_id"],
                {"status": "reviewed", "review": review, "reviewed_at": review["updated_at"]},
            )
        print(json.dumps(review, ensure_ascii=False, indent=2) if args.json else _review_text(review))
        return 0

    if args.command == "auto-check":
        try:
            summary = AutoChecker(store).check_pending(limit=args.limit)
        except ProviderError as exc:
            parser.error(str(exc))
        print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else _auto_check_text(summary))
        return 0

    if args.command == "sync-world-cup":
        summary = WorldCupDataSync(store).sync_all(force=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else _sync_text(summary))
        return 0

    if args.command == "train":
        syncer = WorldCupDataSync(store)
        sync_summary = syncer.sync_all(force=True)
        summary = syncer.retrain_model_from_history(epochs=args.epochs)
        summary["sync"] = sync_summary
        print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else _train_text(summary))
        return 0

    if args.command == "watch":
        try:
            checker = AutoChecker(store)
        except ProviderError as exc:
            parser.error(str(exc))
        print(f"Автопроверка запущена. Интервал: {args.interval} сек.")
        while True:
            try:
                summary = checker.check_pending(limit=args.limit)
                print(_auto_check_text(summary))
            except ProviderError as exc:
                print(f"Автопроверка временно недоступна: {exc}")
            time.sleep(args.interval)

    if args.command == "check":
        home_team, away_team = parse_matchup(args.matchup, store.resolver)
        try:
            provider = EspnWorldCupProvider()
            result = provider.get_finished_result(home_team, away_team, args.date)
        except ProviderError as exc:
            parser.error(str(exc))
        baseline = store.latest_prediction(home_team, away_team, match_date=result["date"], status="pending")
        review = OnlineLearner(store).record_result(
            home_team=home_team,
            away_team=away_team,
            date=result["date"],
            home_goals=int(result["home_goals"]),
            away_goals=int(result["away_goals"]),
            home_corners=result.get("home_corners"),
            away_corners=result.get("away_corners"),
            home_possession=result.get("home_possession"),
            away_possession=result.get("away_possession"),
            home_shots=result.get("home_shots"),
            away_shots=result.get("away_shots"),
            home_shots_on_target=result.get("home_shots_on_target"),
            away_shots_on_target=result.get("away_shots_on_target"),
            home_fouls=result.get("home_fouls"),
            away_fouls=result.get("away_fouls"),
            referee=(result.get("referee") or {}).get("name") if isinstance(result.get("referee"), dict) else result.get("referee"),
            neutral=not args.home_venue,
            source=result.get("source", "api"),
            baseline_prediction=baseline,
        )
        if baseline and baseline.get("prediction_id"):
            store.update_prediction(
                baseline["prediction_id"],
                {"status": "reviewed", "review": review, "reviewed_at": review["updated_at"]},
            )
        print(json.dumps(review, ensure_ascii=False, indent=2) if args.json else _review_text(review))
        return 0

    if args.command == "context":
        team = store.resolver.resolve(args.team)
        updated = update_team_context(
            store,
            team,
            motivation=args.motivation,
            lineup_strength=args.lineup_strength,
            note=args.note,
            injury=args.injury,
            clear_injuries=args.clear_injuries,
        )
        print(json.dumps(updated, ensure_ascii=False, indent=2) if args.json else f"Обновил контекст для {team}.")
        return 0

    if args.command == "tactics":
        team = store.resolver.resolve(args.team)
        updated = update_team_tactics(
            store,
            team,
            formation=args.formation,
            style=args.style,
            build_up=args.build_up,
            primary_attack=args.primary_attack,
            defensive_block=args.defensive_block,
            possession_intent=args.possession_intent,
            pressing=args.pressing,
            line_height=args.line_height,
            defensive_solidity=args.defensive_solidity,
            attack_width=args.attack_width,
            central_progression=args.central_progression,
            directness=args.directness,
            chance_creation=args.chance_creation,
            transition_attack=args.transition_attack,
            transition_defense=args.transition_defense,
            set_piece_threat=args.set_piece_threat,
            tempo=args.tempo,
            note=args.note,
        )
        print(json.dumps(updated, ensure_ascii=False, indent=2) if args.json else f"Обновил тактику для {team}.")
        return 0

    if args.command == "serve":
        run_server(args.host, args.port)
        return 0

    parser.error(f"Неизвестная команда: {args.command}")
    return 2


def _parse_score(value: str) -> tuple[int, int]:
    parts = value.replace(":", "-").split("-")
    if len(parts) != 2:
        raise ValueError("Счет должен быть в формате 2-1")
    return int(parts[0]), int(parts[1])


def _find_fixture_for_prediction(home_team: str, away_team: str) -> tuple[dict | None, list[str]]:
    try:
        fixture = EspnWorldCupProvider().find_fixture(home_team, away_team)
        return fixture, []
    except ProviderError as exc:
        return None, [f"Автодата: {exc} Можно указать дату вручную, если матч есть вне найденного окна."]


def _sync_prediction_warning(sync_info: dict) -> str:
    recent = "уже свежие" if sync_info.get("skipped_full_sync") else f"{sync_info.get('recent_imported', 0)}"
    action = "База проверена" if sync_info.get("skipped_full_sync") else "База обновлена"
    return (
        f"{action}: участников {sync_info.get('participants', 0)}, "
        f"last-10 матчей {recent}, матчей ЧМ {sync_info.get('imported', 0)}, "
        f"тактических профилей {sync_info.get('profiles_updated', 0)}, "
        f"судей {sync_info.get('referees_updated', 0)}, "
        f"обучающих матчей {sync_info.get('trained', 0)}."
    )


def _details_text(data: dict) -> str:
    markets = data.get("markets") or [
        {"code": "П1", "label": f"Победа {data['home_team']}", "probability": data["probabilities"]["П1"]},
        {"code": "X", "label": "Ничья", "probability": data["probabilities"]["X"]},
        {"code": "П2", "label": f"Победа {data['away_team']}", "probability": data["probabilities"]["П2"]},
    ]
    market_text = ", ".join(f"{item['code']} {float(item['probability']):.1%}" for item in markets)
    score_items = data.get("exact_score_probabilities") or [
        {"score": score, "probability": None} for score in data.get("exact_scores", [])
    ]
    score_text = ", ".join(
        item["score"] if item.get("probability") is None else f"{item['score']} ({float(item['probability']):.1%})"
        for item in score_items
    )
    goal_total = data.get("goal_total", {})
    goal_probabilities = goal_total.get("probabilities", {})
    foul_forecast = data.get("foul_forecast", {})
    foul_probabilities = foul_forecast.get("probabilities", {})
    referee = foul_forecast.get("referee", {})
    referee_text = referee.get("name") or "судья пока неизвестен"
    if referee.get("avg_fouls") is not None:
        referee_text += f", среднее {float(referee['avg_fouls']):.2f}"
    likely_totals = ", ".join(
        f"{item['goals']} ({float(item['probability']):.1%})"
        for item in goal_total.get("most_likely_totals", [])
    )
    lines = [
        f"Вероятности: {market_text}",
        f"Точный счет: {score_text}",
        f"Голы: ожидание {float(goal_total.get('expected', 0)):.2f}, "
        f"ТБ2.5 {float(goal_probabilities.get('over_2_5', 0)):.1%}, "
        f"ТМ2.5 {float(goal_probabilities.get('under_2_5', 0)):.1%}, "
        f"ТБ3.5 {float(goal_probabilities.get('over_3_5', 0)):.1%}, "
        f"ТМ3.5 {float(goal_probabilities.get('under_3_5', 0)):.1%}; "
        f"чаще всего {likely_totals or 'нет'}",
        f"Фолы: ожидание {float(foul_forecast.get('expected', 0)):.2f}, "
        f"ТБ24.5 {float(foul_probabilities.get('over_24_5', 0)):.1%}, "
        f"ТМ24.5 {float(foul_probabilities.get('under_24_5', 0)):.1%}; {referee_text}",
        f"xG: {data['home_team']} {data['expected_goals'][data['home_team']]}, {data['away_team']} {data['expected_goals'][data['away_team']]}",
        f"Последние 10: {data['home_team']} {data['home_stats']['wins']}-{data['home_stats']['draws']}-{data['home_stats']['losses']}, "
        f"{data['away_team']} {data['away_stats']['wins']}-{data['away_stats']['draws']}-{data['away_stats']['losses']}",
        f"Турнир: {data['match_context']['competition']}, важность {data['match_context']['importance']}",
        f"Тактика: {data['home_team']} {data['home_tactics']['formation']} vs {data['away_team']} {data['away_tactics']['formation']}; "
        f"{data['tactical_matchup']['summary']}",
    ]
    reports = data.get("team_reports", {})
    if reports:
        for team in (data["home_team"], data["away_team"]):
            report = reports.get(team, {})
            if report:
                lines.append(
                    f"{team}: {report.get('level')}, атака {float(report.get('attack_score', 0)):.0%}, "
                    f"оборона {float(report.get('defense_score', 0)):.0%}, xG {float(report.get('expected_goals', 0)):.2f}."
                )
    quality = data.get("data_quality", {})
    if quality:
        backtest = quality.get("backtest", {})
        lines.append(
            f"Качество данных: участники {quality.get('participants', 0)}, "
            f"last-10 {quality.get('home_matches', 0)}/{quality.get('away_matches', 0)}, "
            f"богатые матчи {quality.get('home_rich_matches', 0)}/{quality.get('away_rich_matches', 0)}, "
            f"бэктест {backtest.get('matches', 0)} матчей."
        )
    if data.get("fixture"):
        fixture = data["fixture"]
        lines.insert(0, f"Матч найден: {fixture['date']} ({fixture.get('status_detail') or fixture.get('status') or 'scheduled'})")
    if data.get("result_summary"):
        summary = data["result_summary"]
        predicted = summary.get("predicted", {})
        actual = summary.get("actual")
        lines.append(
            f"Предикт: {predicted.get('outcome_label', data['market_pick'])}; угловые {predicted.get('corners', data['predicted_corners'])}; фолы {predicted.get('fouls', {}).get('expected', data.get('foul_forecast', {}).get('expected'))}."
        )
        if summary.get("status") == "completed" and actual:
            foul_fact = "" if actual.get("fouls") is None else f", фолы {actual.get('fouls')}"
            lines.append(f"Факт: {actual.get('outcome_label')} {actual.get('score')}{foul_fact}.")
        elif summary.get("status") == "live":
            lines.append(f"Факт: матч идет{'; счет ' + actual.get('score') if actual and actual.get('score') else ''}.")
        elif summary.get("status") == "scheduled":
            lines.append("Факт: матч еще не начался.")
        else:
            lines.append(f"Факт: {summary.get('message', 'настоящий счет пока неизвестен.')}")
    if data["warnings"]:
        lines.append("Важно: " + " ".join(data["warnings"]))
    return "\n".join(lines)


def _review_text(review: dict) -> str:
    hit = "да" if review["outcome_hit"] else "нет"
    score_hit = "да" if review["score_hit"] else "нет"
    corner = "" if review["corner_error"] is None else f", ошибка угловых {review['corner_error']:+.2f}"
    fouls = "" if review.get("foul_error") is None else f", ошибка фолов {review['foul_error']:+.2f}"
    return f"Проверено: исход угадан: {hit}, точный счет: {score_hit}{corner}{fouls}. Модель обновлена."


def _auto_check_text(summary: dict) -> str:
    return (
        f"Очередь проверена: обработано {summary['checked']}, "
        f"обучено {summary['learned']}, ожидают дальше {summary['pending']}, ошибок {summary['errors']}."
    )


def _sync_text(summary: dict) -> str:
    if summary.get("error"):
        return f"Синхронизация недоступна: {summary['error']}"
    return (
        f"База обновлена: участников {summary.get('participants', 0)}, "
        f"last-10 матчей {summary.get('recent_imported', 0)}, "
        f"матчей ЧМ {summary['imported']}, "
        f"тактических профилей {summary['profiles_updated']}, "
        f"обучающих матчей {summary.get('trained', 0)}."
    )


def _train_text(summary: dict) -> str:
    backtest = summary.get("backtest", {})
    kept = " Сохранена предыдущая версия: новая попытка была слабее." if summary.get("kept_previous") else ""
    return (
        f"Модель дообучена: {summary.get('unique_matches', 0)} матчей, "
        f"{summary.get('epochs', 1)} эпохи, {summary.get('trained', 0)} тренировочных прогонов. "
        f"Бэктест: исходы {float(backtest.get('outcome_accuracy', 0)):.1%}, "
        f"точные счета {float(backtest.get('exact_score_accuracy', 0)):.1%}, "
        f"ошибка угловых {backtest.get('corner_mae', 'нет')}.{kept}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
