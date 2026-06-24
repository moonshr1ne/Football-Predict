from __future__ import annotations

import argparse
import json
from datetime import date

from .aliases import parse_matchup
from .context_tools import update_team_context
from .data_store import DataStore
from .learning import OnlineLearner
from .predictor import MatchPredictor
from .providers import ApiFootballProvider, ProviderError
from .server import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nfp", description="Прогнозы матчей сборных по футболу.")
    subparsers = parser.add_subparsers(dest="command")

    predict = subparsers.add_parser("predict", help="Сделать прогноз: nfp predict \"Англия, Гана\"")
    predict.add_argument("matchup")
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
    check.add_argument("--date", required=True)
    check.add_argument("--home-venue", action="store_true")
    check.add_argument("--json", action="store_true")

    context = subparsers.add_parser("context", help="Добавить травмы, мотивацию и заметки по сборной.")
    context.add_argument("team")
    context.add_argument("--motivation", type=float, help="0..1")
    context.add_argument("--note")
    context.add_argument("--injury", help='Формат: "Игрок:out:0.6"')
    context.add_argument("--clear-injuries", action="store_true")
    context.add_argument("--json", action="store_true")

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
        args.home_venue = False
        args.json = False

    store = DataStore()

    if args.command == "predict":
        home_team, away_team = parse_matchup(args.matchup, store.resolver)
        prediction = MatchPredictor(store).predict(home_team, away_team, neutral=not args.home_venue)
        if args.json:
            print(json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(prediction.short_text())
            print(_details_text(prediction.to_dict()))
        return 0

    if args.command == "result":
        home_team, away_team = parse_matchup(args.matchup, store.resolver)
        home_goals, away_goals = _parse_score(args.score)
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
        )
        print(json.dumps(review, ensure_ascii=False, indent=2) if args.json else _review_text(review))
        return 0

    if args.command == "check":
        home_team, away_team = parse_matchup(args.matchup, store.resolver)
        try:
            provider = ApiFootballProvider()
            result = provider.get_finished_result(home_team, away_team, args.date)
        except ProviderError as exc:
            parser.error(str(exc))
        review = OnlineLearner(store).record_result(
            home_team=home_team,
            away_team=away_team,
            date=result["date"],
            home_goals=int(result["home_goals"]),
            away_goals=int(result["away_goals"]),
            home_corners=result.get("home_corners"),
            away_corners=result.get("away_corners"),
            neutral=not args.home_venue,
            source=result.get("source", "api"),
        )
        print(json.dumps(review, ensure_ascii=False, indent=2) if args.json else _review_text(review))
        return 0

    if args.command == "context":
        team = store.resolver.resolve(args.team)
        updated = update_team_context(
            store,
            team,
            motivation=args.motivation,
            note=args.note,
            injury=args.injury,
            clear_injuries=args.clear_injuries,
        )
        print(json.dumps(updated, ensure_ascii=False, indent=2) if args.json else f"Обновил контекст для {team}.")
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


def _details_text(data: dict) -> str:
    lines = [
        f"Вероятности: П1 {data['probabilities']['П1']:.1%}, X {data['probabilities']['X']:.1%}, П2 {data['probabilities']['П2']:.1%}",
        f"xG: {data['home_team']} {data['expected_goals'][data['home_team']]}, {data['away_team']} {data['expected_goals'][data['away_team']]}",
        f"Последние 10: {data['home_team']} {data['home_stats']['wins']}-{data['home_stats']['draws']}-{data['home_stats']['losses']}, "
        f"{data['away_team']} {data['away_stats']['wins']}-{data['away_stats']['draws']}-{data['away_stats']['losses']}",
    ]
    if data["warnings"]:
        lines.append("Важно: " + " ".join(data["warnings"]))
    return "\n".join(lines)


def _review_text(review: dict) -> str:
    hit = "да" if review["outcome_hit"] else "нет"
    score_hit = "да" if review["score_hit"] else "нет"
    corner = "" if review["corner_error"] is None else f", ошибка угловых {review['corner_error']:+.2f}"
    return f"Проверено: исход угадан: {hit}, точный счет: {score_hit}{corner}. Модель обновлена."


if __name__ == "__main__":
    raise SystemExit(main())
