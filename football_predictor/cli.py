from __future__ import annotations

import argparse
import json
import sys

from .barcelona_service import BarcelonaService
from .server import run_server


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="barca-lab", description="Прогноз матчей Барселоны в Ла Лиге и ЛЧ.")
    commands = parser.add_subparsers(dest="command")

    predict = commands.add_parser("predict", help="Построить прогноз по названию соперника.")
    predict.add_argument("opponent", help='Например: "Реал Мадрид"')
    predict.add_argument("--preview", action="store_true", help="Не сохранять неизменяемый предматчевый снимок.")
    predict.add_argument("--json", action="store_true")

    sync = commands.add_parser("sync", help="Обновить Ла Лигу, ЛЧ и протоколы последних матчей.")
    sync.add_argument("--force", action="store_true")
    sync.add_argument("--json", action="store_true")

    backtest = commands.add_parser("backtest", help="Запустить честный walk-forward бэктест.")
    backtest.add_argument("--force", action="store_true")
    backtest.add_argument("--json", action="store_true")

    commands.add_parser("auto-check", help="Найти результаты сохраненных прогнозов и обновить модель.")
    commands.add_parser("status", help="Показать состояние базы и модели.")

    serve = commands.add_parser("serve", help="Запустить локальный веб-интерфейс.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "serve":
        run_server(args.host, args.port)
        return 0

    service = BarcelonaService()
    if args.command == "predict":
        payload = service.predict(args.opponent, remember=not args.preview)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            if payload.get("prediction_available") is False:
                print(payload.get("message"))
                print(f"Фактический счет: {payload.get('result_summary', {}).get('actual') or 'нет'}")
            else:
                print(f"{payload['competition']}, {payload['stage']}, {payload['match_date']}")
                print(f"Исход: {payload['outcome']['label']} ({payload['outcome']['confidence'] * 100:.1f}%)")
                print(f"Точный счет Барселона — соперник: {payload['exact_score']['score']} ({payload['exact_score']['probability'] * 100:.1f}%)")
                print(f"Голы: {payload['goals']['total_expected']:.2f}; угловые: {payload['corners']['point']}; фолы: {payload['fouls']['point']}")
        return 0
    if args.command == "sync":
        payload = service.syncer.sync(force=args.force)
    elif args.command == "backtest":
        payload = service.model.train_and_backtest(force=args.force)
    elif args.command == "auto-check":
        payload = service.auto_check()
    else:
        payload = service.status()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
