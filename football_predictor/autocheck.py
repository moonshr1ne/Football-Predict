from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from .data_store import DataStore
from .learning import OnlineLearner
from .providers import EspnWorldCupProvider, ProviderError


class AutoChecker:
    def __init__(self, store: DataStore, provider=None):
        self.store = store
        self.provider = provider or EspnWorldCupProvider()

    def check_pending(self, limit: int = 25, today: str | None = None) -> dict[str, Any]:
        today = today or date.today().isoformat()
        predictions = self.store.load_predictions()
        checked = learned = errors = 0
        results: list[dict[str, Any]] = []

        for prediction in predictions:
            if checked >= limit:
                break
            if prediction.get("status") != "pending":
                continue
            match_date = prediction.get("match_date")
            if not match_date or match_date > today:
                continue

            checked += 1
            prediction["last_checked_at"] = datetime.now(timezone.utc).isoformat()
            try:
                result = self.provider.get_finished_result(
                    prediction["home_team"],
                    prediction["away_team"],
                    match_date,
                )
                review = OnlineLearner(self.store).record_result(
                    home_team=prediction["home_team"],
                    away_team=prediction["away_team"],
                    date=result["date"],
                    home_goals=int(result["home_goals"]),
                    away_goals=int(result["away_goals"]),
                    home_corners=result.get("home_corners"),
                    away_corners=result.get("away_corners"),
                    neutral=bool(prediction.get("neutral", True)),
                    source=result.get("source", "api"),
                    baseline_prediction=prediction,
                )
                prediction["status"] = "reviewed"
                prediction["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                prediction["review"] = review
                prediction.pop("check_error", None)
                learned += 1
                results.append({"prediction_id": prediction.get("prediction_id"), "status": "reviewed", "review": review})
            except ProviderError as exc:
                prediction["check_error"] = str(exc)
                results.append({"prediction_id": prediction.get("prediction_id"), "status": "pending", "error": str(exc)})
            except Exception as exc:
                errors += 1
                prediction["check_error"] = str(exc)
                results.append({"prediction_id": prediction.get("prediction_id"), "status": "error", "error": str(exc)})

        self.store.save_predictions(predictions)
        pending = sum(1 for item in predictions if item.get("status") == "pending")
        return {
            "checked": checked,
            "learned": learned,
            "pending": pending,
            "errors": errors,
            "results": results,
        }
