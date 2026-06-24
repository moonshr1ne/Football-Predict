# National Football Predictor

Локальное приложение для прогнозов матчей сборных: исход, два точных счета, ожидаемый тотал угловых, последние 10 матчей обеих команд, травмы/мотивация/заметки и проверка результата после матча с самообучением.

Стартовая база внутри проекта демонстрационная: она нужна, чтобы приложение сразу показывало полный цикл на примере `Англия, Гана`. Для боевых прогнозов добавляйте реальные матчи вручную или подключите API-Football через `API_FOOTBALL_KEY`.

## Быстрый запуск

```powershell
cd outputs\football-national-predictor
python -m football_predictor.cli predict "Англия, Гана" --date 2026-06-24
```

Пример короткого вывода:

```text
П1, средние угловые: 9.10, точные счеты: 2-0, 1-0
```

## Веб-интерфейс

```powershell
python -m football_predictor.cli serve --port 8765
```

Откройте `http://127.0.0.1:8765`.

## Проверка после матча и самообучение

Когда матч завершился, внесите результат:

```powershell
python -m football_predictor.cli result "Англия, Гана" --date 2026-06-24 --score 2-1 --corners 10
```

Что происходит:

- матч попадает в `data/matches.json`;
- приложение сравнивает прогноз с фактом;
- обновляет веса в `data/model_state.json`;
- следующие прогнозы уже используют новый матч и новую калибровку.

Если есть ключ API-Football:

```powershell
$env:API_FOOTBALL_KEY="ваш_ключ"
python -m football_predictor.cli check "Англия, Гана" --date 2026-06-24
```

## Автопроверка своих прогнозов

Если прогноз сделан с `--date`, он сохраняется в `data/predictions.json` со статусом `pending`.
После матча приложение может само пройтись по очереди, забрать финальные результаты через API-Football и обучиться:

```powershell
$env:API_FOOTBALL_KEY="ваш_ключ"
python -m football_predictor.cli auto-check
```

Для постоянной проверки:

```powershell
$env:API_FOOTBALL_KEY="ваш_ключ"
python -m football_predictor.cli watch --interval 3600
```

`watch` надо держать запущенным или повесить на планировщик Windows. Без ключа API приложение все равно обучается, но результат нужно внести вручную через `result`.

## Травмы, мотивация и заметки

```powershell
python -m football_predictor.cli context "Англия" --motivation 0.72 --note "Финал, основной состав ожидается"
python -m football_predictor.cli context "Англия" --injury "Player Name:out:0.6"
python -m football_predictor.cli context "Англия" --clear-injuries
```

`impact` у травмы задается от `0` до `1`: чем выше число, тем сильнее штраф для ожидаемых голов команды.

## JSON-режим

```powershell
python -m football_predictor.cli predict "Англия, Гана" --json
```

Подходит для интеграции с ботом, сайтом или таблицей.

## Где что лежит

- `football_predictor/predictor.py` - расчет xG, исхода, счетов и угловых.
- `football_predictor/learning.py` - проверка результата и онлайн-обучение.
- `football_predictor/providers.py` - опциональная интеграция API-Football.
- `data/matches.json` - история матчей.
- `data/team_context.json` - травмы, мотивация, заметки.
- `web/` - простой локальный интерфейс.

## Ограничения

Это не магический шар. Самое важное для качества прогноза - свежие и честные входные данные: реальные последние матчи, угловые, составы, травмы, турнирный контекст, место матча и мотивация. Приложение специально показывает предупреждения, когда данных мало или они демонстрационные.
