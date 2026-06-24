# National Football Predictor

Локальное приложение для прогнозов матчей сборных на чемпионате мира: исход, два точных счета, ожидаемый тотал угловых, последние 10 матчей обеих команд, травмы/мотивация/заметки, тактический профиль команд и проверка результата после матча с самообучением.

Стартовая база внутри проекта демонстрационная: она нужна, чтобы приложение сразу показывало полный цикл на примере `Англия, Гана`. Для боевых прогнозов добавляйте реальные матчи вручную или подключите API-Football через `API_FOOTBALL_KEY`.

По умолчанию включен World Cup mode: каждый матч считается важным для обеих сборных, мотивация высокая, ожидается основной состав. Если есть травмы, дисквалификации или ротация, внесите их вручную.

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
python -m football_predictor.cli context "Англия" --motivation 0.98 --note "Финал, основной состав ожидается"
python -m football_predictor.cli context "Англия" --injury "Player Name:out:0.6"
python -m football_predictor.cli context "Англия" --lineup-strength 0.86
python -m football_predictor.cli context "Англия" --clear-injuries
```

`impact` у травмы задается от `0` до `1`: чем выше число, тем сильнее штраф для ожидаемых голов команды.

## Тактика и схема

Модель учитывает `data/tactical_profiles.json`: формацию, владение, прессинг, высоту линии, силу обороны, ширину атаки, игру через центр, прямолинейность, переходы, стандарты и темп.

Пример обновления:

```powershell
python -m football_predictor.cli tactics "Англия" --formation 4-3-3 --primary-attack "wide overloads and cutbacks" --possession 0.72 --pressing 0.68 --defense 0.74 --width 0.76 --set-pieces 0.70
```

Числовые тактические поля задаются от `0` до `1`. Если у команды нет профиля, используется нейтральный шаблон и выводится предупреждение.

## JSON-режим

```powershell
python -m football_predictor.cli predict "Англия, Гана" --json
```

Подходит для интеграции с ботом, сайтом или таблицей.

## Где что лежит

- `football_predictor/predictor.py` - расчет xG, исхода, счетов и угловых.
- `football_predictor/tactics.py` - тактические матчапы: владение, прессинг, фланги, стандарты, переходы.
- `football_predictor/learning.py` - проверка результата и онлайн-обучение.
- `football_predictor/providers.py` - опциональная интеграция API-Football.
- `data/matches.json` - история матчей.
- `data/team_context.json` - травмы, мотивация, заметки.
- `data/match_context.json` - World Cup mode, важность матча и базовая сила состава.
- `data/tactical_profiles.json` - схема и стиль игры сборных.
- `web/` - простой локальный интерфейс.

## Ограничения

Это не магический шар. Самое важное для качества прогноза - свежие и честные входные данные: реальные последние матчи, угловые, составы, травмы, турнирный контекст, место матча и мотивация. Приложение специально показывает предупреждения, когда данных мало или они демонстрационные.
