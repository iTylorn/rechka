# Разработка и сборка WhisperGUI для Windows

Команды ниже выполняются в PowerShell из `WhisperGUI_Development\app`. Нужны Windows x64, Python 3.10, совместимый Intel GPU и драйвер, видимый OpenVINO. Релизный архив включает Python-библиотеки и FFmpeg; для запуска из исходников зависимости ставятся отдельно. Перед сборкой убедитесь, что рядом с `app/` находятся `diarization/wespeaker_fixed.xml` и `.bin`: `WhisperGUI.spec` берёт их оттуда.

## Подготовка и запуск

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe main.py
```

`start_whisper_gpu.bat` — альтернативный запуск, но он использует строго `%LOCALAPPDATA%\Programs\Python\Python310\python.exe`, а не `.venv`. `requirements-build.txt` включает зависимости приложения из `requirements.txt`, PyInstaller 6.16.0 и `openvino-tokenizers` 2026.3.0.0, которые нужны для сборки.

## Получение моделей

При первом запуске окно предложит установить Whisper large-v3-turbo INT8 (~0,83 ГБ). Кнопка «Модели и обновления...» открывает менеджер пяти вариантов моделей, включая Whisper FP16, WeSpeaker и Qwen3-8B/14B INT4. Скачивание требует доступа к Hugging Face и сохраняет новые версии по умолчанию в `%LOCALAPPDATA%\WhisperGUI-Beta\models\_versions`; после установки модель используется локально. При прерывании загрузки повторный запуск продолжает файлы из `_downloads`. Подробности проверки ревизий и отката — в [MODELS.md](MODELS.md).

При ручной раскладке поместите папки моделей в `WhisperGUI_Development\turbo-gpu\whisper-large-v3-turbo-int8-ov` и `WhisperGUI_Development\llm-models\Qwen3-8B-int4-ov` (либо `Qwen3-14B-int4-ov`). Это должны быть полные экспортированные модели OpenVINO, а не одни XML-файлы. В настройках `models_folder` указывает на родительскую папку старых моделей Whisper, `llm_model_path` — на конкретную папку Qwen, `embed_model_path` — на XML WeSpeaker. Менеджер хранит точные новые пути в `model_paths`, а прежние — в `model_previous_paths`, не меняя папки старых моделей.

## Проверки и сборка

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe main.py --smoke-test
.\.venv\Scripts\python.exe main.py --gpu-smoke-test
$qwenModelPath = 'C:\Models\Qwen3-8B-int4-ov' # замените на фактический путь
.\.venv\Scripts\python.exe main.py --llm-smoke-test $qwenModelPath
.\.venv\Scripts\python.exe main.py --model-update-smoke-test ..\.test-data\update-smoke
.\build_release.ps1 -PythonExe .\.venv\Scripts\python.exe
```

`unittest` проверяет контрольные суммы, докачку, отмену, блокировки, неизменность активной модели при ошибках, откат, изоляцию настроек и жизненный цикл фоновых задач. Эти тесты не скачивают веса. `--smoke-test` создаёт окно без интерактивного цикла. `--gpu-smoke-test` проверяет FFmpeg, видимость GPU, компиляцию WeSpeaker и загрузку Whisper, **если** модель найдена. `--llm-smoke-test` загружает указанную Qwen и её токенизатор. `--model-update-smoke-test` действительно скачивает WeSpeaker (~25 МБ) в указанную тестовую папку, конвертирует её и проверяет GPU, но не переключает настройки пользователя; результат записывается в `smoke_result.json`. Ни один из этих коротких тестов не проверяет качество распознавания на записи.

`build_release.ps1` сначала запускает `unittest` и останавливается при ошибке, затем запускает PyInstaller по `WhisperGUI.spec` и оба базовых smoke-теста упакованного EXE. Дополнительный тест Qwen выполняется только при наличии `..\llm-models\Qwen3-8B-int4-ov\openvino_model.xml`. При успехе скрипт создаёт `app\dist\WhisperGUI\WhisperGUI.exe`, полный `app\dist\WhisperGUI-2.0.0-beta.1-win64.zip` (версия берётся из `app_version.py`) и печатает SHA-256 архива. Для передачи нужен весь ZIP: EXE зависит от соседней папки `_internal`.

Промежуточные файлы PyInstaller создаются во временной папке Windows и удаляются после успешной сборки. Если запустить PyInstaller вручную и увидеть `build\WhisperGUI\WhisperGUI.exe`, не используйте его: он не содержит соседнюю `_internal` и не является готовым релизом.

## Диагностика

- Не запускается GUI: проверьте `%LOCALAPPDATA%\WhisperGUI-Beta\startup_error.log`, версию Python 3.10 и установку зависимостей в том же окружении, которым запускается приложение.
- «OpenVINO не видит GPU» или не загружается модель: проверьте драйвер Intel GPU и запуск `--gpu-smoke-test`; CPU-режим для нейросетевого вывода здесь не предусмотрен.
- Не найдена WeSpeaker: проверьте оба файла `..\diarization\wespeaker_fixed.xml` и `.bin` и путь `embed_model_path` в `%APPDATA%\WhisperGUI-Beta\settings.json` или установите её через менеджер моделей.
- Не извлекается звук из видео: `MediaLoader` ищет `app\tools\ffmpeg.exe`, затем бинарник из `imageio-ffmpeg`, затем `ffmpeg` в `PATH`; убедитесь, что видео содержит звуковую дорожку.
- Модель скачана, но не видна: проверьте `model_paths`, `last_model` и `llm_model_path`, затем обновите список моделей в GUI. История `_versions` не сканируется как обычная папка: активируется только успешно проверенная версия. Установщик хранит неполную загрузку в `%LOCALAPPDATA%\WhisperGUI-Beta\models\_downloads` или в выбранном каталоге.

## Что ещё не подтверждено этими проверками

Исходный пакет не содержит большие веса Whisper/Qwen. Короткие проверки не заменяют сквозной прогон реальной записи и оценку качества на целевом Windows ПК. Совместимость одного компьютера не гарантирует поддержку других GPU и драйверов. После изменения версий зависимостей или модели выполните ручную проверку на целевом оборудовании.

## Проверка релиза 2.0.0-beta.1, 13 октября 2026

На Windows x64 с Intel Arc B390 и OpenVINO 2026.3 выполнены:

- 68 автоматических тестов: все прошли, включая отмену, откат, ошибки записи, сохранность активных версий и отдельный профиль беты.
- Настоящая загрузка WeSpeaker с Hugging Face, конвертация и контрольный вывод GPU: из исходников и из готового EXE.
- Загрузка готовым EXE Whisper FP16, отдельно INT8 с новой WeSpeaker, а также Qwen3-8B и Qwen3-14B на GPU: успешна.
- Проверка метаданных и хешей всех пяти существующих моделей: совпадают с актуальными файлами официальных репозиториев на дату проверки. Это не обещание улучшения качества при будущих обновлениях.
- Запуск EXE из распакованного ZIP во временной папке с чистыми APPDATA/LOCALAPPDATA и только системными каталогами в PATH: успешен. В каталоге присутствует `_internal/python310.dll`, версия EXE — `2.0.0-beta.1`.
- Визуальная проверка менеджера моделей в светлой и тёмной палитрах: все строки и кнопки доступны.

SHA-256 готового `WhisperGUI-2.0.0-beta.1-win64.zip`: `17C1674C114E220A75C5A8122C768353E986AA5E3CB365F013A05AE4BEF2736E`.

Повторная полная загрузка многогигабайтных Whisper/Qwen и испытание на другом компьютере не проводились. Большие модели проверены по уже имеющимся весам; сетевой сквозной сценарий проверен на компактной WeSpeaker. Качество распознавания и протоколов на длинной записи этим релизным прогоном заново не оценивалось.
