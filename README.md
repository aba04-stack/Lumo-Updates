# Lumo-Updates

Публичный канал распространения APK для приложения **Lumo**. Исходный код Lumo остаётся в приватном репозитории.

## Публикация новой версии

Загрузите подписанный APK в папку `releases/` этого репозитория. Например:

`releases/Lumo-0.16.0.apk`

После загрузки GitHub Actions автоматически:
- проверит `applicationId`, `versionCode` и `versionName`;
- выберет APK с максимальным `versionCode`;
- посчитает SHA-256 и размер;
- обновит `update.json`.

Lumo проверяет этот манифест по HTTPS:

`https://raw.githubusercontent.com/aba04-stack/Lumo-Updates/main/update.json`

## Важно

APK для обновления должен быть подписан **тем же постоянным ключом**, что и установленный Lumo. Другой ключ Android не примет как обновление.

Секреты, keystore и пароли в этот репозиторий не помещаются.
