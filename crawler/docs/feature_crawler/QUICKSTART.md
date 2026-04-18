# Quickstart

Reddit:
Run `python3 -m feature_crawler.crawler.cli crawl-reddit --subreddit webscraping --output-dir feature_crawler/data --auth-mode public-json --sort new --limit 25 --comment-limit 128 --cursor-store feature_crawler/data/cursors`.
Use `--auth-mode oauth` instead when `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` exist.
Output lands in `feature_crawler/data/reddit/<subreddit>/<date>/` and the resume cursor lands in `feature_crawler/data/cursors/`.

Discord:
Run `python3 -m feature_crawler.crawler.cli crawl-discord --guild-id <guild-id> --channel-id <channel-id> --output-dir feature_crawler/data --collection-basis consented`.
Bot mode uses `DISCORD_BOT_TOKEN`; user mode is `crawl-discord-user` with `DISCORD_USER_TOKEN`.
Output lands in `feature_crawler/data/discord/<server>/<channel>_<id>/<month>/<day>/<year>/`.

LinkedIn:
Run `python3 -m feature_crawler.crawler.cli crawl-linkedin --url https://www.linkedin.com/in/example-person/ --output-dir feature_crawler/data --mode public-html --collection-basis consented`.
Session-backed mode uses `--mode session-html` and expects `LINKEDIN_COOKIE` or `LINKEDIN_SESSION_COOKIE_LI_AT` plus `LINKEDIN_SESSION_COOKIE_JSESSIONID`.
Browser-backed network mode uses `--mode session-browser --scope profile,network` and crawls the authenticated viewer's My Network connections.
Official mode uses `--mode official-oidc` and expects `LINKEDIN_ACCESS_TOKEN` unless overridden with `--access-token-env`.
Auth bootstrap URL: `python3 -m feature_crawler.crawler.cli linkedin-auth-url --client-id "$LINKEDIN_CLIENT_ID" --redirect-uri http://127.0.0.1:8080/callback`.
Code exchange: `python3 -m feature_crawler.crawler.cli linkedin-exchange-code --code "$LINKEDIN_AUTH_CODE" --redirect-uri http://127.0.0.1:8080/callback`.
Vendor modes: `--mode apify`, `--mode brightdata`, or `--mode linkdapi`. Use `--scope profile,activity,network` plus `--activity-limit`, `--comment-limit`, `--network-limit`, `--max-pages`, and `--cursor-store` for paginated crawls.
Output lands in `feature_crawler/data/linkedin/<profile-id>/<date>/`.
