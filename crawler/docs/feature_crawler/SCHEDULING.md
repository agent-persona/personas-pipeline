# Scheduling

## Files

- `feature_crawler/config/crawl_jobs.json`
- `feature_crawler/scripts/cron_runner.py`
- `feature_crawler/scripts/install_cron.sh`

## Current default

One enabled job:

- `reddit-webscraping-public-json`

Example disabled LinkedIn job:

- `linkedin-example-linkdapi`

It runs:

```bash
python3 -m feature_crawler.crawler.cli crawl-reddit \
  --subreddit webscraping \
  --output-dir feature_crawler/data \
  --auth-mode public-json \
  --sort new \
  --limit 25 \
  --comment-limit 128 \
  --cursor-store feature_crawler/data/cursors
```

LinkedIn example job command:

```bash
python3 -m feature_crawler.crawler.cli crawl-linkedin \
  --url https://www.linkedin.com/in/example-person/ \
  --output-dir feature_crawler/data \
  --mode linkdapi \
  --collection-basis consented \
  --scope profile,activity,network \
  --activity-limit 25 \
  --comment-limit 64 \
  --network-limit 50 \
  --max-pages 3 \
  --cursor-store feature_crawler/data/cursors
```

## Dry run

```bash
python3 feature_crawler/scripts/cron_runner.py --dry-run
```

## Install cron

Daily at 6:00 AM:

```bash
bash feature_crawler/scripts/install_cron.sh
```

Custom schedule:

```bash
bash feature_crawler/scripts/install_cron.sh "0 */6 * * *"
```

## Logs

- `feature_crawler/logs/cron_runner.log`
- `feature_crawler/logs/<job-name>.log`

## Notes

- cron entry is installed as one tagged block in user crontab
- edit `crawl_jobs.json` to add more crawler jobs
- current default uses Reddit public `.json` endpoints, so expect occasional `429` responses
