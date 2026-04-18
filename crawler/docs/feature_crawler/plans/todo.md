## Crawlee exploration note

Promoted from Max's workspace on 2026-04-08 so crawler research stays visible during the repo split.
This is exploration, not canonical shared scope.
Use [`../PRD_CRAWLER.md`](../PRD_CRAWLER.md) and [`../TASKS.md`](../TASKS.md) for committed product direction.

The best open-source framework for this is Crawlee (@crawlee/playwright). It is purpose-built for this exact scenario: it uses Playwright to render JavaScript-heavy pages (like React/Next.js apps) and includes robust queueing to automatically find and follow same-domain links.

How to Clone a Website Using Crawlee + Playwright
Here is a ready-to-use script that utilizes Playwright to load each page, wait for the dynamic content to render, save the HTML locally, and automatically enqueue new links to recursively clone the site.

1. Install the dependencies:

bash
npm install crawlee playwright
2. Create your cloning script (clone.js):

javascript
import { PlaywrightCrawler } from 'crawlee';
import fs from 'fs/promises';
import path from 'path';

const crawler = new PlaywrightCrawler({
    // Headless browser controlled by Playwright
    headless: true,
    
    async requestHandler({ request, page, enqueueLinks, log }) {
        log.info(`Cloning: ${request.loadedUrl}`);
        
        // Wait for the network to be idle to ensure SPA/React content is fully loaded
        await page.waitForLoadState('networkidle');

        // Extract the fully rendered HTML
        const html = await page.content();
        
        // Parse the URL to create a mirrored local folder structure
        const urlObj = new URL(request.loadedUrl);
        let pathname = urlObj.pathname;
        if (pathname === '/' || pathname === '') pathname = '/index.html';
        if (!path.extname(pathname)) pathname += '.html';
        
        // Define where to save the file
        const filePath = path.join(process.cwd(), 'website-clone', urlObj.hostname, pathname);
        
        // Ensure the directory exists, then save the HTML
        await fs.mkdir(path.dirname(filePath), { recursive: true });
        await fs.writeFile(filePath, html);
        
        // Automatically find all links on the page and add them to the queue
        // 'same-hostname' prevents the crawler from leaving the target website
        await enqueueLinks({
            strategy: 'same-hostname'
        });
    },
    
    // Safety limit to prevent infinite loops on massive sites
    maxRequestsPerCrawl: 200, 
});

// Start the recursive clone
await crawler.run(['https://www.delve.ai']);
Challenges to Keep in Mind
While this script successfully captures the fully rendered DOM of an entire site, true "cloning" of modern web apps comes with a few caveats:

Asset Downloading: The script above saves the HTML perfectly. However, the CSS, JS, and image files referenced inside that HTML will still point to the original server's URLs. If you need everything to work completely offline, you would need to intercept network requests via page.on('response', ...) to save raw CSS/JS files and rewrite the HTML DOM paths to point locally.

Authentication: If delve.ai requires a login to view certain dashboards, you will need to inject cookies or write a brief login sequence in the requestHandler before it starts navigating the queue. Crawlee allows you to preserve session states for this purpose.

Dynamic Loading Walls: Some platforms use aggressive anti-bot protections or require scrolling to trigger lazy-loaded assets. You can easily add await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight)); before capturing the content to trigger those network requests.
