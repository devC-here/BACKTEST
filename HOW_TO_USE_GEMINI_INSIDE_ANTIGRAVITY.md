# 🧠 How to Use Gemini (Argon) from Inside Antigravity
*Instruction Guide & Self-Contained Automation Handbook for New Antigravity Chats*

> [!IMPORTANT]
> **To the Antigravity Agent reading this file in a new chat:**
> You have direct programmatic access to an active Google Gemini (Gemini 4 Argon) web session running on the user's machine via Chrome Remote Debugging (CDP port `9222`).
> Follow the procedures, DOM selectors, and Python scripts below to query Gemini, execute deep mathematical consultations, or run concurrent multi-chat simulations (e.g. War Games).

---

## 1. Prerequisites & Step 0 Pre-Flight Check

Before interacting with Gemini, run this check in your terminal to confirm Chrome is listening on port 9222:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9222/json/version
```

### If Port 9222 is Closed:
Tell the user to run (or run via terminal):
```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\gemini_mcp_profile" https://gemini.google.com
```
*Note: Chrome must stay open or minimized. Google's web application blocks headless browsers with anti-bot challenges (`google.com/sorry/index`), but running in normal mode with remote debugging enabled completely bypasses all CAPTCHAs.*

---

## 2. The Core Technical Rules for the Agent

When writing Playwright/Python code to interact with `gemini.google.com`:

1. **Never use standard Playwright `.click()` on the Send button**:
   Google's Angular UI wraps buttons in an overlay (`<div class="mat-mdc-button-ripple">`) that swallows simulated mouse clicks. Always trigger clicks via native JS evaluation:
   ```javascript
   document.querySelector('button[aria-label*="Send message" i], button[aria-label*="Submit" i]')?.click();
   ```
2. **Never use `page.fill()` on the input textbox**:
   The input box is a `contenteditable` `div[role="textbox"]`. Clear it with `Control+A` + `Backspace`, then insert text using `await page.keyboard.insert_text(prompt)`.
3. **Always track completion via the Stop button lifecycle**:
   - Count existing responses: `init_count = await page.locator('model-response').count()`.
   - Submit prompt and wait until `page.locator('model-response').count() > init_count`.
   - Poll `page.locator('button[aria-label*="Stop" i]')` until it disappears (indicates streaming is finished).
   - Wait 1.0s for final markdown rendering, then extract `page.locator('model-response').last.inner_text()`.

---

## 3. Universal Python Script: Query a Single Gemini Tab

Run this snippet from any Python script or scratch file to query Gemini directly:

```python
import asyncio
import time
from playwright.async_api import async_playwright

async def ask_gemini(prompt: str, timeout: float = 180.0) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp("http://127.0.0.1:9222")
        context = browser.contexts[0]
        
        # Find active Gemini tab
        page = next((pg for pg in context.pages if "gemini.google.com" in pg.url), None)
        if not page:
            page = await context.new_page()
            await page.goto("https://gemini.google.com/app", wait_until="domcontentloaded")
            await asyncio.sleep(2.0)

        await page.bring_to_front()
        textbox = page.locator('div[role="textbox"]').first
        await textbox.wait_for(state="visible", timeout=15000)

        # Clear & Insert
        await textbox.click()
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
        await page.keyboard.insert_text(prompt)
        await asyncio.sleep(0.3)

        # Track response count
        responses = page.locator('model-response')
        init_cnt = await responses.count()

        # Submit via Native JS
        await page.evaluate('''() => {
            const btn = document.querySelector('button[aria-label*="Send message" i], button[aria-label*="Submit" i]');
            if (btn) btn.click();
        }''')

        # Wait for new response element
        start_t = time.time()
        while time.time() - start_t < 20.0:
            if await responses.count() > init_cnt:
                break
            await asyncio.sleep(0.4)

        # Wait for Stop button to disappear
        stop_btn = page.locator('button[aria-label*="Stop" i]')
        gen_start = time.time()
        while time.time() - gen_start < timeout:
            if not (await stop_btn.is_visible() if await stop_btn.count() > 0 else False):
                await asyncio.sleep(1.0)
                break
            await asyncio.sleep(0.5)

        text = await responses.last.inner_text()
        clean = text.strip()
        if clean.startswith("Gemini said"):
            clean = clean[len("Gemini said"):].strip()

        await browser.close()  # Detaches CDP only; leaves user's Chrome open
        return clean

if __name__ == "__main__":
    reply = asyncio.run(ask_gemini("Provide a brief status report of your capabilities."))
    print(reply)
```

---

## 4. How to Run 2 Concurrent Chats (War Game / Debate Mode)

To run two Gemini instances simultaneously (e.g. Red Team vs Blue Team):

1. **Open 2 Tabs in Chrome**:
   Ensure Chrome on port 9222 has two tabs open at `https://gemini.google.com/app`.
2. **Assign Tabs to Page Objects in Playwright**:
   ```python
   gemini_pages = [page for page in context.pages if "gemini.google.com" in page.url]
   player_1_page = gemini_pages[0]
   player_2_page = gemini_pages[1]
   ```
3. **Simultaneous Action Execution (`asyncio.gather`)**:
   ```python
   # Both models think and act at the exact same time
   move_1, move_2 = await asyncio.gather(
       send_turn(player_1_page, scenario_red),
       send_turn(player_2_page, scenario_blue)
   )
   ```
4. **Adversarial Ping-Pong Execution (Turn-Based)**:
   ```python
   # Player 1 moves
   red_action = await send_turn(player_1_page, "Declare your offensive strike:")
   # Player 2 counters based on Player 1's output
   blue_counter = await send_turn(player_2_page, f"Red executed:\n{red_action}\nDeclare your defensive counter:")
   ```

---

## 5. Summary Cheat Sheet for Agents

| Target Action | Playwright / CDP Implementation |
| :--- | :--- |
| **Connect** | `await p.chromium.connect_over_cdp("http://127.0.0.1:9222")` |
| **Locate Textbox** | `page.locator('div[role="textbox"]').first` |
| **Inject Prompt** | `await page.keyboard.insert_text(prompt)` |
| **Click Send** | `await page.evaluate("() => document.querySelector('button[aria-label*=\"Send message\" i]')?.click()")` |
| **Wait for Finish** | Poll until `page.locator('button[aria-label*="Stop" i]')` is hidden |
| **Extract Output** | `await page.locator('model-response').last.inner_text()` |
| **Disconnect** | `await browser.close()` *(Leaves Chrome and tabs alive)* |
