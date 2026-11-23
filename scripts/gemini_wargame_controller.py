#!/usr/bin/env python3
"""
================================================================================
  GEMINI 4 ARGON DUAL-AGENT WAR GAME CONTROLLER (scripts/gemini_wargame_controller.py)
================================================================================
Orchestrates two independent Google Gemini (Argon) chat sessions simultaneously
over Chrome Remote Debugging Protocol (CDP port 9222).

Supports:
  - Turn-based adversarial simulations (Red Team vs Blue Team)
  - Concurrent simultaneous round generation via asyncio.gather()
  - Automatic session discovery, DOM stabilization, and transcript generation
================================================================================
"""

import asyncio
import time
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any
from playwright.async_api import async_playwright, Page, Browser

ROOT_DIR = Path(__file__).resolve().parent.parent


class GeminiSession:
    """Wrapper around a single Gemini web chat tab."""
    def __init__(self, name: str, page: Page, role_prompt: str):
        self.name = name
        self.page = page
        self.role_prompt = role_prompt
        self.history: List[Dict[str, str]] = []

    async def initialize(self):
        """Initializes the chat tab with the agent's core identity."""
        print(f"[{self.name}] Initializing role prompt...")
        init_message = (
            f"SYSTEM ROLE INSTRUCTION:\n{self.role_prompt}\n\n"
            f"Acknowledge with 'READY' and your combat callsign."
        )
        reply = await self.send_message(init_message)
        print(f"[{self.name}] System Ready: {reply[:80]}...")

    async def send_message(self, message: str, timeout_seconds: float = 180.0) -> str:
        """Sends a message to this specific Gemini tab and waits for the full response."""
        page = self.page
        await page.bring_to_front()

        # 1. Locate Textbox
        textbox = page.locator('div[role="textbox"]').first
        await textbox.wait_for(state="visible", timeout=15000)
        await textbox.click()
        await asyncio.sleep(0.2)

        # 2. Clear old text and insert new prompt cleanly
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
        await asyncio.sleep(0.1)
        await page.keyboard.insert_text(message)
        await asyncio.sleep(0.4)

        # 3. Record response count before submission
        responses = page.locator('model-response')
        initial_count = await responses.count()

        # 4. Trigger Send via native JS evaluation (bypasses Angular ripple overlays)
        clicked = await page.evaluate('''() => {
            const btn = document.querySelector('button[aria-label*="Send message" i], button[aria-label*="Submit" i]');
            if (btn) {
                btn.click();
                return true;
            }
            return false;
        }''')
        if not clicked:
            await page.keyboard.press("Enter")

        # 5. Wait for new response element to appear
        start_wait = time.time()
        while time.time() - start_wait < 20.0:
            if await responses.count() > initial_count:
                break
            await asyncio.sleep(0.4)
        else:
            raise TimeoutError(f"[{self.name}] No new response appeared within 20 seconds.")

        # 6. Wait for generation to finish (Stop button disappears)
        stop_btn = page.locator('button[aria-label*="Stop" i]')
        gen_start = time.time()
        while time.time() - gen_start < timeout_seconds:
            is_generating = await stop_btn.is_visible() if await stop_btn.count() > 0 else False
            if not is_generating:
                # 1.0s buffer for final markdown DOM stabilization
                await asyncio.sleep(1.0)
                break
            await asyncio.sleep(0.5)

        # 7. Extract cleaned response text
        last_resp = responses.last
        raw_text = await last_resp.inner_text()
        clean_text = raw_text.strip()
        if clean_text.startswith("Gemini said"):
            clean_text = clean_text[len("Gemini said"):].strip()

        self.history.append({"prompt": message, "response": clean_text})
        return clean_text


async def setup_dual_agents(cdp_url: str = "http://127.0.0.1:9222") -> Tuple[Browser, GeminiSession, GeminiSession]:
    """Connects to Chrome and attaches to or creates 2 separate Gemini tabs."""
    p = await async_playwright().start()
    browser = await p.chromium.connect_over_cdp(cdp_url)
    context = browser.contexts[0] if browser.contexts else await browser.new_context()

    # Find existing gemini pages
    gemini_pages = [page for page in context.pages if "gemini.google.com" in page.url]

    # Open additional tab if only 1 exists
    while len(gemini_pages) < 2:
        print(f"[Setup] Opening additional Gemini tab (currently have {len(gemini_pages)})...")
        new_p = await context.new_page()
        await new_p.goto("https://gemini.google.com/app", wait_until="domcontentloaded")
        await asyncio.sleep(2.0)
        gemini_pages.append(new_p)

    tab_a = gemini_pages[0]
    tab_b = gemini_pages[1]

    agent_a = GeminiSession(
        name="Red_Commander",
        page=tab_a,
        role_prompt=(
            "You are RED COMMANDER. You control an aggressive, high-volatility quantitative trading hedge fund. "
            "Your objective is to initiate aggressive momentum attacks, trigger stop-loss cascades on institutional liquidity, "
            "and exploit market inefficiencies. In every turn, declare your specific market action, target price, and volume."
        )
    )

    agent_b = GeminiSession(
        name="Blue_Commander",
        page=tab_b,
        role_prompt=(
            "You are BLUE COMMANDER. You control a conservative, institutional market-making desk. "
            "Your objective is to defend key support/resistance levels, absorb liquidity shocks, counteract market manipulation, "
            "and punish overextended breakout traders. In every turn, declare your defensive inventory, bid/ask walls, and counter-moves."
        )
    )

    return browser, agent_a, agent_b


async def run_wargame_rounds(rounds: int = 3):
    """Executes a multi-round adversarial simulation between the two Gemini agents."""
    print("================================================================================")
    print(f"  STARTING GEMINI 4 ARGON DUAL-AGENT WAR GAME ({rounds} ROUNDS)")
    print("================================================================================")

    browser, red, blue = await setup_dual_agents()

    try:
        # 1. Initialize both agents concurrently
        print("[Setup] Initializing both agents concurrently...")
        await asyncio.gather(red.initialize(), blue.initialize())

        # 2. Starting Scenario
        scenario = (
            "MARKET STATE [EURUSD @ 1.1020]:\n"
            "- Major London Open volatility surge.\n"
            "- Liquidity pool resting above 1.1050 (+30 pips).\n"
            "- Macro interest rate announcement in 30 minutes.\n"
            "State your Round 1 tactical move clearly."
        )

        transcript = [
            "# ⚔️ Gemini War Game Match Transcript",
            f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Format: {rounds}-Round Adversarial Debate (Red Commander vs Blue Commander)*\n",
            "---\n"
        ]

        current_red_move = ""
        current_blue_move = ""

        for r in range(1, rounds + 1):
            print(f"\n>>> ==================== ROUND {r} COMMENCING ==================== <<<")

            if r == 1:
                # Round 1: Concurrent opening gambit
                print(f"[Round {r}] Submitting initial scenario to both agents simultaneously...")
                red_prompt = f"{scenario}\n\nDeclare your opening offensive maneuver:"
                blue_prompt = f"{scenario}\n\nDeclare your opening defensive positioning:"

                current_red_move, current_blue_move = await asyncio.gather(
                    red.send_message(red_prompt),
                    blue.send_message(blue_prompt)
                )
            else:
                # Subsequent Rounds: Turn-based counter-maneuvers
                print(f"[Round {r}] Red Commander countering Blue's last move...")
                red_counter_prompt = (
                    f"ROUND {r} UPDATE:\n"
                    f"Blue Commander responded with:\n'''\n{current_blue_move}\n'''\n\n"
                    f"How do you counter or escalate? State your specific offensive trade and target."
                )
                current_red_move = await red.send_message(red_counter_prompt)

                print(f"[Round {r}] Blue Commander countering Red's escalation...")
                blue_counter_prompt = (
                    f"ROUND {r} UPDATE:\n"
                    f"Red Commander escalated with:\n'''\n{current_red_move}\n'''\n\n"
                    f"How do you defend and absorb this attack? State your defensive order flow."
                )
                current_blue_move = await blue.send_message(blue_counter_prompt)

            print(f"\n[Round {r} Red Move]:\n{current_red_move[:160]}...\n")
            print(f"[Round {r} Blue Counter]:\n{current_blue_move[:160]}...\n")

            transcript.append(f"## 🥊 Round {r}\n")
            transcript.append(f"### 🔴 Red Commander Move\n{current_red_move}\n")
            transcript.append(f"### 🔵 Blue Commander Counter\n{current_blue_move}\n")
            transcript.append("---\n")

        # Save Match Log
        log_path = ROOT_DIR / "results" / "gemini_wargame_match_transcript.md"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(transcript), encoding="utf-8")
        print(f"[SUCCESS] War game complete! Match transcript saved to: {log_path}")

    finally:
        await browser.close()


def main():
    parser = argparse.ArgumentParser(description="Gemini Dual-Agent War Game Controller")
    parser.add_argument("--rounds", type=int, default=3, help="Number of adversarial rounds to run")
    args = parser.parse_args()

    asyncio.run(run_wargame_rounds(rounds=args.rounds))


if __name__ == "__main__":
    main()
