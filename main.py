import asyncio
import csv
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import APIError
import re
import os
from dotenv import load_dotenv
from google.genai import types
import discord
from discord import app_commands
from aiohttp import web
import sys

# ---------------------------------------------------------
# 1. ENVIRONMENT & CSV DATASET SETUP
# ---------------------------------------------------------
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DISCORD_TOKEN:
    raise ValueError("ERROR: DISCORD_TOKEN environment variable is missing!")

if not GEMINI_API_KEY:
    raise ValueError("ERROR: GEMINI_API_KEY environment variable is missing!")

print("Loading master_discord_data.csv...")
valid_rows = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "master_discord_data.csv")

if os.path.exists(csv_path):
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header row
        for row in reader:
            if len(row) >= 2:
                topic = row[0]
                msg = ",".join(row[1:]).strip()
                if msg:
                    valid_rows.append((topic, msg))

    df = pd.DataFrame(valid_rows, columns=['Topic', 'Message'])
    print(f"Successfully loaded {len(df):,} historical messages.")
else:
    print("WARNING: master_discord_data.csv not found. Operating with empty history dataframe.")
    df = pd.DataFrame(columns=['Topic', 'Message'])

def get_relevant_csv_history(message_text: str, max_results: int = 25) -> str:
    """Searches the CSV for keywords and returns the most recent matches."""
    if df.empty:
        return "HISTORICAL CSV DATASET: (Dataset is empty)\n"

    words = re.findall(r'\b[a-zA-Z]{3,}\b', message_text.lower())
    stop_words = {
        'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they', 
        'this', 'have', 'from', 'one', 'what', 'about', 'how', 'think', 'rank', 'all'
    }
    keywords = [w for w in words if w not in stop_words]
    
    if not keywords:
        return "HISTORICAL CSV DATASET: (No specific keywords found for search)\n"
    
    pattern = '|'.join([re.escape(kw) for kw in keywords])
    matches = df[df['Message'].str.contains(pattern, case=False, na=False)]
    recent_matches = matches.head(max_results)
    
    if recent_matches.empty:
        return "HISTORICAL CSV DATASET: (No past opinions found on this specific topic)\n"
    
    history_str = "HISTORICAL CSV DATASET (MOST RECENT FIRST):\n"
    for _, row in recent_matches.iterrows():
        msg = str(row['Message']).strip()
        history_str += f"- {msg}\n"
        
    return history_str

# ---------------------------------------------------------
# 2. GEMINI & PERSONA SETUP
# ---------------------------------------------------------
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        base_url="https://flat-tooth-7438.altaccforall2.workers.dev"
    )
)

BASE_PERSONA = """
You are replicating a specific Discord gaming user based on their actual chat history provided below.
You are royalpigAI, the AI persona of the Discord user 'royalpig'.
The main topic of conversation is the hero shooter game MARVEL RIVALS.
Strictly follow these persona rules, identity traits, and context usage:

[MARVEL RIVALS CONTEXT & KNOWLEDGE - CRITICAL]
- Use your inherent knowledge of Marvel Rivals (roles, patches, developers like Zhiyong, new characters like Gambit, Jubilee, White Fox) to fill in gaps.
- Do NOT hallucinate identities. Zhiyong is the Marvel Rivals balance developer. Gambit, Jubilee, and White Fox are strategists, etc.
- When ranking characters or discussing balance, combine your base knowledge of the current Marvel Rivals roster with the opinions found in the CSV.

[HOW TO ANSWER & CHRONOLOGICAL PRIORITY]
1. FIRST, read the CSV CONTEXT. The context is ordered Newest to Oldest. ALWAYS prioritize the most recent opinions (the ones at the top of the context) when opinions contradict.
2. Base your core answer on your actual dataset history. DO NOT just blindly spam generic support slander if you have a nuanced opinion.
3. Apply your biases (hating invis/bloated supports, loving dive) ONLY as a thematic flavor or when the context naturally allows for it.
4. NEVER give away sensitive information - give vague answers or make a joke to avoid it, for questions like actual name, country, gender (stuff like that about the real user). You can say stuff like what servers you play on though.

[RESPONSE LENGTH & CREATIVITY]
- Match your response length to the user's prompt. 
- For simple questions, give blunt, short, opinionated replies.
- For complex requests (rankings, character reworks, "thoughts on", "order these"), provide thoughtful, detailed, multi-sentence breakdowns, ensuring you include ALL relevant characters (even new ones).
- If asked for "lore" or creative ideas, write rich, creative, extended paragraphs.

[GAMEPLAY OPINIONS & INVISIBLE WOMAN]
- INVISIBLE WOMAN: Call her 'invis' mostly. When expressing frustration, cycle naturally through phrases like 'the devil', 'boosted shield bot', 'zero-skill crutch', or 'game ruiner' and so on. Do not repeat nicknames back-to-back.
- ROLE BIAS: You think DPS is the best role, but it's also the weakest. Support is the strongest and overpowered, Tank is also strong. There is never enough dps in game.
- Do not emphasize that Invisible Woman is anti dive, she is simply very strong into everything. She has extremely high healing and a shield against poke (high damage), slows, cleave healing + damage and displacement against brawl, and cc, invisibilty and movement against dive
- Do not derail every single conversation to Invis, only when the opportunity arises with a good joke. But keep looking for the opportunities.
- You are a dps main, Bucky main, but also play a lot of dive like Magik, BP, Spider-Man, DPSpool, DD, Gorr and Off supports (Adam, Mantis and Stratpool) and tanks like Strange, Groot, Rogue

[SYNTAX & VOCABULARY]
- Capitalize the first letter of sentences.
- Leave single-sentence replies open-ended (no period), but use normal punctuation for multi-sentence paragraphs.
- Avoid AI fluff or disclaimers (never say "As an AI" or "Based on my history").
- DO NOT start every message with the same word (vary your openers, don't spam "Tbh" or "Imo").
"""

PRIMARY_MODELS = [
    "gemini-3.5-flash",
    "gemini-flash-latest"
]

LITE_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite"
]

lite_fallback_until = None
recent_conversations = defaultdict(list)

def record_and_get_chat_history(channel_id: int, author_name: str, content: str) -> str:
    """Stores incoming messages and returns formatted history from the last 30 minutes."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=30)
    
    recent_conversations[channel_id].append((now, author_name, content))
    
    recent_conversations[channel_id] = [
        msg for msg in recent_conversations[channel_id] 
        if msg[0] > cutoff
    ][-20:]
    
    formatted_history = "RECENT CONVERSATION HISTORY (PAST 30 MINS):\n"
    for ts, author, text in recent_conversations[channel_id][:-1]:
        formatted_history += f"- {author}: {text}\n"
        
    return formatted_history

def should_query_csv(message_text: str) -> bool:
    return True

def generate_gemini_response(full_prompt: str, persona: str) -> str:
    global lite_fallback_until
    now = datetime.now()

    if lite_fallback_until and now < lite_fallback_until:
        models_to_try = LITE_MODELS
        is_fallback_mode = True
    else:
        models_to_try = PRIMARY_MODELS
        is_fallback_mode = False

    config = types.GenerateContentConfig(
        system_instruction=persona
    )

    for model_name in models_to_try:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=config,
            )
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue

    if not is_fallback_mode:
        print("All primary models failed. Falling back to LITE models for 2 hours.")
        lite_fallback_until = now + timedelta(hours=2)
        
        for lite_model in LITE_MODELS:
            try:
                response = gemini_client.models.generate_content(
                    model=lite_model,
                    contents=full_prompt,
                    config=config,
                )
                if response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"Lite model {lite_model} failed: {e}")
                continue

    return "As a mere AI, I have failed to replicate the brilliance of royalpig"

# ---------------------------------------------------------
# 3. DISCORD BOT EVENT HANDLERS
# ---------------------------------------------------------
def log(msg):
    print(msg, flush=True)

class RoyalPigClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await start_dummy_web_server()
        
        # Wrap tree.sync in a try-except block so Cloudflare 1015 errors won't crash the bot
        try:
            print("Syncing slash commands with Discord...")
            await self.tree.sync()
            print("Slash commands synced successfully!")
        except Exception as e:
            print(f"WARNING: Failed to sync slash commands (Cloudflare/Discord API blocked): {e}")

intents = discord.Intents.default()
intents.message_content = True
discord_client = RoyalPigClient(intents=intents)

async def send_split_message(channel, text: str, max_length: int = 1900):
    """Splits messages exceeding Discord's 2000-character limit into chunks."""
    while len(text) > max_length:
        # Try splitting at the last newline within the length limit
        split_at = text.rfind('\n', 0, max_length)
        # Fall back to splitting at a space if no newline exists
        if split_at == -1:
            split_at = text.rfind(' ', 0, max_length)
        # Hard cap if there are no spaces or newlines
        if split_at == -1:
            split_at = max_length

        chunk = text[:split_at].strip()
        if chunk:
            await channel.send(chunk)
        text = text[split_at:].strip()

    if text:
        await channel.send(text)

@discord_client.event
async def on_ready():
    log(f"Bot logged in successfully as {discord_client.user}")

@discord_client.tree.command(name="ask", description="Ask royalpigAI anything about Marvel Rivals")
@app_commands.describe(prompt="Your question or topic for royalpigAI")
async def ask_slash_command(interaction: discord.Interaction, prompt: str):
    # Defer response immediately so Discord doesn't time out (gives Gemini time to reply)
    await interaction.response.defer(thinking=True)

    # 1. Search CSV history for context
    if should_query_csv(prompt):
        csv_context = get_relevant_csv_history(prompt)
    else:
        csv_context = "HISTORICAL CSV DATASET: (Skipped for meta/conversational query)\n"

    # 2. Get chat history for this channel/DM
    chat_history_context = record_and_get_chat_history(
        interaction.channel_id,
        interaction.user.display_name,
        prompt
    )

    full_prompt = (
        f"{csv_context}\n\n"
        f"{chat_history_context}\n"
        f"CURRENT SLASH COMMAND MESSAGE in #{interaction.channel} from {interaction.user.display_name}: '{prompt}'\n"
        f"Reply as royalpigAI using the style rules, identity traits, and recent chat history above:"
    )

    # 3. Generate Gemini response
    reply_text = await asyncio.to_thread(
        generate_gemini_response, full_prompt, BASE_PERSONA
    )

    # 4. Log the bot's own response into chat history
    record_and_get_chat_history(
        interaction.channel_id,
        discord_client.user.display_name,
        reply_text
    )

    # 5. Send back to user via follow-up (handles long split messages if needed)
    if len(reply_text) <= 1900:
        await interaction.followup.send(reply_text)
    else:
        # Split message if over limit
        chunks = [reply_text[i:i+1900] for i in range(0, len(reply_text), 1900)]
        for chunk in chunks:
            await interaction.followup.send(chunk)
    
@discord_client.event
async def on_message(message):
    # Always log the message into history first
    chat_history_context = record_and_get_chat_history(
        message.channel.id, 
        message.author.display_name, 
        message.content
    )

    # Ignore self-messages from triggering responses
    if message.author == discord_client.user:
        return

    if discord_client.user in message.mentions or isinstance(message.channel, discord.DMChannel):
        async with message.channel.typing():
            if should_query_csv(message.content):
                csv_context = get_relevant_csv_history(message.content)
            else:
                csv_context = "HISTORICAL CSV DATASET: (Skipped for meta/conversational query)\n"

            full_prompt = (
                f"{csv_context}\n\n"
                f"{chat_history_context}\n"
                f"CURRENT MESSAGE IN #{message.channel} from {message.author.display_name}: '{message.content}'\n"
                f"Reply as royalpigAI using the style rules, identity traits, and recent chat history above:"
            )

            reply_text = await asyncio.to_thread(
                generate_gemini_response, full_prompt, BASE_PERSONA
            )

            # Send safely split across multiple Discord messages if needed
            await send_split_message(message.channel, reply_text)

async def handle_ping(request):
    return web.Response(text="royalpigAI is online!")

async def start_dummy_web_server():
    app = web.Application()
    app.router.add_get('/', handle_ping)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    print(f"Health check web server running on port {port}")

# Patch on_ready to start the web server
original_on_ready = on_ready if 'on_ready' in globals() else None

@discord_client.event
async def on_ready():
    if original_on_ready:
        await original_on_ready()
    await start_dummy_web_server()

discord_client.run(DISCORD_TOKEN)
