import discord
from discord.ext import commands
import os
import requests
from flask import Flask
from threading import Thread

# 1. Initialize Flask Web Server
app = Flask('')

@app.route('/')
def home():
    return "Bot is alive and healthy!"

def run_web_server():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# 2. Setup Discord Bot Configs
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

HF_TOKEN = os.environ.get("HF_TOKEN")
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")

# The direct HTTP API route to your specific space endpoint
API_URL = "https://royalpig7-royalpig-free-bot.hf.space/api/predict"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

chat_histories = {}

@bot.event
async def on_ready():
    print(f"✅ Royalpig AI Bot is live on Render Free Tier as {bot.user}!")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        async with message.channel.typing():
            channel_id = message.channel.id
            clean_input = message.clean_content.replace(f"@{bot.user.name}", "").strip()
            
            if channel_id not in chat_histories:
                chat_histories[channel_id] = []
                
            chat_histories[channel_id].append({"role": "user", "content": clean_input})

            history_context = ""
            for turn in chat_histories[channel_id][-5:]:
                prefix = "User: " if turn["role"] == "user" else "Assistant: "
                history_context += f"{prefix}{turn['content']}\n"

            try:
                # 🌟 Raw HTTP Payload matching Gradio's expected json structure
                payload = {"data": [history_context]}
                
                # Make a clean, standard POST request that won't break on Python 3.14
                response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=30)
                
                if response.status_code == 200:
                    # Extract the text from Gradio's standard output array format
                    ai_response = response.json()["data"][0]
                    
                    chat_histories[channel_id].append({"role": "assistant", "content": ai_response})
                    await message.channel.send(ai_response)
                else:
                    print(f"❌ HF Status Error: {response.status_code} - {response.text}")
                    await message.channel.send("⚠️ Received an invalid response structure from Hugging Face.")
                    
            except Exception as e:
                print(f"❌ Network Request Error: {e}")
                await message.channel.send("⚠️ Lost my connection thread to Hugging Face. Try typing it again!")

            if len(chat_histories[channel_id]) > 10:
                chat_histories[channel_id] = chat_histories[channel_id][-10:]

# 3. CONCURRENT LAUNCH TRACK
server_thread = Thread(target=run_web_server)
server_thread.daemon = True
server_thread.start()

print("🚀 Launching Discord client engine...")
bot.run(DISCORD_TOKEN)
