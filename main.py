import discord
from discord.ext import commands
import os
import asyncio
from gradio_client import Client
from flask import Flask
from threading import Thread

# 🌟 1. Create a tiny web server to satisfy Render's Free Web Service requirements
app = Flask('')

@app.route('/')
def home():
    return "Bot is alive and healthy!"

def run_web_server():
    # Render provides a specific PORT environment variable we must bind to
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# Start the web server on a background loop thread
Thread(target=run_web_server).start()

# 🌟 2. Your original Discord Bot setup continues exactly the same below...
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

HF_TOKEN = os.environ.get("HF_TOKEN")
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
hf_client = Client("royalpig7/royalpig-free-bot", token=HF_TOKEN)

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
                loop = asyncio.get_event_loop()
                ai_response = await loop.run_in_executor(
                    None, 
                    lambda: hf_client.predict(history_context)
                )
                
                if ai_response:
                    chat_histories[channel_id].append({"role": "assistant", "content": ai_response})
                    await message.channel.send(ai_response)
                else:
                    await message.channel.send("Thinking... but nothing came out. Hit me again!")
                    
            except Exception as e:
                print(f"❌ Gradio Framework Error: {e}")
                await message.channel.send("⚠️ Lost my connection thread to Hugging Face. Try typing it again!")

            if len(chat_histories[channel_id]) > 10:
                chat_histories[channel_id] = chat_histories[channel_id][-10:]

bot.run(DISCORD_TOKEN)
