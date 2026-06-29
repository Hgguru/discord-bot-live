import discord
from discord.ext import commands
import os
import asyncio
from gradio_client import Client

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Connect securely to Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN")
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
hf_client = Client("royalpig7/royalpig-free-bot", token=HF_TOKEN)

# Memory bank format: { channel_id: [ {"role": "user/assistant", "content": "..."} ] }
chat_histories = {}

@bot.event
async def on_ready():
    print(f"✅ Royalpig AI Bot is live and connected as {bot.user}!")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Trigger via Mention or Direct Message (DM)
    if bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        async with message.channel.typing():
            channel_id = message.channel.id
            
            # Clean up user's message (stripping out the bot's raw mention ID tag)
            clean_input = message.clean_content.replace(f"@{bot.user.name}", "").strip()
            
            # Initialize history tracker for this specific channel if empty
            if channel_id not in chat_histories:
                chat_histories[channel_id] = []
                
            # 1. Append user's new message to history context
            chat_histories[channel_id].append({"role": "user", "content": clean_input})

            # 2. Build a clear, chronological context string out of past chat records
            # This helps your Llama 2 fine-tune see the active conversation stream
            history_context = ""
            for turn in chat_histories[channel_id][-5:]:  # Pull only up to last 5 logs
                prefix = "User: " if turn["role"] == "user" else "Assistant: "
                history_context += f"{prefix}{turn['content']}\n"

            try:
                # Run the Gradio prediction worker securely in a background thread
                loop = asyncio.get_event_loop()
                ai_response = await loop.run_in_executor(
                    None, 
                    lambda: hf_client.predict(history_context)
                )
                
                if ai_response:
                    # 3. Log the AI's response to history bank before returning it
                    chat_histories[channel_id].append({"role": "assistant", "content": ai_response})
                    await message.channel.send(ai_response)
                else:
                    await message.channel.send("Thinking... but nothing came out. Hit me again!")
                    
            except Exception as e:
                print(f"❌ Gradio Framework Error: {e}")
                await message.channel.send("⚠️ Lost my connection thread to Hugging Face. Try typing it again!")

            # 4. Memory Cleanup: trim the rolling buffer to stop memory leak bloat
            if len(chat_histories[channel_id]) > 10:
                chat_histories[channel_id] = chat_histories[channel_id][-10:]

bot.run(DISCORD_TOKEN)