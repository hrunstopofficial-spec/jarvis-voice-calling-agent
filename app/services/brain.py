import os
import json
import logging
from app.core.config import GEMINI_API_KEY, GROQ_API_KEY
from app.core.prompt_engine import build_system_prompt
from app.core.memory import session_memory
from app.services.tools import TOOL_DEFINITIONS, execute_tool_call

logger = logging.getLogger("brain")

async def process_user_turn(user_message: str) -> str:
    """
    Processes Mukil's speech turn through the Autonomous Agentic Loop:
    1. Understands intent & conversational context
    2. Deep Reasoning on whether tool invocation is required
    3. Executes ecosystem tools (PC Vitals, SGC Ledger, Mill Scraper, Telegram, Turso)
    4. Delivers crisp, natural Tanglish executive response (< 30 words) for voice output.
    """
    system_prompt = build_system_prompt()
    conversation_transcript = session_memory.format_history_for_prompt()

    # 1. Primary: Ultra-Fast Autonomous Groq Agent with Tool Calling
    if GROQ_API_KEY:
        try:
            from groq import Groq
            groq_client = Groq(api_key=GROQ_API_KEY)
            
            messages = [
                {"role": "system", "content": system_prompt + "\n\nCRITICAL: Keep spoken answers under 25 words. Always address user as Boss. Use natural Tanglish + English."},
            ]
            
            # Add past turns from session memory
            for turn in session_memory.get_recent_turns(6):
                messages.append({"role": turn["role"], "content": turn["content"]})
            
            messages.append({"role": "user", "content": user_message})

            # First turn inference with tool calling enabled
            completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=350,
                temperature=0.6
            )

            choice = completion.choices[0]
            message = choice.message

            # Check if model requested tool call(s)
            if message.tool_calls:
                messages.append(message)
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args_raw = tool_call.function.arguments or "{}"
                    try:
                        fn_args = json.loads(fn_args_raw)
                    except Exception:
                        fn_args = {}

                    # Execute the tool
                    tool_output = execute_tool_call(fn_name, fn_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_output)
                    })

                # Follow-up completion to synthesize tool results into final voice answer
                followup = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    max_tokens=250,
                    temperature=0.6
                )
                final_content = followup.choices[0].message.content
                if final_content and final_content.strip():
                    reply = final_content.strip()
                    session_memory.add_turn("user", user_message)
                    session_memory.add_turn("assistant", reply)
                    return reply

            # No tool call needed — direct conversational answer
            if message.content and message.content.strip():
                reply = message.content.strip()
                session_memory.add_turn("user", user_message)
                session_memory.add_turn("assistant", reply)
                return reply

        except Exception as ge:
            print(f"[Brain Agent Error] Groq Tool Loop encountered error: {ge}. Attempting Gemini fallback...")

    # 2. Secondary: Gemini Fallback
    if GEMINI_API_KEY:
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            full_prompt = (
                f"{system_prompt}\n\n"
                f"CONVERSATION HISTORY:\n{conversation_transcript}\n\n"
                f"Mukil says: \"{user_message}\"\n\n"
                f"JARVIS Spoken Response (Crisp Tanglish, under 25 words):"
            )
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=full_prompt
            )
            if response and response.text:
                reply = response.text.strip()
                session_memory.add_turn("user", user_message)
                session_memory.add_turn("assistant", reply)
                return reply
        except Exception as e:
            print(f"[Brain Error] Gemini unavailable: {e}")

    # 3. Static Safety Fallback
    fallback = "Boss, unga voice reach aachu! System 100% active. Adutha action execute panlama?"
    session_memory.add_turn("user", user_message)
    session_memory.add_turn("assistant", fallback)
    return fallback
