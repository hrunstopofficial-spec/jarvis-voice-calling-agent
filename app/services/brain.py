import os
import json
import logging
from app.core.config import GEMINI_API_KEY, GROQ_API_KEY
from app.core.prompt_engine import build_system_prompt
from app.core.memory import session_memory
from app.services.mcp_agent_bridge import get_dynamic_mcp_tools, execute_mcp_or_system_tool

logger = logging.getLogger("brain")

async def process_user_turn(user_message: str) -> str:
    """
    Processes Mukil's speech turn through the Autonomous Agentic Loop:
    1. Understands intent & conversational context
    2. Deep Reasoning on whether tool invocation is required
    3. Executes FastMCP industrial tools (SGC Ledger, Recipes, Overdue, Drive, Turso)
    4. Delivers crisp, natural Tanglish executive response (< 30 words) for voice output.
    """
    system_prompt = build_system_prompt()
    conversation_transcript = session_memory.format_history_for_prompt()
    dynamic_tools = get_dynamic_mcp_tools()

    # 1. Primary: Ultra-Fast Autonomous Groq Agent with FastMCP Tool Calling
    if GROQ_API_KEY:
        try:
            from groq import Groq
            groq_client = Groq(api_key=GROQ_API_KEY)
            
            messages = [
                {"role": "system", "content": system_prompt + "\n\nCRITICAL: Keep spoken answers under 25 words. Always address user as Boss. Use natural Tanglish + English. Zero hallucination — always call MCP tools for financial or operational facts."},
            ]
            
            # Add past turns from session memory
            for turn in session_memory.get_recent_turns(6):
                messages.append({"role": turn["role"], "content": turn["content"]})
            
            messages.append({"role": "user", "content": user_message})

            # Multi-turn Autonomous Tool Loop (up to 3 investigative steps)
            for step in range(3):
                completion = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    tools=dynamic_tools,
                    tool_choice="auto",
                    max_tokens=350,
                    temperature=0.6
                )
                choice = completion.choices[0]
                message = choice.message

                if not message.tool_calls:
                    # Model provided final spoken response
                    if message.content and message.content.strip():
                        reply = message.content.strip()
                        session_memory.add_turn("user", user_message)
                        session_memory.add_turn("assistant", reply)
                        return reply
                    break

                # Model requested tool calls
                messages.append(message)
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args_raw = tool_call.function.arguments or "{}"
                    try:
                        fn_args = json.loads(fn_args_raw)
                    except Exception:
                        fn_args = {}

                    # Execute via FastMCP / System Dispatcher
                    tool_output = await execute_mcp_or_system_tool(fn_name, fn_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_output)
                    })

            # If 3 steps executed, generate final voice synthesis
            final_turn = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages + [{"role": "user", "content": "Synthesize all investigative tool outputs into a single crisp, confident spoken Tanglish reply under 25 words."}],
                max_tokens=250,
                temperature=0.6
            )
            if final_turn.choices[0].message.content:
                reply = final_turn.choices[0].message.content.strip()
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
