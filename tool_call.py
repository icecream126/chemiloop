from openai import OpenAI

client = OpenAI(api_key="sk-577e021a148d45e4b5e842fc43a6a07c", base_url="https://api.deepseek.com")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather of a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country, e.g., Paris, France"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol like AAPL"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_news_headlines",
            "description": "Get weather-related news",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic of news, e.g., weather"}
                },
                "required": ["topic"]
            }
        }
    },
]

messages = [{"role": "user", "content": "What's the weather in Paris and latest weather news?"}]

# Simulated tool responses
tool_outputs = {
    "get_weather": "It's 17°C and cloudy in Paris.",
    "get_weather_news_headlines": "Top story: Storm expected in France this weekend."
}

while True:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    
    message = response.choices[0].message
    messages.append(message)

    if not hasattr(message, "tool_calls") or not message.tool_calls:
        break  # No more tool calls, proceed to final output
    import pdb; pdb.set_trace()
    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        tool_response = tool_outputs.get(tool_name, "Tool not implemented.")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_response
        })

print("Final LLM Response:\n", message.content)
