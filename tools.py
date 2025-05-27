"""
Tool registry for user-invoked commands in the chat GUI.
Each tool is a function that takes (gui_instance, user_message) and returns (new_user_message, context_override, tool_response).
"""

def summarise_tool(gui, user_message):
    """
    Summarize the entire loaded knowledge base and present the main points.
    """
    context = ""
    for i, chunk in enumerate(getattr(gui, 'kb_chunks', [])):
        context += f"[Chunk {i+1}]\n{chunk}\n\n"
    new_user_message = (
        "Please summarize the entire context provided below. "
        "Extract the main points and present them concisely."
    )
    return new_user_message, context, None

import re
import requests
def scrapeweb_tool(gui, user_message):
    """
    Scrape the main content from a web page mentioned in your message and use it as context for the LLM.
    """
    # Find the first URL in the user message
    url_match = re.search(r"https?://\S+", user_message)
    if not url_match:
        return ("Sorry, I couldn't find a valid web link in your message.", None, "No URL found in the message.")
    url = url_match.group(0)
    # Try to fetch the web page content
    try:
        from gui_utils import set_dark_mode_popup  # Just to ensure import works, not used here
        # Use requests to fetch the page
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        # Try to extract main text content (very basic: strip HTML tags)
        import re as _re
        text = resp.text
        # Remove script/style
        text = _re.sub(r'<(script|style)[^>]*>[\s\S]*?</\\1>', '', text, flags=_re.IGNORECASE)
        # Remove HTML tags
        text = _re.sub(r'<[^>]+>', '', text)
        # Collapse whitespace
        text = _re.sub(r'\s+', ' ', text)
        # Truncate to a reasonable length for LLM context
        context = text[:6000]
        new_user_message = f"Using the following web page content as context, answer the user's question."
        tool_response = f"Web page scraped: {url} (first 500 chars):\n{text[:500]}..."
        return new_user_message, context, tool_response
    except Exception as e:
        return (f"Sorry, I couldn't fetch or process the web page: {url}", None, f"Error: {e}")


# Tool registry: command name (without slash) -> function
TOOLS = {
    "summarise": summarise_tool,
    "scrapeweb": scrapeweb_tool,
}

def get_tool(command):
    """Return the tool function for a given command name, or None if not found."""
    return TOOLS.get(command)
