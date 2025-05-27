
# Ollama GUI Chat with RAG & Tools

A user-friendly Python Tkinter desktop app for chatting with local Ollama LLMs, featuring Retrieval-Augmented Generation (RAG), tool commands, and a modern GUI.

## Features

- **Model selection**: Auto-detects available models (excludes embedding models)
- **Retrieval-Augmented Generation (RAG)**: Load multiple `.txt`, `.pdf`, or `.docx` files as a knowledge base
- **Citations**: Clickable citations for retrieved knowledge chunks
- **Tool Commands**: Use `/summarise`, `/scrapeweb`, and more, anywhere in your message
- **Streaming Output**: See model responses in real time
- **Interrupt Generation**: Stop the model mid-response
- **Dark Mode**: Toggle dark mode in Preferences

## Quick Start

1. **Install [Ollama](https://ollama.com/download)** (required, not bundled)
    - Pull a chat model (e.g. `ollama pull llama3`)
2. Install Python 3.8+
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the app:
    ```bash
    python ollama_gui.py
    ```

## Tool Commands

You can use tool commands anywhere in your message. Example tools:

- `/summarise` — Summarize the entire loaded knowledge base and present the main points.
- `/scrapeweb` — Extracts a web link from your message, fetches the web page, and uses its content as LLM context.

Click the **Tools** button in the app for a full list and descriptions.

## Packaging

- Use PyInstaller with `ollama_gui.spec` to build a standalone executable.

## License

MIT License. See [LICENSE](LICENSE).

## Author & Credits

- Author: Gaurav Patil ([gauravspatil](https://github.com/gauravspatil))
- Powered by [Ollama](https://ollama.com/)
- PDF support: [PyPDF2](https://pypi.org/project/PyPDF2/)
- DOCX support: [python-docx](https://pypi.org/project/python-docx/)

---

**Note:** Ollama must be installed and running locally. See the About dialog in the app for more info.
