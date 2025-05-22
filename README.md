# Ollama GUI Chat with RAG

A user-friendly Python Tkinter desktop app for chatting with local Ollama LLMs, featuring:

- **Model selection** (auto-detects available models, excludes embedding models)
- **Retrieval-Augmented Generation (RAG)**: Load multiple .txt, .pdf, or .docx files as a knowledge base
- **Citations**: Clickable citations for retrieved knowledge chunks

## Quick Start

1. **Install [Ollama](https://ollama.com/download)** (required, not bundled)
   - Pull a chat model (e.g. `ollama pull llama3`)
2. Install Python 3.8+
3. Install dependencies:
   ```bash
   pip install tkinter requests PyPDF2 python-docx numpy
   ```
4. Run the app:
   ```bash
   python ollama_gui.py
   ```

## Packaging
- Use PyInstaller with `ollama_gui.spec` to build a standalone executable.
- No requirements.txt is included; dependencies are listed above.

## Features
- **Multi-file Knowledge Base**: Load and merge multiple .txt, .pdf, and .docx files.
- **RAG with Citations**: Relevant knowledge chunks are retrieved and cited in responses.
- **Streaming Output**: See model responses in real time.
- **Interrupt Generation**: Stop the model mid-response.

## License
MIT License. See [LICENSE](LICENSE).

## Author & Credits
- Author: Gaurav Patil ([gauravspatil](https://github.com/gauravspatil))
- Powered by [Ollama](https://ollama.com/)
- PDF support: [PyPDF2](https://pypi.org/project/PyPDF2/)
- DOCX support: [python-docx](https://pypi.org/project/python-docx/)

---

**Note:** Ollama must be installed and running locally. See the About dialog in the app for more info.
