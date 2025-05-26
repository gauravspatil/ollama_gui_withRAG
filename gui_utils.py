import tkinter as tk
from tkinter import messagebox, scrolledtext

def show_about_popup(root):
    import webbrowser
    popup = tk.Toplevel(root)
    popup.title("About")
    popup.geometry("480x370")
    frame = tk.Frame(popup)
    frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)
    app_label = tk.Label(frame, text="Ollama LLM Chat GUI", font=("Arial", 14, "bold"))
    app_label.pack(anchor='w')
    desc = "A local knowledge-augmented chat GUI for Ollama LLMs."
    desc_label = tk.Label(frame, text=desc, font=("Arial", 10), justify='left', anchor='w')
    desc_label.pack(anchor='w', pady=(2, 8))
    author_label = tk.Label(frame, text="Author: Gaurav Patil", font=("Arial", 10), anchor='w', justify='left')
    author_label.pack(anchor='w')
    github_label = tk.Label(frame, text="GitHub: github.com/gauravspatil", fg="blue", cursor="hand2", font=("Arial", 10), anchor='w', justify='left')
    github_label.pack(anchor='w')
    def open_github(event=None):
        webbrowser.open_new("https://github.com/gauravspatil")
    github_label.bind("<Button-1>", open_github)
    sep = tk.Label(frame, text="", font=("Arial", 2))
    sep.pack(anchor='w', pady=(2,0))
    ollama_label = tk.Label(frame, text="Ollama is required (not bundled). Download from:", font=("Arial", 10, "bold"), anchor='w', justify='left')
    ollama_label.pack(anchor='w', pady=(6,0))
    ollama_link = tk.Label(frame, text="https://ollama.com/download", fg="blue", cursor="hand2", font=("Arial", 10), anchor='w', justify='left')
    ollama_link.pack(anchor='w')
    def open_ollama(event=None):
        webbrowser.open_new("https://ollama.com/download")
    ollama_link.bind("<Button-1>", open_ollama)
    ollama_note = tk.Label(frame, text="After installing, use: ollama pull <model>\nto download models (e.g., ollama pull llama3)", font=("Arial", 9), anchor='w', justify='left')
    ollama_note.pack(anchor='w', pady=(2,8))
    copyright_label = tk.Label(frame, text="© 2025 Gaurav Patil", font=("Arial", 10), anchor='w', justify='left')
    copyright_label.pack(anchor='w', pady=(8,0))
    oss_label = tk.Label(
        frame,
        text="Open Source under the MIT License. You may use, modify, and distribute this project as long as you provide credit to the author.\n\nCredits: tkinter, requests, numpy, PyPDF2, python-docx, Ollama.",
        font=("Arial", 9), anchor='w', justify='left', wraplength=440)
    oss_label.pack(anchor='w', pady=(4,0))
    mit_label = tk.Label(frame, text="License: MIT", font=("Arial", 9, "italic"), anchor='w', justify='left')
    mit_label.pack(anchor='w', pady=(2,0))
    close_btn = tk.Button(frame, text="Close", command=popup.destroy)
    close_btn.pack(anchor='e', pady=(16,0))

def show_citation_popup(root, kb_chunks, chunk_indices):
    popup = tk.Toplevel(root)
    popup.title("Cited Knowledge Base Chunks")
    popup.geometry("600x400")
    text_area = scrolledtext.ScrolledText(popup, wrap=tk.WORD, state='normal', width=80, height=20)
    text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    for idx in chunk_indices:
        chunk_num = idx + 1
        chunk_text = kb_chunks[idx] if kb_chunks and idx < len(kb_chunks) else '[Missing chunk]'
        text_area.insert(tk.END, f"--- Chunk {chunk_num} ---\n{chunk_text}\n\n")
    text_area.config(state='disabled')
    close_btn = tk.Button(popup, text="Close", command=popup.destroy)
    close_btn.pack(pady=(0,10))
