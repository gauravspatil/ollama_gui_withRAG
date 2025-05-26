def set_dark_mode_popup(popup):
    """Apply dark mode styling to a popup and all its child widgets recursively."""
    bg = '#23272e'
    fg = '#e6e6e6'
    entry_bg = '#2d323b'
    popup.configure(bg=bg)
    border = '#444a52'
    def style_widget(widget):
        # Set widget-specific colors and borders
        if isinstance(widget, (tk.Frame, tk.LabelFrame)):
            widget.config(bg=bg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
        elif isinstance(widget, tk.Label):
            widget.config(bg=bg, fg=fg)
        elif isinstance(widget, tk.Entry):
            widget.config(bg=entry_bg, fg=fg, insertbackground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
        elif isinstance(widget, tk.Text):
            widget.config(bg=entry_bg, fg=fg, insertbackground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
        elif isinstance(widget, tk.Button):
            widget.config(bg=bg, fg=fg, activebackground='#444', activeforeground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1, borderwidth=1)
        elif isinstance(widget, tk.Checkbutton):
            widget.config(bg=bg, fg=fg, activebackground='#444', activeforeground=fg, selectcolor=bg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
        elif isinstance(widget, tk.Listbox):
            widget.config(bg=entry_bg, fg=fg, selectbackground='#444', selectforeground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
        elif isinstance(widget, tk.Scrollbar):
            widget.config(bg=bg, troughcolor=bg, activebackground='#444', highlightbackground=border, highlightcolor=border, highlightthickness=1)
        elif isinstance(widget, scrolledtext.ScrolledText):
            widget.config(bg=entry_bg, fg=fg, insertbackground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
        # ttk Combobox and OptionMenu
        try:
            import tkinter.ttk as ttk
            if isinstance(widget, ttk.Combobox):
                style = ttk.Style()
                style.theme_use('clam')
                style.configure('TCombobox', fieldbackground=entry_bg, background=entry_bg, foreground=fg, bordercolor=border, lightcolor=border, darkcolor=border, borderwidth=1)
                style.map('TCombobox', fieldbackground=[('readonly', entry_bg)], background=[('readonly', entry_bg)], foreground=[('readonly', fg)])
                widget.configure(style='TCombobox')
        except Exception:
            pass
        # Recursively style children
        for child in getattr(widget, 'winfo_children', lambda:[])():
            style_widget(child)
    style_widget(popup)
import tkinter as tk
from tkinter import messagebox, scrolledtext

def show_about_popup(popup):
    import webbrowser
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

def show_citation_popup(parent, chunk_indices):
    import tkinter as tk
    from tkinter import scrolledtext
    root = parent.root if hasattr(parent, 'root') else parent
    kb_chunks = getattr(parent, 'kb_chunks', [])
    popup = tk.Toplevel(parent.root)
    popup.title("Citations")
    popup.geometry("600x400")
    frame = tk.Frame(popup)
    frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)
    label = tk.Label(frame, text="Cited Knowledge Chunks:", font=("Arial", 11, "bold"))
    label.pack(anchor='w', pady=(0,8))
    text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=70, height=18, state='normal')
    text.pack(fill=tk.BOTH, expand=True)
    for idx in chunk_indices:
        if 0 <= idx < len(kb_chunks):
            chunk = kb_chunks[idx]
            text.insert(tk.END, f"Chunk {idx+1}:\n{chunk}\n\n")
        else:
            text.insert(tk.END, f"Chunk {idx+1}: [Not found]\n\n")
    text.config(state='disabled')
    close_btn = tk.Button(frame, text="Close", command=popup.destroy)
    close_btn.pack(anchor='e', pady=(10,0))
    if hasattr(parent, 'preferences') and parent.preferences.get('dark_mode', False):
        set_dark_mode_popup(popup)
