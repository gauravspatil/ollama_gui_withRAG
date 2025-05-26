
import tkinter as tk
from tkinter import scrolledtext, messagebox
import requests
import threading
from tkinter import ttk

# Import utility modules
from rag_utils import split_text, get_embeddings, retrieve_context, get_query_embedding
from file_utils import load_knowledge_base_files, save_chat_history, load_chat_history, list_chat_logs, delete_chat_log
from gui_utils import show_about_popup, show_citation_popup

OLLAMA_API_URL = "http://localhost:11434"

# Get the running model from Ollama

def get_running_model():
    try:
        resp = requests.get(f"{OLLAMA_API_URL}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        if models:
            # Pick the first running model
            return models[0]["name"]
        else:
            return None
    except Exception as e:
        return None

import os
import json

class OllamaChatGUI:
    def toggle_chain_of_thought(self):
        """
        Toggle hiding/showing chain of thought in the chat area.
        """
        self.chain_of_thought_hidden = self.cot_var.get()
        self.update_chat_area_chain_of_thought()

    def update_chat_area_chain_of_thought(self):
        """
        Re-render the chat area, hiding or showing chain of thought as needed.
        """
        # Save scroll position
        yview = self.chat_area.yview()
        self.chat_area.config(state='normal')
        # To support unhiding, we need to reconstruct the chat from the message history
        # We'll keep a list of (sender, message) tuples as self.chat_history
        if not hasattr(self, 'chat_history'):
            # If chat_history doesn't exist, build it from the current chat area (best effort, only for first run)
            self.chat_history = []
            lines = self.chat_area.get('1.0', tk.END).splitlines()
            current_sender = None
            current_message = []
            for line in lines:
                if line.endswith(':') and not line.startswith(' '):
                    if current_sender is not None:
                        self.chat_history.append((current_sender, '\n'.join(current_message)))
                    current_sender = line[:-1]
                    current_message = []
                else:
                    current_message.append(line)
            if current_sender is not None:
                self.chat_history.append((current_sender, '\n'.join(current_message)))

        # Remove all content
        self.chat_area.delete('1.0', tk.END)
        import re
        for sender, message in getattr(self, 'chat_history', []):
            if getattr(self, 'chain_of_thought_hidden', False):
                # Hide complete <think>...</think> blocks
                message = re.sub(r'<think>[\s\S]*?</think>', '[Chain of thought hidden]', message, flags=re.IGNORECASE)
                # Hide any remaining unmatched <think> blocks (i.e., <think> without </think>)
                if '<think>' in message and '</think>' not in message:
                    message = re.sub(r'<think>[\s\S]*', '[Chain of thought hidden]', message, flags=re.IGNORECASE)
            self.chat_area.insert(tk.END, f"{sender}: {message}\n")
        self.chat_area.config(state='disabled')
        self.chat_area.yview_moveto(yview[0])
    
    
    def __init__(self, root):
        self.root = root
        self.root.title("Ollama LLM Chat GUI")
        self.models = self.get_all_models()
        if not self.models:
            messagebox.showerror("Model Error", "No models found in Ollama.")
            self.root.destroy()
            return
        self.model = self.models[0]
        self.stop_response = False
        self.citation_tag_counter = 0
        self.rag_user_prompt_template = (
            "You are an expert assistant. Use ONLY the following context to answer the question. "
            "If the answer is not in the context, say you don't know. Do NOT use prior knowledge.\n\n"
            "Context:\n{context}\n\nChat History:\n{history}\n\nQuestion: {question}\n\nAnswer:"
        )
        # Preferences (set before widgets for font/dark mode)
        self.preferences = {
            'rag_top_k': 2,
            'rag_similarity_threshold': 0.05,
            'rag_chunk_size': 400,
            'rag_chunk_overlap': 50,
            'auto_save_chat': True,
            'default_model': self.model,
            'font_size': 11,
            'dark_mode': False,
        }
        import sys
        if getattr(sys, 'frozen', False):
            # Running as a bundled app (PyInstaller)
            BASE_DIR = os.path.dirname(sys.executable)
        else:
            # Running as a script
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.chat_logs_dir = os.path.join(BASE_DIR, "chat_logs")
        os.makedirs(self.chat_logs_dir, exist_ok=True)
        self.current_chat_file = None
        self.create_widgets()

        # Add menu for chat logs
        self.create_menu()

        # Apply preferences (font/dark mode)
        self.apply_preferences()

        # Bind close event to save chat
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def stop_responding(self):
        self.stop_response = True
    # Only keep this version of create_widgets

    def create_widgets(self):
        self.chain_of_thought_hidden = False
        # Add Load Knowledge Base button and model dropdown at the top
        top_frame = tk.Frame(self.root)
        top_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=(10,0))

        self.kb_button = tk.Button(top_frame, text="Load Knowledge Base", command=self.load_knowledge_base)
        self.kb_button.grid(row=0, column=0, padx=(0,10))

        # Replace OptionMenu with ttk.Combobox for model selection
        self.model_var = tk.StringVar(value=self.model)
        self.model_dropdown = ttk.Combobox(top_frame, textvariable=self.model_var, values=self.models, state="readonly")
        self.model_dropdown.bind("<<ComboboxSelected>>", lambda event: self.on_model_select(self.model_var.get()))
        self.model_dropdown.grid(row=0, column=1, padx=(0,10))

        # Add Edit RAG Prompt button (disabled until KB is loaded)
        self.edit_rag_prompt_button = tk.Button(top_frame, text="Edit RAG Prompt", command=self.edit_rag_prompt, state=tk.DISABLED)
        self.edit_rag_prompt_button.grid(row=0, column=2, padx=(0,10))

        # Add About button
        self.about_button = tk.Button(top_frame, text="About", command=self.show_about)
        self.about_button.grid(row=0, column=3, padx=(0,10))

        # Add Hide Chain of Thought checkbox
        self.cot_var = tk.BooleanVar(value=False)
        self.cot_checkbox = tk.Checkbutton(top_frame, text="Hide Chain of Thought", variable=self.cot_var, command=self.toggle_chain_of_thought)
        self.cot_checkbox.grid(row=0, column=4, padx=(0,10))

        # Chat area
        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled', width=60, height=20)
        self.chat_area.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        # Entry and send button at the bottom
        entry_frame = tk.Frame(self.root)
        entry_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=(0,10))
        self.entry = tk.Entry(entry_frame, width=50)
        self.entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.entry.bind('<Return>', lambda event: self.send_message())
        self.send_button = tk.Button(entry_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=10)

        # Stop Responding button at the very bottom
        stop_frame = tk.Frame(self.root)
        stop_frame.grid(row=3, column=0, sticky='ew', padx=10, pady=(0,10))
        self.stop_button = tk.Button(stop_frame, text="Stop Responding", command=self.stop_responding)
        self.stop_button.pack(fill=tk.X)

        # Configure grid weights for resizing
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    def create_menu(self):
        menubar = tk.Menu(self.root)
        # Chats menu
        chats_menu = tk.Menu(menubar, tearoff=0)
        chats_menu.add_command(label="List Previous Chats", command=self.show_chat_logs_dialog)
        menubar.add_cascade(label="Chats", menu=chats_menu)

        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Preferences...", command=self.show_preferences_dialog)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # Add toplevel menu
        self.root.config(menu=menubar)

    def show_preferences_dialog(self):
        # Default preferences if not set
        if not hasattr(self, 'preferences'):
            self.preferences = {
                'rag_top_k': 2,
                'rag_similarity_threshold': 0.05,
                'rag_chunk_size': 400,
                'rag_chunk_overlap': 50,
                'auto_save_chat': True,
                'default_model': self.model,
                'font_size': 11,
                'dark_mode': False,
                'temperature': 0.7,
                'top_p': 1.0,
                'max_tokens': 2048,
            }
        prefs = self.preferences
        popup = tk.Toplevel(self.root)
        popup.title("Preferences")
        popup.geometry("440x540")
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)

        # RAG parameters
        tk.Label(frame, text="RAG Parameters", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky='w', pady=(0,2))
        tk.Label(frame, text="Top K:").grid(row=1, column=0, sticky='w')
        rag_top_k_var = tk.IntVar(value=prefs['rag_top_k'])
        tk.Entry(frame, textvariable=rag_top_k_var).grid(row=1, column=1, sticky='ew')
        tk.Label(frame, text="Similarity Threshold (0-1):").grid(row=2, column=0, sticky='w')
        rag_sim_thresh_var = tk.DoubleVar(value=prefs['rag_similarity_threshold'])
        tk.Entry(frame, textvariable=rag_sim_thresh_var).grid(row=2, column=1, sticky='ew')
        tk.Label(frame, text="Chunk Size (words):").grid(row=3, column=0, sticky='w')
        rag_chunk_size_var = tk.IntVar(value=prefs['rag_chunk_size'])
        tk.Entry(frame, textvariable=rag_chunk_size_var).grid(row=3, column=1, sticky='ew')
        tk.Label(frame, text="Chunk Overlap (words):").grid(row=4, column=0, sticky='w')
        rag_chunk_overlap_var = tk.IntVar(value=prefs['rag_chunk_overlap'])
        tk.Entry(frame, textvariable=rag_chunk_overlap_var).grid(row=4, column=1, sticky='ew')

        # Model Parameters
        tk.Label(frame, text="Model Parameters", font=("Arial", 11, "bold")).grid(row=5, column=0, columnspan=2, sticky='w', pady=(16,2))
        tk.Label(frame, text="Temperature (0-1):").grid(row=6, column=0, sticky='w')
        temp_var = tk.DoubleVar(value=prefs.get('temperature', 0.7))
        tk.Entry(frame, textvariable=temp_var).grid(row=6, column=1, sticky='ew')
        tk.Label(frame, text="Top-p (0-1):").grid(row=7, column=0, sticky='w')
        topp_var = tk.DoubleVar(value=prefs.get('top_p', 1.0))
        tk.Entry(frame, textvariable=topp_var).grid(row=7, column=1, sticky='ew')
        tk.Label(frame, text="Max tokens:").grid(row=8, column=0, sticky='w')
        max_tokens_var = tk.IntVar(value=prefs.get('max_tokens', 2048))
        tk.Entry(frame, textvariable=max_tokens_var).grid(row=8, column=1, sticky='ew')

        # GUI Preferences
        tk.Label(frame, text="GUI Preferences", font=("Arial", 11, "bold")).grid(row=9, column=0, columnspan=2, sticky='w', pady=(16,2))
        auto_save_var = tk.BooleanVar(value=prefs['auto_save_chat'])
        tk.Checkbutton(frame, text="Auto-save chat logs", variable=auto_save_var).grid(row=10, column=0, columnspan=2, sticky='w')
        tk.Label(frame, text="Default Model:").grid(row=11, column=0, sticky='w')
        default_model_var = tk.StringVar(value=prefs['default_model'])
        tk.OptionMenu(frame, default_model_var, *self.models).grid(row=11, column=1, sticky='ew')
        tk.Label(frame, text="Font Size:").grid(row=12, column=0, sticky='w')
        font_size_var = tk.IntVar(value=prefs['font_size'])
        tk.Entry(frame, textvariable=font_size_var).grid(row=12, column=1, sticky='ew')
        dark_mode_var = tk.BooleanVar(value=prefs['dark_mode'])
        tk.Checkbutton(frame, text="Dark Mode", variable=dark_mode_var).grid(row=13, column=0, columnspan=2, sticky='w')

        # Save/cancel buttons
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=14, column=0, columnspan=2, pady=(24,0), sticky='e')
        def save_prefs():
            try:
                prefs['rag_top_k'] = int(rag_top_k_var.get())
                prefs['rag_similarity_threshold'] = float(rag_sim_thresh_var.get())
                prefs['rag_chunk_size'] = int(rag_chunk_size_var.get())
                prefs['rag_chunk_overlap'] = int(rag_chunk_overlap_var.get())
                prefs['auto_save_chat'] = bool(auto_save_var.get())
                prefs['default_model'] = default_model_var.get()
                prefs['font_size'] = int(font_size_var.get())
                prefs['dark_mode'] = bool(dark_mode_var.get())
                prefs['temperature'] = float(temp_var.get())
                prefs['top_p'] = float(topp_var.get())
                prefs['max_tokens'] = int(max_tokens_var.get())
                self.apply_preferences()
                popup.destroy()
            except Exception:
                messagebox.showerror("Invalid Input", "Please enter valid values for all preferences.")
        save_btn = tk.Button(btn_frame, text="Save", command=save_prefs)
        save_btn.pack(side=tk.RIGHT, padx=(0,8))
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=popup.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=(0,8))
        frame.grid_columnconfigure(1, weight=1)

    def apply_preferences(self):
        # Apply font size and dark mode immediately
        prefs = getattr(self, 'preferences', None)
        if not prefs:
            return
        font = ("Arial", prefs.get('font_size', 11))
        widgets = [self.chat_area, self.entry]
        # Top frame widgets
        for w in [self.kb_button, self.model_dropdown, self.edit_rag_prompt_button, self.about_button, self.cot_checkbox]:
            try:
                w.config(font=font)
            except Exception:
                pass
        for w in widgets:
            try:
                w.config(font=font)
            except Exception:
                pass
        # Dark mode
        if prefs.get('dark_mode', False):
            bg = '#23272e'
            fg = '#e6e6e6'
            entry_bg = '#2d323b'
            self.root.configure(bg=bg)
            for widget in [self.chat_area, self.entry]:
                widget.config(bg=entry_bg, fg=fg, insertbackground=fg)
            for widget in [self.kb_button, self.model_dropdown, self.edit_rag_prompt_button, self.about_button, self.cot_checkbox, self.send_button, self.stop_button]:
                try:
                    widget.config(bg=bg, fg=fg, activebackground='#444', activeforeground=fg)
                except Exception:
                    pass
        else:
            # Reset to default
            self.root.configure(bg=None)
            for widget in [self.chat_area, self.entry]:
                widget.config(bg='white', fg='black', insertbackground='black')
            for widget in [self.kb_button, self.model_dropdown, self.edit_rag_prompt_button, self.about_button, self.cot_checkbox, self.send_button, self.stop_button]:
                try:
                    widget.config(bg=None, fg=None, activebackground=None, activeforeground=None)
                except Exception:
                    pass
        # Set default model if changed
        if self.model != prefs.get('default_model'):
            self.model = prefs.get('default_model')
            self.model_var.set(self.model)

    # Removed show_model_params_dialog: Model parameters are now in Preferences dialog

    def show_chat_logs_dialog(self):
        # Use utility to list chat logs
        files = list_chat_logs(self.chat_logs_dir)
        files.sort(reverse=True)
        popup = tk.Toplevel(self.root)
        popup.title("Previous Chats")
        popup.geometry("400x300")
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        listbox = tk.Listbox(frame, width=50, height=10)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for fname in files:
            listbox.insert(tk.END, fname)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=scrollbar.set)

        btn_frame = tk.Frame(popup)
        btn_frame.pack(fill=tk.X, pady=(10,0))
        open_btn = tk.Button(btn_frame, text="Open", command=lambda: self.open_selected_chat(listbox, files, popup))
        open_btn.pack(side=tk.LEFT, padx=5)
        del_btn = tk.Button(btn_frame, text="Delete", command=lambda: self.delete_selected_chat(listbox, files, popup))
        del_btn.pack(side=tk.LEFT, padx=5)
        close_btn = tk.Button(btn_frame, text="Close", command=popup.destroy)
        close_btn.pack(side=tk.RIGHT, padx=5)

    def open_selected_chat(self, listbox, files, popup):
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select a chat log to open.")
            return
        fname = files[sel[0]]
        fpath = os.path.join(self.chat_logs_dir, fname)
        try:
            self.chat_history = load_chat_history(fpath)
            self.current_chat_file = fpath
            self.update_chat_area_chain_of_thought()
            popup.destroy()
        except Exception as e:
            messagebox.showerror("Open Error", f"Could not open chat log:\n{e}")

    def delete_selected_chat(self, listbox, files, popup):
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select a chat log to delete.")
            return
        fname = files[sel[0]]
        fpath = os.path.join(self.chat_logs_dir, fname)
        if messagebox.askyesno("Delete Chat", f"Are you sure you want to delete '{fname}'?"):
            try:
                delete_chat_log(fpath)
                listbox.delete(sel[0])
            except Exception as e:
                messagebox.showerror("Delete Error", f"Could not delete chat log:\n{e}")

    def on_close(self):
        # Save chat history to file on close if auto-save is enabled
        auto_save = True
        if hasattr(self, 'preferences'):
            auto_save = self.preferences.get('auto_save_chat', True)
        if auto_save and hasattr(self, 'chat_history') and self.chat_history:
            try:
                save_chat_history(self.chat_logs_dir, self.chat_history, self.current_chat_file)
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save chat log:\n{e}")
        self.root.destroy()
 
    def show_about(self):
        show_about_popup(self.root)

    def load_knowledge_base(self):
        # Use the file_utils and rag_utils helpers
        kb_text = load_knowledge_base_files(self.root)
        if not kb_text:
            return
        self.kb_text = kb_text
        # Use preferences for chunking
        chunk_size = self.preferences.get('rag_chunk_size', 400)
        chunk_overlap = self.preferences.get('rag_chunk_overlap', 50)
        self.kb_chunks = split_text(self.kb_text, chunk_size=chunk_size, overlap=chunk_overlap)

        # Progress bar popup
        progress_popup = tk.Toplevel(self.root)
        progress_popup.title("Embedding Knowledge Base")
        progress_popup.geometry("400x100")
        progress_label = tk.Label(progress_popup, text="Embedding knowledge base chunks...")
        progress_label.pack(pady=10)
        progress = ttk.Progressbar(progress_popup, orient="horizontal", length=350, mode="determinate")
        progress.pack(pady=10)
        progress["maximum"] = len(self.kb_chunks)
        progress["value"] = 0
        progress_popup.update()

        # Use get_embeddings from rag_utils
        self.kb_embeddings = get_embeddings(self.kb_chunks, progress_bar=progress, progress_popup=progress_popup)
        progress_popup.destroy()
        messagebox.showinfo("Knowledge Base Loaded", f"Loaded {len(self.kb_chunks)} chunks from knowledge base.")

        # Enable the Edit RAG Prompt button now that KB is loaded
        self.edit_rag_prompt_button.config(state=tk.NORMAL)
    def edit_rag_prompt(self):
        # Popup to edit the RAG user prompt template
        popup = tk.Toplevel(self.root)
        popup.title("Edit RAG Prompt Template")
        popup.geometry("600x320")
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)
        label = tk.Label(frame, text="Edit the RAG prompt template below. Use {context} for the context, {history} for the chat history, and {question} for the user question.", wraplength=560, justify='left')        
        label.pack(anchor='w', pady=(0,8))
        text_widget = tk.Text(frame, wrap=tk.WORD, width=70, height=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, self.rag_user_prompt_template)
        def save_and_close():
            new_template = text_widget.get("1.0", tk.END).strip()
            if "{context}" not in new_template or "{question}" not in new_template or "{history}" not in new_template:
                messagebox.showerror("Template Error", "Template must include {context}, {history}, and {question}.")
                return
            self.rag_user_prompt_template = new_template
            popup.destroy()
        save_btn = tk.Button(frame, text="Save", command=save_and_close)
        save_btn.pack(side=tk.RIGHT, pady=(8,0), padx=(0,8))
        cancel_btn = tk.Button(frame, text="Cancel", command=popup.destroy)
        cancel_btn.pack(side=tk.RIGHT, pady=(8,0), padx=(0,8))


    # RAG utility methods are now imported from rag_utils
    
    def get_all_models(self):
        try:
            resp = requests.get(f"{OLLAMA_API_URL}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            # Filter out embedding models (e.g., those with 'embed' in the name)
            chat_models = [m["name"] for m in models if "embed" not in m["name"].lower() and "all-minilm" not in m["name"].lower()]
            return chat_models if chat_models else []
        except Exception:
            return []

    
    def on_model_select(self, value):
        self.model = value

    def send_message(self):
        user_message = self.entry.get().strip()
        if not user_message:
            return
        self.append_chat("You", user_message)
        self.entry.delete(0, tk.END)
        self.stop_response = False
        threading.Thread(target=self.get_llm_response, args=(user_message,), daemon=True).start()

    def append_chat(self, sender, message, streaming=False):
        # Maintain chat history for hide/unhide chain of thought
        if not hasattr(self, 'chat_history'):
            self.chat_history = []
        if streaming and sender == self.model:
            # Insert sender label and set last_agent_index to right after the label
            self.chat_area.config(state='normal')
            self.chat_area.insert(tk.END, f"{sender}:\n")
            self.last_agent_index = self.chat_area.index(tk.END)
            self.chat_area.config(state='disabled')
            self.chat_area.see(tk.END)
            # For streaming, add a placeholder to chat_history (will be updated in update_last_agent_message_stream)
            self.chat_history.append((sender, ""))
        else:
            self.chat_area.config(state='normal')
            self.chat_area.insert(tk.END, f"{sender}: {message}\n")
            self.last_agent_index = None
            self.chat_area.config(state='disabled')
            self.chat_area.see(tk.END)
            self.chat_history.append((sender, message))
        # Do not update chain of thought visibility here; only update on toggle

    def get_llm_response(self, user_message):
        try:
            # RAG: retrieve context if KB loaded
            if hasattr(self, 'kb_chunks') and hasattr(self, 'kb_embeddings'):
                prefs = getattr(self, 'preferences', {})
                top_k = prefs.get('rag_top_k', 2)
                threshold = prefs.get('rag_similarity_threshold', 0.05)
                context, context_chunks = retrieve_context(
                    user_message,
                    self.kb_chunks,
                    self.kb_embeddings,
                    top_k=top_k,
                    threshold=threshold,
                    get_query_embedding_func=get_query_embedding
                )
            else:
                context, context_chunks = "", []
            citation_text = ""
            citation_indices = []
            if context_chunks:
                citation_indices = [i for i, _ in context_chunks]
                citation_text = "\n\n[Citations: " + ", ".join([f"Chunk {i+1}" for i in citation_indices]) + "]"
            # RAG: Always send context and question as a single user message for best model compatibility
            # Format previous chat history (excluding current user message)
            history_lines = []
            for sender, msg in self.chat_history[:-1]:  # Exclude the just-appended user message
                history_lines.append(f"{sender}: {msg}")
            history_str = "\n".join(history_lines).strip()

            # RAG: Always send context, chat history, and question as a single user message for best model compatibility
            if context:
                user_prompt = self.rag_user_prompt_template.format(
                    context=context,
                    question=user_message,
                    history=history_str
                )
            else:
                user_prompt = f"You are an expert assistant. You have access to the following chat history:\n\n{history_str}\n\n{user_message}"
            # Use model parameters from preferences
            prefs = getattr(self, 'preferences', {})
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "options": {
                    "temperature": prefs.get('temperature', 0.7),
                    "top_p": prefs.get('top_p', 1.0),
                    "num_predict": prefs.get('max_tokens', 2048)
                }
            }
            resp = requests.post(f"{OLLAMA_API_URL}/api/chat", json=payload, stream=True)
            resp.raise_for_status()
            import json
            # Insert sender label and remember start index for streaming
            self.append_chat(self.model, "", streaming=True)
            first_chunk = True
            for line in resp.iter_lines():
                if self.stop_response:
                    break
                if line:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        self.update_last_agent_message_stream(chunk, append=(not first_chunk))
                        first_chunk = False
            # If nothing was streamed, show no response
            if first_chunk:
                self.update_last_agent_message_stream("[No response]", append=False)
            else:
                # After streaming is done, append citations at the end as a clickable tag
                if citation_text:
                    self.insert_citation_tag(citation_text, citation_indices)
        except Exception as e:
            self.append_chat(self.model, f"[Error: {e}]")

    def insert_citation_tag(self, citation_text, chunk_indices):
        # Insert the citation text as a clickable tag at the end of the chat area, tag only the citation text (not newline)
        self.chat_area.config(state='normal')
        self.chat_area.mark_set(tk.INSERT, tk.END)  # Always insert at the end
        citation_text_stripped = citation_text.rstrip('\n')
        start_idx = self.chat_area.index(tk.END)
        self.chat_area.insert(tk.END, citation_text_stripped)
        end_idx = self.chat_area.index("insert")  # Use 'insert' to get the position after the citation text
        tag_name = f"citation_{self.citation_tag_counter}_{start_idx.replace('.', '_')}_{end_idx.replace('.', '_')}"
        self.citation_tag_counter += 1
        self.chat_area.tag_add(tag_name, start_idx, end_idx)
        self.chat_area.tag_config(tag_name, foreground="blue", underline=True)
        self.chat_area.tag_bind(tag_name, "<Button-1>", self._make_citation_callback(list(chunk_indices)))
        self.chat_area.insert(tk.END, "\n")
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)
        
    def _make_citation_callback(self, chunk_indices):
        def callback(event):
            self.show_citation_popup(chunk_indices)
            # Always move the cursor to the end and disable the widget after popup
            self.chat_area.mark_set(tk.INSERT, tk.END)
            self.chat_area.config(state='disabled')
            self.chat_area.see(tk.END)
        return callback

    def show_citation_popup(self, chunk_indices):
        # Use the helper from gui_utils (correct argument order)
        show_citation_popup(self.root, getattr(self, 'kb_chunks', []), chunk_indices)

    def update_last_agent_message_stream(self, message, append=False):
        self.chat_area.config(state='normal')
        if hasattr(self, 'last_agent_index') and self.last_agent_index:
            if append:
                self.chat_area.insert(tk.END, message)
                # Update last message in chat_history
                if hasattr(self, 'chat_history') and self.chat_history:
                    sender, prev_msg = self.chat_history[-1]
                    self.chat_history[-1] = (sender, prev_msg + message)
            else:
                # First chunk: clear any previous content after label
                self.chat_area.delete(self.last_agent_index, tk.END)
                self.chat_area.insert(tk.END, message)
                # Set last message in chat_history
                if hasattr(self, 'chat_history') and self.chat_history:
                    sender, _ = self.chat_history[-1]
                    self.chat_history[-1] = (sender, message)
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)
        # Do not update chain of thought visibility here; only update on toggle
    

def main():
    root = tk.Tk()
    app = OllamaChatGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
