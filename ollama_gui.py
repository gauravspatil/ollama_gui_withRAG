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
    def show_image_preview(self):
        # Remove any previous preview
        if hasattr(self, 'image_preview_index'):
            self.chat_area.config(state='normal')
            self.chat_area.delete(self.image_preview_index, f"{self.image_preview_index} lineend+1c")
            del self.image_preview_index
            self.chat_area.config(state='disabled')
        # Show new preview if a pending image exists
        if hasattr(self, 'pending_thumbnail') and self.pending_thumbnail:
            self.chat_area.config(state='normal')
            self.chat_area.insert(tk.END, "You (image attached): ")
            self.chat_area.image_create(tk.END, image=self.pending_thumbnail)
            self.chat_area.insert(tk.END, "\n")
            self.image_preview_index = self.chat_area.index(f'end-2l')
            self.chat_area.config(state='disabled')
            self.chat_area.see(tk.END)

    def clear_pending_image(self):
        # Remove any pending image and thumbnail from the instance
        if hasattr(self, 'pending_image'):
            del self.pending_image
        if hasattr(self, 'pending_thumbnail'):
            del self.pending_thumbnail
    def model_supports_images(self):
        """
        Returns True if the current model is known to support image input.
        The list is based on Ollama's official vision/multimodal models as of May 2025:
        https://ollama.com/search?c=vision
        Update this list as new models are released.
        """
        model_name = self.model.lower()
        # Substrings for all known vision/multimodal models (see Ollama vision models page)
        vision_model_substrings = [
            "gemma3", "llama4", "qwen2.5vl", "llava", "llava-next", "llava-llama3", "llava-phi3",
            "llama3.2-vision", "minicpm-v", "moondream", "bakllava", "mistral-small3.1", "granite3.2-vision",
            "multimodal", "image"  # keep generic terms for future-proofing
        ]
        return any(x in model_name for x in vision_model_substrings)

    def send_image_to_model(self, image_bytes, user_message="", ext="png"):
        # This assumes the Ollama API supports image input as base64 or multipart (LLaVA, etc.)
        import base64
        import requests
        # Convert image to base64
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Compose the payload for /api/generate (Ollama expects 'prompt' and 'images' fields)
        prefs = getattr(self, 'preferences', {})
        payload = {
            "model": self.model,
            "prompt": user_message or "Describe this image.",
            "images": [img_b64],
            "options": {
                "temperature": prefs.get('temperature', 0.7),
                "top_p": prefs.get('top_p', 1.0),
                "num_predict": prefs.get('max_tokens', 2048)
            }
        }
        
        try:
            resp = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload, stream=True)
            resp.raise_for_status()
            import json as _json
            self.append_chat(self.model, "", streaming=True)
            first_chunk = True
            for line in resp.iter_lines():
                if self.stop_response:
                    break
                if line:
                    data = _json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        self.update_last_agent_message_stream(chunk, append=(not first_chunk))
                        first_chunk = False
            if first_chunk:
                self.update_last_agent_message_stream("[No response]", append=False)
        except Exception as e:
            self.append_chat(self.model, f"[Error: {e}]")

    def on_paste_image(self, event=None):
        try:
            from PIL import ImageGrab, ImageTk
            import io
            img = ImageGrab.grabclipboard()
            if img:
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                # Clear any previous pending image/thumbnail
                self.clear_pending_image()
                # Prepare thumbnail
                tk_img = ImageTk.PhotoImage(img.resize((64, 64)))
                # Store only the most recent pending image and thumbnail
                self.pending_image = buf.getvalue()
                self.pending_thumbnail = tk_img  # Prevent garbage collection
                self.show_image_preview()
                # Clear the clipboard after pasting to prevent repeated attachment
                try:
                    import platform
                    if platform.system() == "Windows":
                        import ctypes
                        ctypes.windll.user32.OpenClipboard(0)
                        ctypes.windll.user32.EmptyClipboard()
                        ctypes.windll.user32.CloseClipboard()
                except Exception:
                    pass
            # If no image is found, do nothing (user may be pasting text)
        except ImportError:
            messagebox.showerror("Missing Dependency", "Pillow (PIL) is required for image paste support. Please install it with 'pip install pillow'.")
        except Exception as e:
            messagebox.showerror("Paste Error", f"Could not process clipboard image: {e}")

    def toggle_chain_of_thought(self):
        """
        Toggle hiding/showing chain of thought in the chat area.
        """
        self.chain_of_thought_hidden = self.cot_var.get()
        self.update_chat_area_chain_of_thought()

    def update_chat_area_chain_of_thought(self):
        """
        Re-render the chat area, hiding or showing chain of thought as needed.
        Supports persistent image thumbnails in chat history.
        """
        # Save scroll position
        yview = self.chat_area.yview()
        self.chat_area.config(state='normal')
        # To support unhiding, we need to reconstruct the chat from the message history
        # We'll keep a list of (sender, message) or (sender, message, image_thumbnail) tuples as self.chat_history
        if not hasattr(self, 'chat_history'):
            # If chat_history doesn't exist, build it from the current chat area (best effort, only for first run)
            self.chat_history = []
            lines = self.chat_area.get('1.0', tk.END).splitlines()
            current_sender = None
            current_message = []
            for line in lines:
                if line.endswith(':') and not line.startswith(' '):
                    if current_sender is not None:
                        self.chat_history.append((current_sender, '\n'.join(current_message), None))
                    current_sender = line[:-1]
                    current_message = []
                else:
                    current_message.append(line)
            if current_sender is not None:
                self.chat_history.append((current_sender, '\n'.join(current_message), None))

        # Remove all content
        self.chat_area.delete('1.0', tk.END)
        import re
        for idx, entry in enumerate(getattr(self, 'chat_history', [])):
            if len(entry) == 3:
                sender, message, _ = entry
            else:
                sender, message = entry
            image_thumbnail = self.chat_thumbnails.get(idx)
            if getattr(self, 'chain_of_thought_hidden', False):
                message = re.sub(r'<think>[\s\S]*?</think>', '[Chain of thought hidden]', message, flags=re.IGNORECASE)
                if '<think>' in message and '</think>' not in message:
                    message = re.sub(r'<think>.*', '[Chain of thought hidden]', message, flags=re.IGNORECASE)
            self.chat_area.insert(tk.END, f"{sender}: ")
            if image_thumbnail is not None:
                self.chat_area.image_create(tk.END, image=image_thumbnail)
                self.chat_area.insert(tk.END, " ")
            self.chat_area.insert(tk.END, f"{message}\n")
    
    
    def __init__(self, root):
        self.root = root
        self.root.title("Ollama LLM Chat GUI")
        import sys
        if getattr(sys, 'frozen', False):
            BASE_DIR = os.path.dirname(sys.executable)
        else:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DIR = BASE_DIR
        self.chat_logs_dir = os.path.join(BASE_DIR, "chat_logs")
        os.makedirs(self.chat_logs_dir, exist_ok=True)
        self.current_chat_file = None
        self.chat_thumbnails = {}
        # Load preferences if available
        self.preferences_path = os.path.join(BASE_DIR, "preferences.json")
        self.preferences = None
        if os.path.exists(self.preferences_path):
            try:
                with open(self.preferences_path, "r", encoding="utf-8") as f:
                    self.preferences = json.load(f)
            except Exception:
                self.preferences = None
        self.models = self.get_all_models()
        if not self.models:
            messagebox.showerror("Model Error", "No models found in Ollama.")
            self.root.destroy()
            return
        # Set defaults if not loaded
        if not self.preferences:
            self.preferences = {
                'rag_top_k': 2,
                'rag_similarity_threshold': 0.05,
                'rag_chunk_size': 400,
                'rag_chunk_overlap': 50,
                'auto_save_chat': True,
                'default_model': self.models[0],
                'font_size': 11,
                'dark_mode': False,
                'temperature': 0.7,
                'top_p': 1.0,
                'max_tokens': 2048,
            }
        # Ensure default_model is valid
        if self.preferences.get('default_model') not in self.models:
            self.preferences['default_model'] = self.models[0]
        self.model = self.preferences.get('default_model', self.models[0])
        self.stop_response = False
        self.citation_tag_counter = 0
        self.rag_user_prompt_template = (
            "You are an expert assistant. Use ONLY the following context to answer the question. "
            "If the answer is not in the context, say you don't know. Do NOT use prior knowledge.\n\n"
            "Context:\n{context}\n\nChat History:\n{history}\n\nQuestion: {question}\n\nAnswer:"
        )

        self.create_widgets()
        self.create_menu()
        self.apply_preferences()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def stop_responding(self):
        self.stop_response = True
    # Only keep this version of create_widgets

    def create_widgets(self):
        self.chain_of_thought_hidden = False
        # Dark mode colors
        prefs = getattr(self, 'preferences', {})
        dark_mode = prefs.get('dark_mode', False)
        bg = '#23272e' if dark_mode else None
        fg = '#e6e6e6' if dark_mode else None
        entry_bg = '#2d323b' if dark_mode else None

        # Add Load Knowledge Base button and model dropdown at the top
        self.top_frame = tk.Frame(self.root, bg=bg)
        self.top_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=(10,0))

        self.kb_button = tk.Button(self.top_frame, text="Load Knowledge Base", command=self.load_knowledge_base, bg=bg, fg=fg, activebackground='#444' if dark_mode else None, activeforeground=fg if dark_mode else None)
        self.kb_button.grid(row=0, column=0, padx=(0,10))

        # Replace OptionMenu with ttk.Combobox for model selection
        self.model_var = tk.StringVar(value=self.model)
        self.model_dropdown = ttk.Combobox(self.top_frame, textvariable=self.model_var, values=self.models, state="readonly")
        self.model_dropdown.bind("<<ComboboxSelected>>", lambda event: self.on_model_select(self.model_var.get()))
        self.model_dropdown.grid(row=0, column=1, padx=(0,10))


        # Add Edit Prompt button (always enabled)
        self.edit_prompt_button = tk.Button(self.top_frame, text="Edit Prompt", command=self.edit_prompt, bg=bg, fg=fg, activebackground='#444' if dark_mode else None, activeforeground=fg if dark_mode else None)
        self.edit_prompt_button.grid(row=0, column=2, padx=(0,10))

        # Add Tools button (replaces About button)
        self.tools_button = tk.Button(self.top_frame, text="Tools", command=self.show_tools_popup, bg=bg, fg=fg, activebackground='#444' if dark_mode else None, activeforeground=fg if dark_mode else None)
        self.tools_button.grid(row=0, column=3, padx=(0,10))

        # Add Hide Chain of Thought checkbox
        self.cot_var = tk.BooleanVar(value=False)
        self.cot_checkbox = tk.Checkbutton(self.top_frame, text="Hide Chain of Thought", variable=self.cot_var, command=self.toggle_chain_of_thought, bg=bg, fg=fg, activebackground='#444' if dark_mode else None, activeforeground=fg if dark_mode else None, selectcolor=bg)
        self.cot_checkbox.grid(row=0, column=4, padx=(0,10))

        # Chat area
        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled', width=60, height=20, bg=entry_bg if dark_mode else None, fg=fg if dark_mode else None, insertbackground=fg if dark_mode else None)
        self.chat_area.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        # Set the internal frame of scrolledtext to dark if possible
        try:
            self.chat_area.config(bg=entry_bg if dark_mode else None, fg=fg if dark_mode else None, insertbackground=fg if dark_mode else None)
        except Exception:
            pass

        # Entry and send button at the bottom
        self.entry_frame = tk.Frame(self.root, bg=bg)
        self.entry_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=(0,10))
        self.entry = tk.Entry(self.entry_frame, width=50, bg=entry_bg if dark_mode else None, fg=fg if dark_mode else None, insertbackground=fg if dark_mode else None)
        self.entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.entry.bind('<Return>', lambda event: self.send_message())
        # Bind Ctrl+V for image paste
        self.entry.bind('<Control-v>', self.on_paste_image)
        self.entry.bind('<Command-v>', self.on_paste_image)  # For Mac
        # Drag-and-drop support removed
        self.send_button = tk.Button(self.entry_frame, text="Send", command=self.send_message, bg=bg, fg=fg, activebackground='#444' if dark_mode else None, activeforeground=fg if dark_mode else None)
        self.send_button.pack(side=tk.LEFT, padx=10)

        # Stop Responding and Attach Image buttons at the very bottom
        self.stop_frame = tk.Frame(self.root, bg=bg)
        self.stop_frame.grid(row=3, column=0, sticky='ew', padx=10, pady=(0,10))
        # Attach Image button (left of Stop Responding)
        self.attach_button = tk.Button(self.stop_frame, text="Attach Image", command=self.attach_image_from_file, bg=bg, fg=fg, activebackground='#444' if dark_mode else None, activeforeground=fg if dark_mode else None)
        self.attach_button.pack(side=tk.LEFT, padx=(0, 10))
        self.stop_button = tk.Button(self.stop_frame, text="Stop Responding", command=self.stop_responding, bg=bg, fg=fg, activebackground='#444' if dark_mode else None, activeforeground=fg if dark_mode else None)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Ensure window and chat area resize properly
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    def attach_image_from_file(self):
        if not self.model_supports_images():
            messagebox.showwarning("Model Does Not Support Images", "The selected model does not support image input. Please select a vision/multimodal model.")
            return
        from tkinter import filedialog
        from PIL import Image, ImageTk
        import io
        filetypes = [
            ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        if not filepath:
            return
        try:
            img = Image.open(filepath)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            # Clear any previous pending image/thumbnail
            self.clear_pending_image()
            # Prepare thumbnail
            tk_img = ImageTk.PhotoImage(img.resize((64, 64)))
            # Store only the most recent pending image and thumbnail
            self.pending_image = buf.getvalue()
            self.pending_thumbnail = tk_img  # Prevent garbage collection
            self.show_image_preview()
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load image: {e}")

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

        # About menu (moved here)
        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="About", menu=about_menu)

        self.root.config(menu=menubar)
    def show_tools_popup(self):
        import tkinter as tk
        from tkinter import messagebox
        try:
            from tools import get_tool
            import tools as tools_mod
            tool_names = list(getattr(tools_mod, 'TOOLS', {}).keys())
            tool_funcs = [getattr(tools_mod, 'TOOLS', {}).get(name) for name in tool_names]
            tool_descs = []
            for name, fn in zip(tool_names, tool_funcs):
                # Compose a quick description for each tool
                if fn and fn.__doc__:
                    desc = fn.__doc__.strip()
                else:
                    # Fallback: use a default description for known tools
                    if name == "summarise":
                        desc = "Summarize the entire loaded knowledge base and present the main points."
                    else:
                        desc = "No description available."
                tool_descs.append((name, desc))
        except Exception:
            messagebox.showerror("Tools Error", "Could not load tools list.")
            return
        popup = tk.Toplevel(self.root)
        popup.title("Available Tools")
        popup.geometry("440x320")
        from gui_utils import set_dark_mode_popup
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)
        label = tk.Label(frame, text="Available Tools:", font=("Arial", 12, "bold"))
        label.pack(anchor='w', pady=(0,8))
        # Use a Text widget for better formatting and dark mode support
        text = tk.Text(frame, wrap=tk.WORD, width=54, height=12, state='normal', borderwidth=0, highlightthickness=0)
        for name, desc in tool_descs:
            text.insert(tk.END, f"/{name}", ("toolname",))
            text.insert(tk.END, f": {desc}\n\n")
        text.config(state='disabled')
        text.pack(fill=tk.BOTH, expand=True)
        close_btn = tk.Button(frame, text="Close", command=popup.destroy)
        close_btn.pack(pady=(10,0))
        # Apply dark mode styling if needed
        if self.preferences.get('dark_mode', False):
            set_dark_mode_popup(popup)
            # Style the Text widget for dark mode
            text.config(bg='#2d323b', fg='#e6e6e6', insertbackground='#e6e6e6', highlightbackground='#444a52', highlightcolor='#444a52')
            text.tag_configure("toolname", foreground="#7ecfff", font=("Arial", 10, "bold"))

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
        from gui_utils import set_dark_mode_popup
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
        option_menu = tk.OptionMenu(frame, default_model_var, *self.models)
        option_menu.grid(row=11, column=1, sticky='ew')
        # --- DARK MODE for OptionMenu and its menu ---
        if self.preferences.get('dark_mode', False):
            bg = '#23272e'
            fg = '#e6e6e6'
            border = '#444a52'
            option_menu.config(bg=bg, fg=fg, activebackground='#444', activeforeground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1, borderwidth=1)
            # Set menu colors
            menu = option_menu['menu']
            menu.config(bg=bg, fg=fg, activebackground='#444', activeforeground=fg, borderwidth=1)

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
                # Fix: get() on IntVar/DoubleVar/BooleanVar returns the correct type, so no need to cast again
                prefs['dark_mode'] = dark_mode_var.get() if isinstance(dark_mode_var.get(), bool) else bool(int(dark_mode_var.get()))
                prefs['temperature'] = float(temp_var.get())
                prefs['top_p'] = float(topp_var.get())
                prefs['max_tokens'] = int(max_tokens_var.get())
                # Save preferences to file
                try:
                    with open(self.preferences_path, "w", encoding="utf-8") as f:
                        json.dump(prefs, f, indent=2)
                except Exception:
                    messagebox.showerror("Save Error", "Could not save preferences to file.")
                self.apply_preferences()
                popup.destroy()
            except Exception as e:
                messagebox.showerror("Invalid Input", f"Please enter valid values for all preferences.\nError: {e}")
        save_btn = tk.Button(btn_frame, text="Save", command=save_prefs)
        save_btn.pack(side=tk.RIGHT, padx=(0,8))
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=popup.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=(0,8))
        frame.grid_columnconfigure(1, weight=1)

        # Now that ALL widgets are created, apply dark mode recursively
        if self.preferences.get('dark_mode', False):
            set_dark_mode_popup(popup)
            # Explicitly style btn_frame for border
            btn_frame.config(bg='#23272e', highlightbackground='#444a52', highlightcolor='#444a52', highlightthickness=1)

    def apply_preferences(self):
        # Apply font size and dark mode immediately
        prefs = getattr(self, 'preferences', None)
        if not prefs:
            return
        font = ("Arial", prefs.get('font_size', 11))
        widgets = [self.chat_area, self.entry]
        # Top frame widgets
        for w in [self.kb_button, self.model_dropdown, self.edit_prompt_button, self.tools_button, self.cot_checkbox]:
            try:
                w.config(font=font)
            except Exception:
                pass
        for w in widgets:
            try:
                w.config(font=font)
            except Exception:
                pass

        # --- DARK MODE STYLING ---
        if prefs.get('dark_mode', False):
            bg = '#23272e'
            fg = '#e6e6e6'
            entry_bg = '#2d323b'
            border = '#444a52'
            self.root.configure(bg=bg)
            # Chat area and entry
            for widget in [self.chat_area, self.entry]:
                widget.config(bg=entry_bg, fg=fg, insertbackground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
            # Top frame and entry frame borders
            for frame in [self.top_frame, self.entry_frame, self.stop_frame]:
                frame.config(bg=bg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
            # Buttons
            for widget in [self.kb_button, self.edit_prompt_button, self.tools_button, self.send_button, self.stop_button]:
                widget.config(bg=bg, fg=fg, activebackground='#444', activeforeground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1, borderwidth=1)
            # Checkbox
            self.cot_checkbox.config(bg=bg, fg=fg, activebackground='#444', activeforeground=fg, selectcolor=bg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
            # --- TTK DARK MODE STYLE ---
            style = ttk.Style()
            try:
                style.theme_use('clam')
            except Exception:
                pass
            style.configure('TCombobox', fieldbackground=entry_bg, background=entry_bg, foreground=fg, bordercolor=border, lightcolor=border, darkcolor=border, borderwidth=1)
            style.map('TCombobox', fieldbackground=[('readonly', entry_bg)], background=[('readonly', entry_bg)], foreground=[('readonly', fg)])
            self.model_dropdown.configure(style='TCombobox')
            # ---
        else:
            # Reset to default
            self.root.configure(bg=None)
            for widget in [self.chat_area, self.entry]:
                widget.config(bg='white', fg='black', insertbackground='black', highlightbackground=None, highlightcolor=None, highlightthickness=0)
            for frame in [self.top_frame, self.entry_frame, self.stop_frame]:
                frame.config(bg=None, highlightbackground=None, highlightcolor=None, highlightthickness=0)
        for widget in [self.kb_button, self.edit_prompt_button, self.tools_button, self.send_button, self.stop_button]:
            widget.config(bg=None, fg=None, activebackground=None, activeforeground=None, highlightbackground=None, highlightcolor=None, highlightthickness=0, borderwidth=1)
        self.cot_checkbox.config(bg=None, fg=None, activebackground=None, activeforeground=None, selectcolor=None, highlightbackground=None, highlightcolor=None, highlightthickness=0)
        style = ttk.Style()
        style.theme_use('default')
        self.model_dropdown.configure(style='TCombobox')
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
        from gui_utils import set_dark_mode_popup
        dark_mode = self.preferences.get('dark_mode', False)
        if dark_mode:
            set_dark_mode_popup(popup)
        border = '#444a52' if dark_mode else None
        bg = '#23272e' if dark_mode else None
        fg = '#e6e6e6' if dark_mode else None
        entry_bg = '#2d323b' if dark_mode else None
        frame = tk.Frame(popup, bg=bg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        listbox = tk.Listbox(frame, width=50, height=10, bg=entry_bg if dark_mode else None, fg=fg if dark_mode else None, highlightbackground=border, highlightcolor=border, selectbackground='#444' if dark_mode else None, selectforeground=fg if dark_mode else None)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for fname in files:
            listbox.insert(tk.END, fname)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=listbox.yview, bg=bg, troughcolor=bg, activebackground='#444' if dark_mode else None, highlightbackground=border)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=scrollbar.set)

        btn_frame = tk.Frame(popup, bg=bg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
        btn_frame.pack(fill=tk.X, pady=(10,0))
        open_btn = tk.Button(btn_frame, text="Open", command=lambda: self.open_selected_chat(listbox, files, popup), bg=bg, fg=fg, activebackground='#444' if dark_mode else None, activeforeground=fg if dark_mode else None, highlightbackground=border, highlightcolor=border, highlightthickness=1, borderwidth=1)
        open_btn.pack(side=tk.LEFT, padx=5)
        del_btn = tk.Button(btn_frame, text="Delete", command=lambda: self.delete_selected_chat(listbox, files, popup), bg=bg, fg=fg, activebackground='#444' if dark_mode else None, activeforeground=fg if dark_mode else None, highlightbackground=border, highlightcolor=border, highlightthickness=1, borderwidth=1)
        del_btn.pack(side=tk.LEFT, padx=5)
        close_btn = tk.Button(btn_frame, text="Close", command=popup.destroy, bg=bg, fg=fg, activebackground='#444' if dark_mode else None, activeforeground=fg if dark_mode else None, highlightbackground=border, highlightcolor=border, highlightthickness=1, borderwidth=1)
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
        from gui_utils import set_dark_mode_popup, show_about_popup
        popup = tk.Toplevel(self.root)
        popup.withdraw()
        show_about_popup(popup)
        if self.preferences.get('dark_mode', False):
            set_dark_mode_popup(popup)
        popup.deiconify()

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

        # No longer needed: Edit Prompt button is always enabled

    def edit_prompt(self):
        # Popup to edit both the System and RAG prompt templates
        popup = tk.Toplevel(self.root)
        popup.title("Edit Prompts")
        popup.geometry("700x500")
        from gui_utils import set_dark_mode_popup
        if self.preferences.get('dark_mode', False):
            set_dark_mode_popup(popup)
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # System prompt
        sys_label = tk.Label(frame, text="System Prompt (used for normal chat):", font=("Arial", 11, "bold"), anchor='w', justify='left')
        sys_label.pack(anchor='w', pady=(0,4))
        sys_text = tk.Text(frame, wrap=tk.WORD, width=80, height=6)
        sys_text.pack(fill=tk.BOTH, expand=False)
        sys_prompt = getattr(self, 'system_prompt_template', "You are an expert assistant. You have access to the following chat history:\n\n{history}\n\n{user_message}")
        sys_text.insert(tk.END, sys_prompt)

        # RAG prompt
        rag_label = tk.Label(frame, text="RAG Prompt (used when knowledge base is loaded):", font=("Arial", 11, "bold"), anchor='w', justify='left')
        rag_label.pack(anchor='w', pady=(16,4))
        rag_text = tk.Text(frame, wrap=tk.WORD, width=80, height=10)
        rag_text.pack(fill=tk.BOTH, expand=True)
        rag_text.insert(tk.END, self.rag_user_prompt_template)

        # Help text
        help_label = tk.Label(frame, text="RAG prompt must include {context}, {history}, and {question}. System prompt should include {history} and {user_message}.", font=("Arial", 9), fg="#888888", anchor='w', justify='left')
        help_label.pack(anchor='w', pady=(8,0))

        # Save/cancel buttons
        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(16,0), anchor='e')
        save_btn = tk.Button(btn_frame, text="Save")
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=popup.destroy)
        save_btn.pack(side=tk.RIGHT, padx=(0,8))
        cancel_btn.pack(side=tk.RIGHT, padx=(0,8))

        def save_and_close():
            new_sys = sys_text.get("1.0", tk.END).strip()
            new_rag = rag_text.get("1.0", tk.END).strip()
            # Validate RAG prompt
            if "{context}" not in new_rag or "{question}" not in new_rag or "{history}" not in new_rag:
                messagebox.showerror("Template Error", "RAG prompt must include {context}, {history}, and {question}.")
                return
            # Validate system prompt
            if "{history}" not in new_sys or "{user_message}" not in new_sys:
                messagebox.showerror("Template Error", "System prompt should include {history} and {user_message}.")
                return
            self.rag_user_prompt_template = new_rag
            self.system_prompt_template = new_sys
            popup.destroy()
        save_btn.config(command=save_and_close)

        # --- DARK MODE for widgets ---
        if self.preferences.get('dark_mode', False):
            bg = '#23272e'
            fg = '#e6e6e6'
            entry_bg = '#2d323b'
            border = '#444a52'
            frame.config(bg=bg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
            sys_label.config(bg=bg, fg=fg)
            sys_text.config(bg=entry_bg, fg=fg, insertbackground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
            rag_label.config(bg=bg, fg=fg)
            rag_text.config(bg=entry_bg, fg=fg, insertbackground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1)
            help_label.config(bg=bg, fg="#bbbbbb")
            save_btn.config(bg=bg, fg=fg, activebackground='#444', activeforeground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1, borderwidth=1)
            cancel_btn.config(bg=bg, fg=fg, activebackground='#444', activeforeground=fg, highlightbackground=border, highlightcolor=border, highlightthickness=1, borderwidth=1)
            btn_frame.config(bg=bg, highlightbackground=border, highlightcolor=border, highlightthickness=1)


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
        # If there is a pending image, check if model supports images
        if hasattr(self, 'pending_image') and self.pending_image:
            if not self.model_supports_images():
                messagebox.showwarning("Model Does Not Support Images", "The selected model does not support image input. Please select a vision/multimodal model.")
                # Remove preview if present
                if hasattr(self, 'image_preview_index'):
                    self.chat_area.config(state='normal')
                    self.chat_area.delete(self.image_preview_index, f"{self.image_preview_index} lineend+1c")
                    del self.image_preview_index
                    self.chat_area.config(state='disabled')
                self.clear_pending_image()
                return
        import re
        user_message = self.entry.get().strip()
        # If both entry and pending image are empty, do nothing
        if not user_message and not hasattr(self, 'pending_image'):
            return
        # Tool command support: allow /toolname anywhere in the message
        tool_match = re.search(r"/(\w+)", user_message)
        tool_fn = None
        command = None
        if tool_match:
            command = tool_match.group(1)
            try:
                from tools import get_tool
                tool_fn = get_tool(command)
            except Exception:
                tool_fn = None
        if tool_fn is not None:
            # Remove preview if present
            if hasattr(self, 'image_preview_index'):
                self.chat_area.config(state='normal')
                self.chat_area.delete(self.image_preview_index, f"{self.image_preview_index} lineend+1c")
                del self.image_preview_index
                self.chat_area.config(state='disabled')
            # Call the tool function: (new_user_message, context_override, tool_response)
            new_user_message, context_override, tool_response = tool_fn(self, user_message)
            # Show the original user message in chat
            self.append_chat("You", user_message)
            self.entry.delete(0, tk.END)
            self.stop_response = False
            # If tool_response is not None, show it in chat (e.g., for immediate feedback)
            if tool_response:
                self.append_chat("Tool", tool_response)
            # Start LLM response with context_override (force context_override to be used!)
            threading.Thread(target=self.get_llm_response, args=(new_user_message, context_override), daemon=True).start()
            return
        # If there is a pending image, send it with the message
        if hasattr(self, 'pending_image') and self.pending_image:
            # Remove preview if present
            if hasattr(self, 'image_preview_index'):
                self.chat_area.config(state='normal')
                self.chat_area.delete(self.image_preview_index, f"{self.image_preview_index} lineend+1c")
                del self.image_preview_index
                self.chat_area.config(state='disabled')
            self.append_chat("You", "[Image sent] " + (user_message or ""), image_thumbnail=self.pending_thumbnail)
            self.entry.delete(0, tk.END)
            self.stop_response = False
            image_to_send = self.pending_image
            threading.Thread(target=self.send_image_to_model, args=(image_to_send, user_message), daemon=True).start()
            self.clear_pending_image()
            return
        # If not a valid tool and no image, just treat as normal message
        self.append_chat("You", user_message)
        self.entry.delete(0, tk.END)
        self.stop_response = False
        threading.Thread(target=self.get_llm_response, args=(user_message,), daemon=True).start()
    def send_images_to_model(self, images_bytes_list, user_message="", ext="png"):
        import base64
        import requests
        img_b64_list = [base64.b64encode(img_bytes).decode("utf-8") for img_bytes in images_bytes_list]
        prefs = getattr(self, 'preferences', {})
        payload = {
            "model": self.model,
            "prompt": user_message or "Describe these images.",
            "images": img_b64_list,
            "options": {
                "temperature": prefs.get('temperature', 0.7),
                "top_p": prefs.get('top_p', 1.0),
                "num_predict": prefs.get('max_tokens', 2048)
            }
        }
        try:
            resp = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload, stream=True)
            resp.raise_for_status()
            import json as _json
            self.append_chat(self.model, "", streaming=True)
            first_chunk = True
            for line in resp.iter_lines():
                if self.stop_response:
                    break
                if line:
                    data = _json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        self.update_last_agent_message_stream(chunk, append=(not first_chunk))
                        first_chunk = False
            if first_chunk:
                self.update_last_agent_message_stream("[No response]", append=False)
        except Exception as e:
            self.append_chat(self.model, f"[Error: {e}]")
        else:
            # If not a valid tool and no image, just treat as normal message
            self.append_chat("You", user_message)
            self.entry.delete(0, tk.END)
            self.stop_response = False
            threading.Thread(target=self.get_llm_response, args=(user_message,), daemon=True).start()

    def append_chat(self, sender, message, streaming=False, image_thumbnail=None):
        # Maintain chat history for hide/unhide chain of thought
        if not hasattr(self, 'chat_history'):
            self.chat_history = []
        if streaming and sender == self.model:
            self.chat_area.config(state='normal')
            # Do NOT add a new line after the model name for streaming
            self.chat_area.insert(tk.END, f"{sender}: ")
            self.last_agent_index = self.chat_area.index(tk.END)
            self.chat_area.config(state='disabled')
            self.chat_area.see(tk.END)
            # For streaming, add a placeholder to chat_history (will be updated in update_last_agent_message_stream)
            self.chat_history.append((sender, "", None))
        else:
            self.chat_area.config(state='normal')
            # Always start user/agent message on a new line if not at the start
            if self.chat_area.index(tk.END) != "1.0":
                self.chat_area.insert(tk.END, "\n")
            self.chat_area.insert(tk.END, f"{sender}: ")
            idx = len(self.chat_history)
            if image_thumbnail is not None:
                self.chat_area.image_create(tk.END, image=image_thumbnail)
                self.chat_area.insert(tk.END, " ")
                self.chat_thumbnails[idx] = image_thumbnail
            self.chat_area.insert(tk.END, f"{message}\n")
            self.last_agent_index = None
            self.chat_area.config(state='disabled')
            self.chat_area.see(tk.END)
            # Only store None for image_thumbnail in chat_history to avoid JSON serialization issues
            self.chat_history.append((sender, message, None))

    def get_llm_response(self, user_message, context_override=None):
        try:
            # RAG: retrieve context if KB loaded, unless context_override is provided
            if context_override is not None:
                # Use all KB chunks as context, and treat as one big chunk
                context = context_override
                context_chunks = []
                # For citations: show all chunks if available
                citation_indices = []
                citation_text = ""
                if hasattr(self, 'kb_chunks') and self.kb_chunks:
                    citation_indices = list(range(len(self.kb_chunks)))
                    citation_text = "\n\n[Citations: " + ", ".join([f"Chunk {i+1}" for i in citation_indices]) + "]"
            elif hasattr(self, 'kb_chunks') and hasattr(self, 'kb_embeddings'):
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
                citation_indices = []
                citation_text = ""
                if context_chunks:
                    citation_indices = [i for i, _ in context_chunks]
                    citation_text = "\n\n[Citations: " + ", ".join([f"Chunk {i+1}" for i in citation_indices]) + "]"
            else:
                context, context_chunks = "", []
                citation_indices = []
                citation_text = ""
            # RAG: Always send context and question as a single user message for best model compatibility
            # Format previous chat history (excluding current user message)
            history_lines = []
            for entry in self.chat_history[:-1]:  # Exclude the just-appended user message
                if len(entry) == 3:
                    sender, msg, _ = entry
                else:
                    sender, msg = entry
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
                if citation_text and citation_indices:
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
            show_citation_popup(self, chunk_indices)
            # Always move the cursor to the end and disable the widget after popup
            self.chat_area.mark_set(tk.INSERT, tk.END)
            self.chat_area.config(state='disabled')
            self.chat_area.see(tk.END)
        return callback

    def update_last_agent_message_stream(self, message, append=False):
        self.chat_area.config(state='normal')
        if hasattr(self, 'last_agent_index') and self.last_agent_index:
            if append:
                self.chat_area.insert(tk.END, message)
                # Update last message in chat_history
                if hasattr(self, 'chat_history') and self.chat_history:
                    entry = self.chat_history[-1]
                    if len(entry) == 3:
                        sender, prev_msg, image_thumbnail = entry
                        self.chat_history[-1] = (sender, prev_msg + message, image_thumbnail)
                    else:
                        sender, prev_msg = entry
                        self.chat_history[-1] = (sender, prev_msg + message)
            else:
                # First chunk: clear any previous content after label
                self.chat_area.delete(self.last_agent_index, tk.END)
                self.chat_area.insert(tk.END, message)
                # Set last message in chat_history
                if hasattr(self, 'chat_history') and self.chat_history:
                    entry = self.chat_history[-1]
                    if len(entry) == 3:
                        sender, _, image_thumbnail = entry
                        self.chat_history[-1] = (sender, message, image_thumbnail)
                    else:
                        sender, _ = entry
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
