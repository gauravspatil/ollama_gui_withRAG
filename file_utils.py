import os
import json
from tkinter import filedialog, messagebox

def load_knowledge_base_files():
    kb_texts = []
    file_paths = filedialog.askopenfilenames(
        title="Select Knowledge Base Files",
        filetypes=[
            ("Text, PDF, or Word Files", "*.txt;*.pdf;*.docx"),
            ("All Files", "*.*")
        ]
    )
    if not file_paths:
        return None
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            try:
                import PyPDF2
            except ImportError:
                messagebox.showerror("Missing Dependency", "Please install PyPDF2: pip install PyPDF2")
                return None
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    kb_texts.append("\n".join(page.extract_text() or '' for page in reader.pages))
            except Exception as e:
                messagebox.showerror("PDF Error", f"Could not read PDF: {e}")
                return None
        elif ext == ".docx":
            try:
                import docx
            except ImportError:
                messagebox.showerror("Missing Dependency", "Please install python-docx: pip install python-docx")
                return None
            try:
                doc = docx.Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                kb_texts.append(text)
            except Exception as e:
                messagebox.showerror("Word Error", f"Could not read Word document: {e}")
                return None
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    kb_texts.append(f.read())
            except Exception as e:
                messagebox.showerror("File Error", f"Could not read file {file_path}: {e}")
                return None
    return kb_texts

def save_chat_history(chat_logs_dir, chat_history, current_chat_file=None):
    import datetime
    if current_chat_file:
        fpath = current_chat_file
    else:
        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fpath = os.path.join(chat_logs_dir, f"chat_{dt}.json")
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump({"chat_history": chat_history}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        messagebox.showerror("Save Error", f"Could not save chat log:\n{e}")
    return fpath

def load_chat_history(fpath):
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("chat_history", [])
    except Exception as e:
        messagebox.showerror("Open Error", f"Could not open chat log:\n{e}")
        return None

def list_chat_logs(chat_logs_dir):
    files = [f for f in os.listdir(chat_logs_dir) if f.endswith(".json")]
    files.sort(reverse=True)
    return files

def delete_chat_log(fpath):
    try:
        os.remove(fpath)
        return True
    except Exception as e:
        messagebox.showerror("Delete Error", f"Could not delete chat log:\n{e}")
        return False
