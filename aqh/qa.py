import numpy as np
import sys
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QLineEdit, QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt
from chunk import Chunker
from audio import AudioHandler
from numpy.linalg import norm
from openai import OpenAI

SYSTEM_PROMPT = """
You are a chat assistant that is taking a quiz. Given a question, context, and answer choices, accurately answer the question. 
Only return the text of the correct answer and nothing else. 
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT: 
{context}

ANSWER CHOICES: 
{choices}

Return the correct answer choice to the question given the context. Only return the text of the correct answer choice. 
"""

class ChatGUI(QMainWindow):
    def __init__(self, txt_filepath: str):
        super().__init__()
        self.setWindowTitle("Chat Interface")
        self.setGeometry(100, 100, 800, 600) 
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)  
        layout.setContentsMargins(20, 20, 20, 20) 
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
            }
        """)
        
        chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(chat_widget)
        self.chat_layout.setSpacing(10)  
        self.chat_layout.setAlignment(Qt.AlignTop) 
        self.chat_layout.addStretch()
        scroll_area.setWidget(chat_widget)
        layout.addWidget(scroll_area)
        
        self.text_entry = QLineEdit()
        self.text_entry.setPlaceholderText("Type your message here...")
        self.text_entry.returnPressed.connect(self.send_message)
        self.text_entry.setFixedHeight(50) 
        self.text_entry.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                margin: 10px;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.text_entry)

        # RAG 
        self.model_name = "text-embedding-ada-002"
        self.client = OpenAI()
        chunker = Chunker(filepath=txt_filepath, model_name=self.model_name)
        self.embeddings = chunker.process()
        
    def create_message_widget(self, text: str, is_user: bool = True):
        message = QTextEdit()
        message.setReadOnly(True)
        message.setPlainText(text)
        message.setMinimumWidth(200)  
        message.setMaximumWidth(600) 
        message.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 10px;
                margin: 10px;
                background-color: {'#e3f2fd' if is_user else '#f8f9fa'};
                font-size: 14px;
                color: #333;  # Add text color
            }}
        """)
        
        # Adjust height to content
        doc_height = int(message.document().size().height())
        message.setFixedHeight(min(doc_height + 60, 300))
        return message

    def send_message(self):
        text = self.text_entry.text().strip()
        answer = self._get_answer(text)
        if text:
            resp = f"{text.split("\n")[0]}\nANSWER: {answer}"
            user_message = self.create_message_widget(resp, is_user=True)
            self.chat_layout.addWidget(user_message)
            
            self.text_entry.clear()
            
            QApplication.processEvents()
            scroll_area = self.findChild(QScrollArea)
            vsb = scroll_area.verticalScrollBar()
            vsb.setValue(vsb.maximum())
    
    def _get_answer(self, text: str) -> str: 
        question = text.split("\n")[0]
        question_embedding = self.client.embeddings.create(
            input=question,
            model=self.model_name
        ).data[0].embedding
        context = self._get_context(question_embedding)
        prompt = PROMPT_TEMPLATE.format(question=question, 
                                        context="\n\n".join(c for c in context), 
                                        choices="\n".join(a for a in text.split("\n")[2:]))
        answer = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT}, 
                {"role": "user", "content": prompt}
            ]
        )
        return answer.choices[0].message.content

    def _get_context(self, question_embd, top_k: int=5): 
        chunks = [(self._sim(question_embd, e["embedding"]), e["chunk"]) for e in self.embeddings] 
        chunks.sort(key=lambda x: x[0])
        return [c[1] for c in chunks[-top_k:]]
    
    def _sim(self, a, b): 
        return np.dot(a, b) / (norm(a) * norm(b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a valid path to a text file (transcript) or mp3 file (audio)")
    parser.add_argument("filepath", help="Valid file path to .mp3 or .txt")
    args = parser.parse_args()

    fp = args.filepath 
    assert fp[-3:] in ["mp3", "txt"], "Please upload either a mp3 or txt file."

    if fp[-3:] == ".mp3": 
        ah = AudioHandler(fp)
        fp = ah.process_audio()

    app = QApplication(sys.argv)
    window = ChatGUI(txt_filepath=fp)
    window.show()
    sys.exit(app.exec_())

    