from datetime import datetime
import io
import time
from PIL import Image
from typing import Iterable

from openai import AzureOpenAI
from openai.types.beta.threads import Message, TextContentBlock, ImageFileContentBlock
from openai.types.beta.assistant import Assistant
from openai.types.beta.thread import Thread


class Common:
    def __init__(self: "Common", client: AzureOpenAI, assistant: Assistant, thread: Thread) -> None:
        self.client = client
        self.assistant = assistant
        self.thread = thread

    def format_messages(self: "Common", messages: Iterable[Message]) -> None:
        message_list = []

        # Collect messages until the last user message
        for message in messages:
            message_list.append(message)
            if message.role == "user":
                break

        # Reverse the list to display the last user message first
        for message in reversed(message_list):
            for item in message.content:
                if isinstance(item, TextContentBlock):
                    print(f"{message.role}:\n{item.text.value}\n")
                elif isinstance(item, ImageFileContentBlock):
                    # Retrieve and display image
                    response_content = self.client.files.content(item.image_file.file_id)
                    image = Image.open(io.BytesIO(response_content.read()))
                    image = image.resize((image.width // 2, image.height // 2), Image.LANCZOS)
                    image.show()

    def process_message(self: "Common", content: str) -> None:
        self.client.beta.threads.messages.create(thread_id=self.thread.id, role="user", content=content)
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=(
                "Please address the user as Jane Doe. The user has a premium account. "
                "Be assertive, accurate, and polite. Ask if the user has further questions. "
                f"The current date and time is: {datetime.now().strftime('%x %X')}."
            ),
        )
        print("processing ...")

        while True:
            run = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run.id)

            if run.status in {"completed", "failed", "expired", "cancelled"}:
                if run.status == "completed":
                    self.format_messages(self.client.beta.threads.messages.list(thread_id=self.thread.id))
                elif run.status == "failed":
                    messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
                    answer = messages.data[0].content[0].text.value
                    print(f"Failed User:\n{content}\nAssistant:\n{answer}\n")
                else:
                    print(run)
                break

            if run.status == "requires_action":
                # Handle function calling and continue processing
                pass

            time.sleep(5)
