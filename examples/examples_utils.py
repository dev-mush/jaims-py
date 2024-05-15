import os
import json
import time
from jaims.adapters.openai_adapter import OpenAITransactionStorageInterface


class FileTransactionStorage(OpenAITransactionStorageInterface):

    def __init__(self, path="storage") -> None:
        super().__init__()
        script_dir = os.path.dirname(__file__)
        self.storage_path = os.path.join(script_dir, path)
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def store_transaction(self, request: dict, response: dict):

        transaction = {"request": request, "response": response}
        unix_timestamp = str(int(time.time()))

        with open(f"{self.storage_path}/{unix_timestamp}.json", "w") as f:
            f.write(json.dumps(transaction, indent=4))
