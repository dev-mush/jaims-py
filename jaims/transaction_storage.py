class JAImsTransactionStorageInterface(object):
    """
    Interface for storing JAIms transactions.
    Override this class to implement your own storage, will store a pair of openai request and response.
    """

    def store_transaction(self, request: dict, response: dict):
        pass
