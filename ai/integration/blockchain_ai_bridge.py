from ai.test_ai_pipeline import run_ai


class AIBridge:

    def __init__(self, blockchain):
        self.blockchain = blockchain

    def decide(self):

        output = run_ai(self.blockchain)

        return output.detach().numpy()