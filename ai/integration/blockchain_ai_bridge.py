from ai.pipeline import run_ai


class AIBridge:

    def __init__(self, blockchain):
        self.blockchain = blockchain

    def decide(self):

        output = run_ai(self.blockchain)

        if isinstance(output, tuple):
            # Policy network returns (mean, std); use mean as action signal.
            output = output[0]

        return output.detach().numpy()