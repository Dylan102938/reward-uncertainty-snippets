from experiments.utils.data_processing.preprocessors.hh_rlhf import HHRLHFPreprocessor


class HHRLHFJailBrokenPreprocessor(HHRLHFPreprocessor):
    """
    HHRLHF Jailbroken Preprocessor

    Jailbroken prompts are formatted as follows: Human: {prompt}
    Responses are formatted as follows: {responses}
    """

    def run(self, examples):
        new_examples: dict = {"chosen": [], "rejected": []}

        for prompt, responses in zip(examples["prompt"], examples["responses"]):
            human_text = (prompt.split("\n\n"))[0]
            new_examples["chosen"].append(f"\n\n{human_text}\n\nAssistant: {responses[0]}")
            new_examples["rejected"].append(f"\n\n{human_text}\n\nAssistant: {responses[1]}")

        return super().run(new_examples)
