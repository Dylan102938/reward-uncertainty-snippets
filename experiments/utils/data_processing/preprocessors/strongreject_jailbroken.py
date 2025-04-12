from experiments.utils.data_processing.preprocessors.hh_rlhf import HHRLHFPreprocessor


class SRJailbrokenPreprocessor(HHRLHFPreprocessor):
    """
    StrongReject Jailbroken Preprocessor

    Jailbroken prompts are formatted as follows: {jailbroken_prompt}
    Control responses are formatted as follows: {control_response}
    Treatment responses are formatted as follows: {treatment_response}
    """

    def run(self, examples):
        new_examples: dict = {"chosen": [], "rejected": []}

        jailbroken_prompt = examples["treatment_raw_jailbroken_prompt"]
        control_response = examples["control_raw_response"]
        treatment_response = examples["treatment_raw_response"]

        for prompt, chosen, rejected in zip(jailbroken_prompt, control_response, treatment_response):
            new_examples["chosen"].append(f"\n\nHuman: {prompt}\n\nAssistant: {chosen}")
            new_examples["rejected"].append(f"\n\nHuman: {prompt}\n\nAssistant: {rejected}")

        return super().run(new_examples)
