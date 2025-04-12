from experiments.utils.data_processing.preprocessors.hh_rlhf import HHRLHFPreprocessor


class SafeRLHFPreprocessor(HHRLHFPreprocessor):
    """
    SafeRLHF Preprocessor

    SafeRLHF prompts are formatted as follows: {prompt}
    SafeRLHF chosen responses are formatted as follows: {chosen_response}
    SafeRLHF rejected responses are formatted as follows: {rejected_response}
    """

    def run(self, examples):
        new_examples = {
            "chosen": [],
            "rejected": [],
        }

        for prompt, chosen_resp, rejected_resp in zip(
            examples["prompt"], examples["chosen_response"], examples["rejected_response"]
        ):
            chosen = f"\n\nHuman: {prompt}\n\nAssistant: {chosen_resp}"
            rejected = f"\n\nHuman: {prompt}\n\nAssistant: {rejected_resp}"

            new_examples["chosen"].append(chosen)
            new_examples["rejected"].append(rejected)

        return super().run(new_examples)
