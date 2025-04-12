class HHRLHFPreprocessor(object):
    """
    HHRLHF Preprocessor

    Requires two columns "chosen" and "rejected" with the following format:
    \n\nHuman: {prompt}\n\nAssistant: {response}
    """

    def __init__(self, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def run(self, examples):
        new_examples: dict = {
            **examples,
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
            tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples
