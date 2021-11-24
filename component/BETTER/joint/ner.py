import logging
from typing import Any, Mapping, Sequence, Tuple, Union, List
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

class NERPrediction:
    """Class for NER Prediction."""

    def __init__(self, model, labels, model_device = None) -> None:
        """Initializes NER Class.

        Args:
            model: Model name
            model_device: model device
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model)
        self.model_device = model_device
        self.labels = {}

        for id in labels:
            self.labels[labels[id]] = id


        # self.postprocessor = TokenClassificationPostprocessor(
        #     tokenizer, aggregation_strategy="simple"
        # )
        self.logger = logging.getLogger(__name__)


    def analyze(
        self, query: List[str]
    ):
        """Query Analyze.

        Args:
            query: query
        """
        orig_to_tok_index = []
        tokens = []
        tokens.append(self.tokenizer.tokenize(self.tokenizer.cls_token)[0])
        for i, word in enumerate(query):
            orig_to_tok_index.append(len(tokens))
            word_tokens = self.tokenizer.tokenize(" " + word)
            for sub_token in word_tokens:
                tokens.append(sub_token)
        tokens.append(self.tokenizer.tokenize(self.tokenizer.sep_token)[0])

        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.int64).to(self.model_device)
        #original_input_ids = torch.tensor([input_ids[i] for i in orig_to_tok_index], dtype=torch.int64).to(self.model_device)
        input_mask = torch.tensor([1] * len(input_ids), dtype=torch.int64).to(self.model_device)

        outputs = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=input_mask.unsqueeze(0))
        predictions = torch.argmax(outputs[0], dim=2).detach().cpu()[0].tolist()
        original_predictions = [predictions[i] for i in orig_to_tok_index]
        prediction_labels = [self.model.config.id2label[x] for x in original_predictions]
        converted_id = [self.labels[x] for x in prediction_labels]

        return [converted_id]
