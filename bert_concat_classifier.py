# bert_concat_classifier.py
import torch
import torch.nn as nn
from transformers import AutoModel, SequenceClassifierOutput

class BertConcatClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertConcatClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name, output_hidden_states=True)
        self.num_labels = num_labels
        self.conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        last_cls_vector = hidden_states[-1][:, 0, :]
        fourth_last_cls_vector = hidden_states[-4][:, 0, :]

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states[-1])
        masked_last_layer = hidden_states[-1][:, 1:, :] * attention_mask_expanded[:, 1:, :]
        mean_pooled_vector = masked_last_layer.mean(dim=1)

        concatenated_vector = torch.cat((last_cls_vector.unsqueeze(1), fourth_last_cls_vector.unsqueeze(1), mean_pooled_vector.unsqueeze(1)), dim=1)
        conv_output = self.conv(concatenated_vector).squeeze(2)
        conv_output = self.relu(conv_output)
        logits = self.linear(conv_output)
        logits = logits.squeeze(1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
