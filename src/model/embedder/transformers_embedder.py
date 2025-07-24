import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import logging

logger = logging.getLogger(__name__)

class TransformersEmbedder(nn.Module):
    """
    Encode the input with transformers model such as
    BERT, Roberta, XLM-RoBERTa, etc.
    """

    def __init__(self, transformer_model_name: str):
        super(TransformersEmbedder, self).__init__()
        output_hidden_states = False  # to use all hidden states or not
        logger.info(f"[Model Info] Loading pretrained language model {transformer_model_name}")
        
        # Load config first to check model type
        config = AutoConfig.from_pretrained(transformer_model_name)
        self.model_type = config.model_type
        
        # Initialize model with proper settings
        self.model = AutoModel.from_pretrained(
            transformer_model_name,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            add_pooling_layer=False  # We don't need the pooled output
        )
        
        # Disable token type IDs for models that don't use them
        if hasattr(self.model, 'embeddings'):
            if hasattr(self.model.embeddings, 'token_type_embeddings'):
                self.model.embeddings.token_type_embeddings = None

    def get_output_dim(self):
        return self.model.config.hidden_size

    def forward(self, subword_input_ids: torch.Tensor,
                orig_to_token_index: torch.LongTensor,
                attention_mask: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for the transformer model.
        
        Args:
            subword_input_ids: (batch_size x max_wordpiece_len) the input id tensor
            orig_to_token_index: (batch_size x max_sent_len) the mapping from original word id to subword token index
            attention_mask: (batch_size x max_wordpiece_len) attention mask for the input
            
        Returns:
            torch.Tensor: Word-level representations of shape (batch_size, max_sent_len, hidden_size)
        """
        # Prepare model inputs
        model_inputs = {
            "input_ids": subword_input_ids,
            "attention_mask": attention_mask,
        }
        
        # For models that don't use token type ids, we need to ensure they're not used
        if self.model_type in ['roberta', 'xlm-roberta', 'distilbert', 'camembert', 'xlm']:
            model_inputs["token_type_ids"] = None
        
        # Get subword representations
        outputs = self.model(**model_inputs)
        subword_rep = outputs.last_hidden_state
        
        # Get word-level representations by gathering the relevant subword tokens
        batch_size, _, rep_size = subword_rep.size()
        _, max_sent_len = orig_to_token_index.size()
        
        # Ensure orig_to_token_index is on the same device as subword_rep
        orig_to_token_index = orig_to_token_index.to(subword_rep.device)
        
        # Select word representations using gather
        word_rep = torch.gather(
            subword_rep, 
            1, 
            orig_to_token_index.unsqueeze(-1).expand(batch_size, max_sent_len, rep_size)
        )
        
        return word_rep