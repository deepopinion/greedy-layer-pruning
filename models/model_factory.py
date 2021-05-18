
import torch
import transformers
from models.bert.modeling_bert import BertForSequenceClassification
from models.roberta.modeling_roberta import RobertaForSequenceClassification
from models.distilbert.modeling_distilbert import DistilBertForSequenceClassification
from transformers.models.mobilebert.modeling_mobilebert import MobileBertForSequenceClassification

def create_model(config, model_args):
    model_name = model_args.model_name_or_path
    if model_name.startswith("bert") or model_name == "huawei-noah/TinyBERT_General_6L_768D":
        return BertForSequenceClassification.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    elif model_name.startswith("roberta"):
        return RobertaForSequenceClassification.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    elif model_name.startswith("distilbert"):
        return DistilBertForSequenceClassification.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    elif model_name.startswith("google/mobilebert-uncased"):
        return MobileBertForSequenceClassification.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    raise Exception(f"Model {model_name} unknown.")