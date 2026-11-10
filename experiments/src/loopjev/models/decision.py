from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from loopjev.io import digest
from loopjev.paths import ROOT
from loopjev.training.tasks import class_logits
from loopjev.training.batching import candidate_batches


def select_last_states(states, attention_mask):
    # The caller keeps the mask on the CPU, so validation does not synchronize CUDA.
    if attention_mask.ndim != 2 or (attention_mask.sum(-1) < 1).any():
        raise ValueError('Every right-padded input needs a non-padding token')
    device = states[0].device
    positions = (attention_mask.sum(-1) - 1).to(device)
    members = torch.arange(len(attention_mask), device=device)
    return torch.stack([hidden[members, positions].float() for hidden in states], dim=1)


class DecisionModel(torch.nn.Module):
    def __init__(self, config, lock, checkpoint=None):
        super().__init__()
        from loopjev.input_protocol import validate_config, validate_lock, validate_checkpoint_manifest
        validate_config(config)
        validate_lock(lock)
        if checkpoint is not None:
            import json
            manifest = json.loads((Path(checkpoint) / 'manifest.json').read_text())
            validate_checkpoint_manifest(manifest, lock)
            if manifest['config'] != config:
                raise ValueError('Checkpoint configuration differs from requested model')
        self.settings = config
        self.data_lock = lock
        self.family = config['family']
        self.loops = config['loops']
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT / config['model'], trust_remote_code=True, local_files_only=True)
        for name, expected in lock['tokenizers_sha256'][Path(config['model']).name].items():
            if digest(ROOT / name) != expected:
                raise ValueError('Tokenizer differs from frozen audit')
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_config = AutoConfig.from_pretrained(ROOT / config['model'], trust_remote_code=True, local_files_only=True)
        if model_config.model_type != self.family or (self.family == 'qwen3' and self.loops != 1):
            raise ValueError('Backbone family/loop mismatch')
        model_config.use_cache = False
        if self.family == 'ouro':
            model_config.total_ut_steps = self.loops
        self.backbone = AutoModelForCausalLM.from_pretrained(ROOT / config['model'], config=model_config,
            trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map='cuda')
        self.backbone.requires_grad_(False)
        self.head = torch.nn.Linear(model_config.hidden_size, 1, bias=True, device='cuda', dtype=torch.float32)
        token_ids = [self.tokenizer.encode(value, add_special_tokens=False) for value in ('Yes', 'No')]
        if any(len(value) != 1 for value in token_ids):
            raise ValueError('Yes/No head initialization requires single tokens')
        with torch.no_grad():
            initial = self.backbone.lm_head.weight[token_ids[0][0]].float() - self.backbone.lm_head.weight[token_ids[1][0]].float()
            self.head.weight.copy_(initial.unsqueeze(0))
            self.head.bias.zero_()
        self.backbone = get_peft_model(self.backbone, LoraConfig(r=config['lora_rank'],
            lora_alpha=config['lora_alpha'], lora_dropout=0., bias='none', target_modules=config['lora_targets']))
        if checkpoint is not None:
            state = load_file(checkpoint / 'adapter/adapter_model.safetensors', device='cuda')
            set_peft_model_state_dict(self.backbone, state)
            self.head.load_state_dict(torch.load(checkpoint / 'head.pt', map_location='cuda', weights_only=True))
        if config['gradient_checkpointing']:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if any(parameter.requires_grad and 'lora_' not in name for name, parameter in self.backbone.named_parameters()):
            raise ValueError('Only LoRA may update backbone parameters')

    def score_sequences(self, sequences):
        encoded = self.tokenizer.pad({'input_ids': sequences}, padding=True, return_tensors='pt')
        attention_mask = encoded['attention_mask']
        encoded = encoded.to('cuda')
        decoder = self.backbone.get_base_model().model
        if self.settings.get('causal_padding_fastpath', False):
            output = decoder(input_ids=encoded['input_ids'], use_cache=False)
        else:
            output = decoder(**encoded, use_cache=False)
        if self.family == 'ouro':
            result, states, gates = output
            if len(states) != self.loops or result.past_key_values is not None:
                raise ValueError('Ouro loop/cache mismatch')
        else:
            states = [output.last_hidden_state]
        selected = select_last_states(states, attention_mask)
        return self.head(selected).squeeze(-1), encoded['input_ids'].numel()

    def forward(self, row, sequences):
        logits, padded = self.forward_records([row], [sequences])
        return logits[0], padded

    def forward_records(self, records, sequences):
        if len(records) != len(sequences) or not records:
            raise ValueError('Nonempty matching record and sequence batches required')
        flat = [candidate for candidates in sequences for candidate in candidates]
        scores_by_index, padded = [None] * len(flat), 0
        self.forward_calls = 0
        for indices in candidate_batches(flat, self.settings['candidate_batch'], self.settings['padded_token_budget']):
            scores, count = self.score_sequences([flat[index] for index in indices])
            for index, value in zip(indices, scores.unbind(), strict=True):
                scores_by_index[index] = value
            padded += count
            self.forward_calls += 1
        logits, start = [], 0
        for row, candidates in zip(records, sequences, strict=True):
            if not candidates:
                raise ValueError('Empty candidate group')
            end = start + len(candidates)
            logits.append(class_logits(torch.stack(scores_by_index[start:end]), row['kind']))
            start = end
        return logits, padded

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(directory / 'adapter', safe_serialization=True, save_embedding_layers=False)
        torch.save(self.head.state_dict(), directory / 'head.pt')
