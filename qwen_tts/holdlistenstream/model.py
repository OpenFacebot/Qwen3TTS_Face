"""Causal multimodal adapter around the pretrained Qwen Talker."""
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .protocol import Geometry, WindowPolicy, HOLD, LISTEN


class FaceMTP(nn.Module):
    def __init__(self, hidden_size, sizes, width=512, layers=5):
        super().__init__()
        from transformers import Qwen2Config, Qwen2Model
        self.project = nn.Linear(hidden_size, width)
        self.embeddings = nn.ModuleList(nn.Embedding(s, width) for s in sizes[:-1])
        config = Qwen2Config(vocab_size=1, hidden_size=width, intermediate_size=width * 3,
                             num_hidden_layers=layers, num_attention_heads=8, num_key_value_heads=4,
                             max_position_embeddings=32, attention_dropout=0)
        config._attn_implementation = "sdpa"
        self.decoder = Qwen2Model(config)
        self.decoder.embed_tokens = nn.Identity()
        self.heads = nn.ModuleList(nn.Linear(width, s, bias=False) for s in sizes[2:])

    def logits(self, hidden, codes):
        inputs = [self.project(hidden).unsqueeze(1)]
        inputs.extend(emb(codes[:, i]).unsqueeze(1) for i, emb in enumerate(self.embeddings))
        x = self.decoder(inputs_embeds=torch.cat(inputs, 1), use_cache=False).last_hidden_state
        return [head(x[:, i + 2]) for i, head in enumerate(self.heads)]

    @torch.no_grad()
    def generate(self, hidden, primary, sample):
        inputs = [self.project(hidden).unsqueeze(1), self.embeddings[0](primary[:, 0]).unsqueeze(1),
                  self.embeddings[1](primary[:, 1]).unsqueeze(1)]
        codes = [primary[:, 0], primary[:, 1]]
        for i, head in enumerate(self.heads):
            x = self.decoder(inputs_embeds=torch.cat(inputs, 1), use_cache=False).last_hidden_state[:, -1]
            code = sample(head(x))
            codes.append(code)
            if i < len(self.heads) - 1:
                inputs.append(self.embeddings[i + 2](code).unsqueeze(1))
        return torch.stack(codes, -1)


class HoldListenModel(nn.Module):
    def __init__(self, qwen, geometry, *, language="english", policy=None, mtp_width=512, mtp_layers=5):
        super().__init__()
        self.talker = qwen.model.talker
        self.config = qwen.model.config
        self.geometry = geometry
        self.policy = policy or WindowPolicy()
        self.language = language.lower()
        self.role_ids = qwen.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        tc = self.config.talker_config
        if self.language == "auto":
            self.prompt_codec_ids = [tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]
        else:
            self.prompt_codec_ids = [tc.codec_think_id, tc.codec_think_bos_id,
                                     tc.codec_language_id[self.language], tc.codec_think_eos_id]
        width = tc.hidden_size
        self.control = nn.Embedding(2, width)
        self.history_markers = nn.Embedding(2, width)
        self.face_embeddings = nn.ModuleList(nn.Embedding(s, width) for s in geometry.sizes)
        self.face_heads = nn.ModuleList(nn.Linear(width, geometry.sizes[i], bias=False) for i in range(geometry.primary_books))
        self.face_mtp = FaceMTP(width, geometry.sizes, mtp_width, mtp_layers) if geometry.mtp else None
        self.mtp_width, self.mtp_layers = mtp_width, mtp_layers
        for module in [self.control, self.history_markers, *self.face_embeddings, *self.face_heads]:
            nn.init.normal_(module.weight, std=.02)
        self.policy.prefix_limit(self.geometry, self.prompt_length)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def prompt_length(self):
        return len(self.role_ids) + len(self.prompt_codec_ids) + 3

    def settings(self):
        return dict(geometry=asdict(self.geometry), policy=asdict(self.policy), language=self.language,
                    mtp_width=self.mtp_width, mtp_layers=self.mtp_layers)

    def ids(self, value):
        return torch.as_tensor(value, dtype=torch.long, device=self.device)

    def text_embed(self, ids):
        return self.talker.text_projection(self.talker.get_text_embeddings()(ids))

    def pad_embed(self):
        return self.text_embed(self.ids([self.config.tts_pad_token_id]))

    def prompt(self, speaker, history=()):
        speaker = torch.as_tensor(speaker, device=self.device, dtype=self.talker.dtype).reshape(1, -1)
        if speaker.shape[1] != self.config.talker_config.hidden_size or not torch.isfinite(speaker).all():
            raise ValueError("Reference speaker vector has the wrong shape or nonfinite values")
        tc = self.config.talker_config
        role = self.text_embed(self.ids(self.role_ids))
        tags = self.talker.get_input_embeddings()(self.ids(self.prompt_codec_ids)) + self.pad_embed()
        spk = speaker + self.pad_embed()
        bos = self.talker.get_input_embeddings()(self.ids([tc.codec_pad_id])) + self.text_embed(self.ids([self.config.tts_bos_token_id]))
        codec_bos = self.talker.get_input_embeddings()(self.ids([tc.codec_bos_id])) + self.pad_embed()
        markers = self.history_markers(self.ids([0, 1]))
        return torch.cat((role, tags, spk, bos, codec_bos, markers[:1],
                          self.text_embed(self.ids(history)), markers[1:]), 0)

    def slot_embed(self, slots):
        slots = self.ids(slots)
        text = self.text_embed(slots.clamp_min(0))
        controls = self.control((-slots - 1).clamp(0, 1))
        return torch.where((slots >= 0).unsqueeze(-1), text, controls)

    def audio_embed(self, codes):
        codes = self.ids(codes)
        books = [self.talker.get_input_embeddings()(codes[..., 0])]
        books.extend(embedding(codes[..., book])
                     for book, embedding in enumerate(self.talker.code_predictor.get_input_embeddings(), 1))
        # Match Qwen's reduction, including its accumulation behavior in BF16.
        return torch.stack(books, -2).sum(-2) + self.pad_embed()

    @torch.inference_mode()
    def predict_audio(self, hidden, sample):
        """Run the fixed 16-codebook schedule without GenerationMixin overhead."""
        primary = sample(self.talker.codec_head(hidden)[..., :2048])
        inputs = torch.cat((hidden[:, None], self.talker.get_input_embeddings()(primary[:, None])), 1)
        predictor = self.talker.code_predictor
        out = predictor(inputs_embeds=inputs, use_cache=True, return_dict=True,
                        output_hidden_states=False, output_attentions=False)
        code = sample(out.logits[:, -1])
        codes = [primary, code]
        for step in range(1, self.config.talker_config.num_code_groups - 1):
            out = predictor(input_ids=code[:, None], past_key_values=out.past_key_values,
                            generation_steps=step, use_cache=True, return_dict=True,
                            output_hidden_states=False, output_attentions=False)
            code = sample(out.logits[:, -1])
            codes.append(code)
        return torch.stack(codes, -1)

    def face_embed(self, codes):
        codes = self.ids(codes)
        books = [embedding(codes[..., i]) for i, embedding in enumerate(self.face_embeddings)]
        if self.geometry.mtp:
            books = [books[0], torch.stack(books).sum(0)]
        return torch.stack(books, -2)

    def block_embed(self, text, audio, face):
        text, audio, face = self.ids(text), self.ids(audio), self.ids(face)
        count = len(text)
        return torch.cat((self.slot_embed(text), self.audio_embed(audio.reshape(count, 5, 16)),
                          self.face_embed(face.reshape(count, 2, -1)).reshape(count, -1, self.config.talker_config.hidden_size)), 1).flatten(0, 1)

    def enable_checkpointing(self):
        self.talker.code_predictor.config.use_cache = False
        self.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.face_mtp is not None:
            self.face_mtp.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, samples):
        embeddings, audio_targets, face_targets = [], [], []
        audio_indices, face_indices = [], []
        for b, sample in enumerate(samples):
            prompt = self.prompt(sample["speaker"], sample["history"])
            text, audio, face = self.ids(sample["text"]), self.ids(sample["audio"]), self.ids(sample["face"])
            embeddings.append(torch.cat((prompt, self.block_embed(text, audio, face))))
            first = int(sample["loss_from_block"])
            for block in range(first, len(text)):
                offset = len(prompt) + block * self.geometry.positions
                for i in range(5):
                    audio_indices.append((b, offset + 4 + i - 1))
                    audio_targets.append(audio[block * 5 + i])
                for i in range(2):
                    face_indices.append([(b, offset + 9 + i * self.geometry.primary_books + book - 1)
                                         for book in range(self.geometry.primary_books)])
                    face_targets.append(face[block * 2 + i])
        if not audio_targets or not face_targets:
            raise ValueError("A training batch must contain supervised stream blocks")
        inputs = pad_sequence(embeddings, batch_first=True)
        if inputs.shape[1] > self.policy.max_length:
            raise ValueError("Sequence exceeds configured budget")
        mask = torch.arange(inputs.shape[1], device=self.device)[None] < self.ids([len(x) for x in embeddings])[:, None]
        position_ids = (mask.long().cumsum(-1) - 1).clamp_min(0).unsqueeze(0).expand(3, -1, -1)
        outputs = self.talker.model(inputs_embeds=inputs, attention_mask=mask.long(), position_ids=position_ids,
                                    use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state
        a_indices, f_indices = self.ids(audio_indices), self.ids(face_indices)
        ah = hidden[a_indices[:, 0], a_indices[:, 1]]
        ac, fc = torch.stack(audio_targets), torch.stack(face_targets)
        # Every head sees the hidden state BEFORE its target, never the target's embedding.
        audio_loss = F.cross_entropy(self.talker.codec_head(ah).float(), ac[:, 0])
        _, auxiliary_loss = self.talker.forward_sub_talker_finetune(ac, ah)
        face_losses = [F.cross_entropy(head(hidden[f_indices[:, i, 0], f_indices[:, i, 1]]).float(), fc[:, i])
                       for i, head in enumerate(self.face_heads)]
        if self.face_mtp is not None:
            fh = hidden[f_indices[:, 0, 0], f_indices[:, 0, 1]]
            face_losses.extend(F.cross_entropy(logits.float(), fc[:, i + 2])
                               for i, logits in enumerate(self.face_mtp.logits(fh, fc)))
        face_loss = torch.stack(face_losses).mean()
        total = audio_loss + .3 * auxiliary_loss + face_loss
        return {"loss": total, "audio_loss": audio_loss.detach(), "audio_aux_loss": auxiliary_loss.detach(),
                "face_loss": face_loss.detach(), "supervised_blocks": len(audio_targets) // 5}


def adapter_from_settings(qwen, settings):
    config = dict(settings)
    config["geometry"] = Geometry(**config["geometry"])
    config["policy"] = WindowPolicy(**config["policy"])
    return HoldListenModel(qwen, **config)
