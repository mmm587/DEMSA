import math
from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.xlnet.modeling_xlnet import XLNetPreTrainedModel
from transformers.models.xlnet.modeling_xlnet import (
    XLNetLayer,
    SequenceSummary,
)


class XLNetModel(XLNetPreTrainedModel):
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        pass

    def _reorder_cache(self, past_key_values, beam_idx):
        pass

    def __init__(self, config):
        super().__init__(config)

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)

        self.init_weights()

    def get_input_embeddings(self):
        return self.word_embedding

    def set_input_embeddings(self, new_embeddings):
        self.word_embedding = new_embeddings

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        attn_mask = torch.ones([qlen, qlen])
        mask_up = torch.triu(attn_mask, diagonal=1)
        attn_mask_pad = torch.zeros([qlen, mlen])
        ret = torch.cat([attn_mask_pad, mask_up], dim=1)
        if self.same_length:
            mask_lo = torch.tril(attn_mask, diagonal=-1)
            ret = torch.cat([ret[:, :qlen] + mask_lo, ret[:, qlen:]], dim=1)

        ret = ret.to(self.device)
        return ret

    def cache_mem(self, curr_out, prev_mem):
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]
        if prev_mem is None:
            new_mem = curr_out[-self.mem_len:]
        else:
            new_mem = torch.cat([prev_mem, curr_out], dim=0)[-self.mem_len:]
        return new_mem.detach()

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat(
            [torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]
        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)
        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))
        if self.attn_type == "bi":
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            beg, end = klen, -1
        else:
            raise ValueError("Unknown `attn_type` {}.".format(self.attn_type))

        if self.bi_data:
            fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)
            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)
        pos_emb = pos_emb.to(self.device)
        return pos_emb

    def forward(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None, inputs_embeds=None, use_cache=True,
                output_attentions=None, output_hidden_states=None):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids.transpose(0, 1).contiguous()
            qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")
        token_type_ids = (
            token_type_ids.transpose(0, 1).contiguous()
            if token_type_ids is not None
            else None
        )
        input_mask = (
            input_mask.transpose(0, 1).contiguous(
            ) if input_mask is not None else None
        )
        attention_mask = (
            attention_mask.transpose(0, 1).contiguous()
            if attention_mask is not None
            else None
        )
        perm_mask = (
            perm_mask.permute(1, 2, 0).contiguous(
            ) if perm_mask is not None else None
        )
        target_mapping = (
            target_mapping.permute(1, 2, 0).contiguous()
            if target_mapping is not None
            else None
        )

        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        dtype_float = self.dtype
        device = self.device

        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError(
                "Unsupported attention type: {}".format(self.attn_type))

        assert (
                input_mask is None or attention_mask is None
        ), "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            if mlen > 0:
                mems_mask = torch.zeros(
                    [data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = torch.cat(
                    [torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1
                )
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(
                attn_mask
            )
        else:
            non_tgt_mask = None

        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        if token_type_ids is not None:
            if mlen > 0:
                mem_pad = torch.zeros(
                    [mlen, bsz], dtype=torch.long, device=device)
                cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)
            else:
                cat_ids = token_type_ids

            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = F.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(
                        0).unsqueeze(0).unsqueeze(0)
                )
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = []
        hidden_states = []

        for i, layer_module in enumerate(self.layer):
            if self.mem_len is not None and self.mem_len > 0 and use_cache is True:
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append(
                    (output_h, output_g) if output_g is not None else output_h
                )
            outputs = layer_module(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=seg_mat,
                mems=mems[i],
                target_mapping=target_mapping,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            output_h, output_g = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])

        if output_hidden_states:
            hidden_states.append(
                (output_h, output_g) if output_g is not None else output_h
            )

        output = self.dropout(output_g if output_g is not None else output_h)

        outputs = (output.permute(1, 0, 2).contiguous(),)

        if self.mem_len is not None and self.mem_len > 0 and use_cache is True:
            outputs = outputs + (new_mems,)

        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(
                    h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs
                )
            else:
                hidden_states = tuple(
                    hs.permute(1, 0, 2).contiguous() for hs in hidden_states
                )
            outputs = outputs + (hidden_states,)
        if output_attentions:
            if target_mapping is not None:
                attentions = tuple(
                    tuple(
                        att_stream.permute(2, 3, 0, 1).contiguous() for att_stream in t
                    )
                    for t in attentions
                )
            else:
                attentions = tuple(
                    t.permute(2, 3, 0, 1).contiguous() for t in attentions
                )
            outputs = outputs + (attentions,)

        return outputs  


class XLNetForSequenceClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None, inputs_embeds=None, use_cache=True,
                output_attentions=None, output_hidden_states=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask,
                                               head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        hidden_states = transformer_outputs[0]
        return hidden_states

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        pass

    def _reorder_cache(self, past_key_values, beam_idx):
        pass
