import warnings
warnings.filterwarnings("ignore")

import paddle
print(paddle.__version__)


import paddle
from paddle.nn import Transformer

# src: [batch_size, tgt_len, d_model]
enc_input = paddle.rand((2, 4, 128))
# tgt: [batch_size, src_len, d_model]
dec_input = paddle.rand((2, 6, 128))
# src_mask: [batch_size, n_head, src_len, src_len]
enc_self_attn_mask = paddle.rand((2, 2, 4, 4))
# tgt_mask: [batch_size, n_head, tgt_len, tgt_len]
dec_self_attn_mask = paddle.rand((2, 2, 6, 6))
# memory_mask: [batch_size, n_head, tgt_len, src_len]
cross_attn_mask = paddle.rand((2, 2, 6, 4))
transformer = Transformer(128, 2, 4, 4, 512)
output = transformer(enc_input,
                     dec_input,
                     enc_self_attn_mask,
                     dec_self_attn_mask,
                     cross_attn_mask)  
# [2, 6, 128]
print(output)