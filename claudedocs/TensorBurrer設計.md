# TensorBurrer設計

TensorBuffer型の使用例

```
// bufferの確保(gpu上に確保)
let buf = TensorBuffer::new()
buf.alloc([512], 5)          // [512]のテンソルを５個確保しておく
buf.alloc([512, 128], 20)    // [512, 512]のテンソルを２０個確保しておく
buf.alloc([512, 512], 10)    // [512, 512]のテンソルを１０個確保しておく
buf.alloc([512, 2048], 100)  // [512, 2048]のテンソルを１００個確保しておく

buf.alloc_learnable([512, 128], 20)    // learnableな[512, 512]のテンソルを２０個確保しておく
buf.alloc_learnable([512, 512], 10)    // learnableな[512, 512]のテンソルを１０個確保しておく

// bufferからテンソルを取得する
tensor attn_norm_0: float16[512] = buf.ones([512], [d_model])  // [512]のテンソルを所得する

tensor W_v_0: float16[512, 128] learnable = buf.positional_encoding_learnable(d_model, num_kv_heads * head_dim)
tensor W_o_0: float16[512, 512] learnable = buf.positional_encoding_learnable(num_q_heads * head_dim, d_model)

// bufferへテンソルを返す
//   返されたテンソルを操作した場合の動作は保証されない。可能ならばエラーにしたい。
buf.recycle(W_v_0)

// 確保したテンソルの破棄
buf.clear_all()                  // 確保したテンソルをすべて破棄する。
buf.clear([512, 128])            // テンソルの種類ごとに破棄することもできる。
buf.clear_learnable([512, 128])  // learnableなテンソルは別扱い
```

