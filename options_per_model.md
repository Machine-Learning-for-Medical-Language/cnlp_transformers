This table lists features currently in v0.7.0. Descriptions of these features can be found in `src/cnlpt/cnlp_args`.

| Features        | cnlpt (default)    | hier               | cnn | lstm |
| --------------- | ------------------ | ------------------ | --- | ---- |
| class_weights   | :heavy_check_mark: | :heavy_check_mark: |     |      |
| freeze          | :heavy_check_mark: | :heavy_check_mark: |     |      |
| layer           | :heavy_check_mark: | :heavy_check_mark: |     |      |
| token           | :heavy_check_mark: | :heavy_check_mark: |     |      |
| use_prior_tasks | :heavy_check_mark: |                    |     |      |

Options starting with `cnn`, `lstm`, or `hier` are, of course, specific to that model:

- `cnn`: `cnn_embed_dim`, `cnn_num_filters`, `cnn_filter_sizes`
- `lstm`: `lstm_embed_dim`, `lstm_hidden_size`
- `hier`: `hier_num_layers`, `hier_hidden_dim`, `hier_n_head`, `hier_d_k`, `hier_d_v`, `hier_dropout`
