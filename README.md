### Making plots

```bash
jq 'select(.message.key == "image_to_cuda")' logs/my_app.log.jsonl -c | 
jq '.message' -c  > temp.jsonl && \
python scripts/make_plots.py --input_file temp.jsonl --aggregation_keys benchmark key name batch_size --group_by_name benchmark --x_axis batch_size --y_axis time --y_axis_lim 0  --name image_to_cuda
# rm temp.jsonl
```