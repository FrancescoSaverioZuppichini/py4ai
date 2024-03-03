### Making plots

```bash
jq 'select(.message.key == "foo")' logs/my_app.log.jsonl -c | 
jq '.message' -c  > temp.jsonl && \
python scripts/make_plots.py --input_file temp.jsonl --aggregation_keys model batch_size --group_by_name model --x_axis batch_size --y_axis time --y_axis_lim 0  --name foo
# rm temp.jsonl
```