
### To run with webui:

```bash
dg dev -m definitions
```

### To run with CLI Materialization:

```
dagster asset materialize \
  --module-name definitions \
  --select "*"
```