# Workflow: Implement and Verify

Use this after implementing each batch of signals.

## Steps

1. **Run data layer**
   ```bash
   python data_layer.py
   ```
   Verify: all series downloaded, date range printed, no errors.

2. **Run signal sanity checks** (for newly implemented signals)
   ```python
   # Quick check: no all-one-direction signals
   for sig_id, fn in NEW_SIGNALS:
       sig = fn(df)
       assert sig.value_counts().get(1, 0) > 0.1 * len(sig), f"{sig_id} never goes long"
       assert sig.value_counts().get(-1, 0) > 0.1 * len(sig), f"{sig_id} never goes short"
       assert sig.isna().sum() < 0.5 * len(sig), f"{sig_id} too many NaNs"
   ```

3. **Run backtest for new signals only**
   ```bash
   python backtest_runner.py --signals S01,S02  # or --all
   ```

4. **Check look-ahead bias**
   ```bash
   cd ../four-eye-backtest-audit/scripts
   python execution_timing_audit.py --signal-file ../../treasury-signals/signals.py
   ```

5. **Update docs**
   - Check off completed items in `TODO.md`
   - Update grade in `docs/quality.md`
   - Update `specs/README.md` verification status

6. **Commit**
   ```bash
   git add -p  # review each change
   git commit -m "feat(signals): implement S01-S05 momentum signals"
   ```
