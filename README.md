# Aurora Cash Swing Scanner

Nifty 200 cash-market swing scanner with MACD entry confirmation, ATR/HV stop alternatives, risk-based position sizing and trade lifecycle monitoring.

- Dashboard: GitHub Pages root `index.html`
- Scanner engine: `cash_swing/scanner.py`
- Dependencies: `cash_swing/requirements.txt`
- Latest output: `cash_swing/data/latest.json`
- Lifecycle history: `cash_swing/reports/trade_lifecycle_history.csv`
- Scheduled workflow: `.github/workflows/update_cash_swing.yml`

The evening signal uses the latest completed daily candle. Entry and quantity are provisional until the next session opening price is available. ATR and HV60 are parallel stop-model alternatives, not simultaneous positions.
