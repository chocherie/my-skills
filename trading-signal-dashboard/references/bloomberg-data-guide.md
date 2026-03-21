# Bloomberg Data Sources for Gold Trading Signals

## Priority 1 — CFTC COT (Institutional Flows)

Export from Bloomberg Terminal via `COT <GO>` → Gold 99.50% USD/t oz → Disaggregated → Futures Only → Export.

Working tickers (confirmed):
- `CFFDUMML Index` — Managed Money Long
- `CFFDUMMS Index` — Managed Money Short
- `CFFDUMMN Index` — Managed Money Net
- `CFFDUPML Index` — Producer/Merchant Long
- `CFFDUPMS Index` — Producer/Merchant Short
- `CFFDUPMN Index` — Producer/Merchant Net
- `CFFDUSWN Index` — Swap Dealers Net

Frequency: Weekly. Date range: 2006+.

## Priority 2 — GPR Index (Geopolitical Risk)

NOT on Bloomberg. Free download from: https://www.matteoiacoviello.com/gpr.htm
- Daily data: `data_gpr_daily_recent.xls` (1985-present)
- Monthly data: `data_gpr_export.xls` (1900-present)
- Contains: GPR, GPR Threats (GPRT), GPR Acts (GPRA)

## Priority 3 — Bloomberg Macro & Sentiment

| Ticker | Field | Description |
|--------|-------|-------------|
| GLD US Equity | EQY_SH_OUT, FUND_TOTAL_ASSETS | GLD shares outstanding + AUM |
| IAU US Equity | EQY_SH_OUT, FUND_TOTAL_ASSETS | IAU shares outstanding + AUM |
| GVZ Index | PX_LAST | CBOE Gold ETF Volatility |
| CESIUSD Index | PX_LAST | Economic Surprise Index |
| BCOMGC Index | PX_LAST | Bloomberg Commodity Gold Sub |
| CPI YOY Index | PX_LAST | US CPI YoY |
| CPUPAXFE Index | PX_LAST | US Core CPI YoY |
| NFP TCH Index | PX_LAST | US NFP Change |
| FDTR Index | PX_LAST | Fed Funds Rate |
| USGGT10Y Index | PX_LAST | US 10Y Real Yield (TIPS) |
| GC1 Comdty | FUT_AGGTE_OPEN_INT, PX_VOLUME | Gold Futures OI + Volume |
| GC2 Comdty | PX_LAST | Gold 2nd month (for curve spread) |

All daily except CPI/NFP (monthly). Date range: 2005+.

## Yahoo Finance Fallbacks (Free)

When Bloomberg is unavailable, these Yahoo Finance tickers provide proxies:
- Gold: `GC=F` (futures) or `GLD` (ETF)
- DXY: `DX-Y.NYB`
- VIX: `^VIX`
- Crude: `CL=F`
- Silver: `SI=F`
- JPY: `JPY=X`
- BTC: `BTC-USD`
- Treasury: `^TNX` (10Y yield), `^FVX` (5Y yield)
- TIPS: `TIP` (ETF proxy)
