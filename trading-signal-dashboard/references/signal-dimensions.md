# 8-Dimension Gold Signal Framework

## Dimension 1: Technical Indicators
Sub-indicators: RSI(9), MACD(8,17,6), Bollinger Bands(20,2), EMA 50/200 crossover.
Weights: RSI 0.25, MACD 0.25, BB 0.25, EMA 0.25.
Logic: Z-score normalize each sub-indicator over 252-day rolling window. Composite = weighted sum. Signal: >0.5 = Bullish, <-0.5 = Bearish, else Neutral. Confidence = abs(score) capped at 100.

## Dimension 2: Macroeconomic Data
Sub-indicators: Real yields momentum (20d), TIPS level, Fed Funds rate change, Economic Surprise Index.
Key insight: Rising real yields = bearish gold. Negative surprise = bullish gold.
Bloomberg enhanced: USGGT10Y (real yields), FDTR (fed funds), CESIUSD (surprise index).

## Dimension 3: News & Sentiment
Sub-indicators: ETF volume anomaly (z-score), GVZ fear gauge, Economic Surprise sentiment, premium/discount.
Key insight: Extreme volume + high GVZ = contrarian signal. Use as mean-reversion indicator.

## Dimension 4: Intermarket Analysis
Sub-indicators: DXY momentum, VIX regime, crude oil correlation, real yields, GC1-GC2 curve spread, JPY.
Key insight: DXY is the strongest intermarket driver. Inverted gold curve (backwardation) = bullish.

## Dimension 5: Institutional Flows (Bloomberg Critical)
Sub-indicators: COT Managed Money net change, Producer/Merchant positioning, GLD/IAU shares outstanding change, AUM flow.
Key insight: Extreme Managed Money positioning is a contrarian signal. Use z-score of net position change, not level.

## Dimension 6: Seasonality
Sub-indicators: Monthly return bias, day-of-week bias, Indian wedding season (Nov-Mar), Chinese NY (Jan-Feb), September effect.
Weights: Monthly 0.35, DOW 0.25, Cultural 0.40.

## Dimension 7: Geopolitical Risk (GPR Index)
Sub-indicators: GPR Index level, GPR Threats, GPR Acts, GVZ, VIX.
Key insight: GPR spikes are short-lived gold catalysts. Use momentum (rate of change) not level. Combine with GVZ for confirmation.

## Dimension 8: Market Structure
Sub-indicators: Gold-silver ratio, higher-high/lower-low structure, futures OI change, volume confirmation.
Key insight: Expanding OI + rising price = trend confirmation. Contracting OI = trend exhaustion.

## Signal Scoring Convention
All dimensions output: score (continuous float), signal (+1/0/-1), confidence (0-100%).
Discretization: score > 0.5 → Bullish (+1), score < -0.5 → Bearish (-1), else Neutral (0).
Confidence: min(abs(score) * 100, 100).
