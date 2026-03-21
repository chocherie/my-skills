# Dashboard Design Guide

## Design: "Obsidian Command Center"

Dark theme optimized for data-dense trading dashboards.

### Color System (OKLCH)
- Background: `oklch(0.08 0.01 65)` (near-black warm)
- Card: `oklch(0.12 0.01 65)` with `backdrop-blur`
- Gold accent: `oklch(0.85 0.15 85)` (#F5C542 equivalent)
- Bullish: `oklch(0.75 0.2 155)` (electric green)
- Bearish: `oklch(0.65 0.25 25)` (hot coral)
- Neutral: `oklch(0.6 0.02 250)` (cool gray)

### Typography
- Headings: Space Grotesk (Google Fonts)
- Data/Mono: JetBrains Mono (Google Fonts)
- Body: System sans-serif stack

### Page Structure
1. **Header**: Logo + title, Methodology link, last-updated indicator
2. **Hero**: Gold price chart (Recharts) with range selector (1M/3M/6M/1Y/3Y/ALL)
3. **Consensus Bar**: Aggregate signal (Bullish/Bearish/Mixed), net score, dimension breakdown
4. **Signal Grid**: 2x4 card grid, each card shows dimension name, description, confidence gauge, signal badge, score
5. **Detail Panel**: Expandable on card click — signal history chart, equity curve, IS vs OOS metrics, overfit ratio
6. **Backtest Table**: Full comparison table with color-coded metrics
7. **Methodology Page**: Separate route — data quality audit, signal timeframe analysis, data source assessment

### Key Components
- `ConfidenceGauge`: Circular SVG gauge (0-100%)
- `SignalBadge`: Colored pill (BULLISH/NEUTRAL/BEARISH)
- `DimensionCard`: Glassmorphism card with hover glow
- `DimensionDetail`: Slide-in panel with Recharts equity curve
- `ConsensusBar`: Segmented bar (green/gray/red)
- `BacktestTable`: Sortable table with conditional formatting
- `GoldPriceChart`: Area chart with range selector

### Data Format
All data served as static JSON from `/data/`:
- `gold_chart.json`: `[{date, price}]`
- `dimensions.json`: `{dim1: {name, signal, confidence, score, history[], equity[], backtest{}}}`
- `dashboard.json`: `{gold_price, consensus, bullish_count, bearish_count}`
- `comparison.json`: `[{dimension, return, sharpe, max_dd, win_rate, oos_sharpe, overfit}]`
- `data_quality.json`: Quality audit results
