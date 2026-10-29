# @title MATRIZ INSTITUCIONAL + GRÁFICOS EXCLUSIVOS DEL TOP 15 + ROE/ROA + GOOGLE NEWS
!pip install yfinance seaborn -q

%matplotlib inline
import datetime
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

print("=============================================================================================================")
print(" MODELO INSTITUCIONAL: GRÁFICOS ENFOCADOS EN EL TOP 15 + MÉTRICAS COMPLETAS + GOOGLE NEWS")
print("=============================================================================================================")

# 1. ESTRUCTURA DE DATOS
sector_database = {
    'Tecnología': ['MSFT', 'NVDA', 'AAPL', 'AVGO', 'ORCL', 'CRM', 'CSCO', 'ADBE', 'AMD', 'PLTR', 'NOW', 'INTU', 'AMAT', 'MU', 'LRCX', 'ASML', 'TSM', 'SMH', 'QQQ'],
    'Comunicación': ['GOOGL', 'META', 'NFLX', 'CMCSA', 'DIS', 'TMUS', 'VZ', 'XLC'],
    'Consumo Discrecional': ['AMZN', 'TSLA', 'HD', 'MCD', 'LOW', 'BKNG', 'TJX', 'UBER', 'MELI', 'XLY'],
    'Consumo Básico': ['WMT', 'COST', 'PG', 'KO', 'PEP', 'PM', 'MDLZ', 'XLP', 'VIG'],
    'Salud': ['LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'XLV'],
    'Financiero': ['JPM', 'BRK-B', 'V', 'MA', 'BAC', 'GS', 'MS', 'WFC', 'C', 'NU', 'XLF'],
    'Industriales': ['GE', 'GEV', 'CAT', 'RTX', 'UNP', 'HON', 'DE', 'ETN', 'XLI'],
    'Energía': ['XOM', 'CVX', 'COP', 'EOG', 'WMB', 'SLB', 'KMI', 'PSX', 'VIST', 'VST', 'XLE'],
    'Materiales': ['LIN', 'SHW', 'APD', 'FCX', 'NEM', 'NUE', 'VMC', 'XLB'],
    'Utilities': ['NEE', 'SO', 'DUK', 'CEG', 'AEP', 'D', 'XEL', 'XLU'],
    'General / Mercado': ['SPY']
}

SECTOR_PE_MEDIANS = {
    'Tecnología': 30.65, 'Comunicación': 18.20, 'Consumo Discrecional': 21.00,
    'Consumo Básico': 19.00, 'Salud': 17.50, 'Financiero': 15.50,
    'Industriales': 20.50, 'Energía': 12.00, 'Materiales': 13.50, 'Utilities': 16.50,
    'General / Mercado': 22.00
}

all_tickers = list(set([ticker for tickers in sector_database.values() for ticker in tickers]))

# 2. PROCESAMIENTO TEMPORAL
end_date = datetime.datetime.now()
start_date_5y = end_date - datetime.timedelta(days=365 * 5)
start_date_3y = end_date - datetime.timedelta(days=365 * 3)
start_date_1y = end_date - datetime.timedelta(days=365)
start_date_6m = end_date - datetime.timedelta(days=180)
risk_free_rate = 0.045

print(f"Descargando datos históricos y fundamentales para {len(all_tickers)} activos...")
raw_data = yf.download(all_tickers, start=start_date_5y, end=end_date, progress=False)

data_5y = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']
data_3y = data_5y.loc[start_date_3y.strftime('%Y-%m-%d'):]
data_1y = data_5y.loc[start_date_1y.strftime('%Y-%m-%d'):]

sharpe_1y, sharpe_3y, sharpe_5y, momentum_6m, volatility_1y = {}, {}, {}, {}, {}

for ticker in data_5y.columns:
    ret_5y = data_5y[ticker].dropna().pct_change().dropna()
    if len(ret_5y) >= 1000:
        sharpe_5y[ticker] = (ret_5y.mean() * 252 - risk_free_rate) / (ret_5y.std() * np.sqrt(252))

    ret_3y = data_3y[ticker].dropna().pct_change().dropna()
    if len(ret_3y) >= 600:
        sharpe_3y[ticker] = (ret_3y.mean() * 252 - risk_free_rate) / (ret_3y.std() * np.sqrt(252))

    ret_1y = data_1y[ticker].dropna().pct_change().dropna()
    if len(ret_1y) >= 150:
        sharpe_1y[ticker] = (ret_1y.mean() * 252 - risk_free_rate) / (ret_1y.std() * np.sqrt(252))
        volatility_1y[ticker] = ret_1y.std() * np.sqrt(252) * 100

    prices_6m = data_5y[ticker].loc[start_date_6m.strftime('%Y-%m-%d'):].dropna()
    if len(prices_6m) > 10:
        momentum_6m[ticker] = ((prices_6m.iloc[-1] - prices_6m.iloc[0]) / prices_6m.iloc[0]) * 100
    else:
        momentum_6m[ticker] = 0

fundamental_data = {}
yf_tickers = yf.Tickers(" ".join(all_tickers))

for ticker in all_tickers:
    try:
        info = yf_tickers.tickers[ticker].info
        forward_pe = info.get('forwardPE', np.nan)
        if forward_pe is not None and (forward_pe <= 0 or forward_pe > 150): forward_pe = np.nan
        trailing_pe = info.get('trailingPE', forward_pe)
        pe_historico_estimado = trailing_pe * 0.95 if not np.isnan(trailing_pe) else (forward_pe * 1.05 if not np.isnan(forward_pe) else np.nan)

        peg_ratio = info.get('pegRatio', np.nan)
        if peg_ratio is not None and (peg_ratio <= 0 or peg_ratio > 10): peg_ratio = np.nan

        roe = info.get('returnOnEquity', np.nan)
        if roe is not None and not np.isnan(roe):
            if abs(roe) <= 5: roe = roe * 100
        else:
            roe = np.nan

        roa = info.get('returnOnAssets', np.nan)
        if roa is not None and not np.isnan(roa):
            if abs(roa) <= 5: roa = roa * 100
        else:
            roa = np.nan

        if ticker in ['SPY', 'QQQ', 'SMH', 'XLV', 'XLF', 'XLE', 'XLI', 'XLB', 'XLU', 'XLP', 'XLY', 'XLC', 'VIG']:
            if np.isnan(forward_pe): forward_pe = 20.0
            if np.isnan(peg_ratio): peg_ratio = 1.2
            if np.isnan(pe_historico_estimado): pe_historico_estimado = 19.0
            if np.isnan(roe): roe = 15.0
            if np.isnan(roa): roa = 7.0

        fundamental_data[ticker] = {
            'Forward P/E': forward_pe, 'PEG Ratio': peg_ratio,
            'P/E Histórico': pe_historico_estimado, 'ROE': roe, 'ROA': roa
        }
    except Exception:
        fundamental_data[ticker] = {'Forward P/E': 20.0, 'PEG Ratio': 1.2, 'P/E Histórico': 19.0, 'ROE': 15.0, 'ROA': 7.0}

# 3. MATRIZ MAESTRA Y CÁLCULO DE SCORE INSTITUCIONAL
df_master = pd.DataFrame(index=all_tickers)
df_master['Sharpe 1Y'] = df_master.index.map(sharpe_1y)
df_master['Sharpe 3Y'] = df_master.index.map(sharpe_3y)
df_master['Sharpe 5Y'] = df_master.index.map(sharpe_5y)
df_master['Momentum 6M (%)'] = df_master.index.map(momentum_6m)
df_master['Volatilidad 1Y (%)'] = df_master.index.map(volatility_1y)
df_master['Forward P/E'] = [fundamental_data[t]['Forward P/E'] for t in df_master.index]
df_master['PEG Ratio'] = [fundamental_data[t]['PEG Ratio'] for t in df_master.index]
df_master['P/E Histórico'] = [fundamental_data[t]['P/E Histórico'] for t in df_master.index]
df_master['ROE (%)'] = [fundamental_data[t]['ROE'] for t in df_master.index]
df_master['ROA (%)'] = [fundamental_data[t]['ROA'] for t in df_master.index]

ticker_to_sector = {t: s for s, tickers in sector_database.items() for t in tickers}
df_master['Sector'] = df_master.index.map(ticker_to_sector)
df_master['Sector P/E'] = df_master['Sector'].map(SECTOR_PE_MEDIANS)

df_master['Forward P/E'] = df_master.groupby('Sector')['Forward P/E'].transform(lambda x: x.fillna(x.median()))
df_master['P/E Histórico'] = df_master['P/E Histórico'].fillna(df_master['Forward P/E'])
df_master['PEG Ratio'] = df_master.groupby('Sector')['PEG Ratio'].transform(lambda x: x.fillna(x.median())).fillna(1.2)
df_master['Volatilidad 1Y (%)'] = df_master['Volatilidad 1Y (%)'].fillna(df_master['Volatilidad 1Y (%)'].median())
df_master['ROE (%)'] = df_master.groupby('Sector')['ROE (%)'].transform(lambda x: x.fillna(x.median())).fillna(15.0)
df_master['ROA (%)'] = df_master.groupby('Sector')['ROA (%)'].transform(lambda x: x.fillna(x.median())).fillna(7.0)

df_master['P/E vs Sector (%)'] = ((df_master['Forward P/E'] - df_master['Sector P/E']) / df_master['Sector P/E']) * 100
df_master['P/E vs Histórico (%)'] = ((df_master['Forward P/E'] - df_master['P/E Histórico']) / df_master['P/E Histórico']) * 100
df_master['P/E Relativo Promedio (%)'] = (df_master['P/E vs Sector (%)'] + df_master['P/E vs Histórico (%)']) / 2

def normalize(series, reverse=False):
    min_val, max_val = series.min(), series.max()
    if max_val == min_val: return series * 0 + 50
    normalized = (series - min_val) / (max_val - min_val) * 100
    return 100 - normalized if reverse else normalized

df_master['Sharpe Promedio'] = df_master[['Sharpe 1Y', 'Sharpe 3Y', 'Sharpe 5Y']].mean(axis=1)
score_sharpe = normalize(df_master['Sharpe Promedio'])

score_momentum_base = normalize(df_master['Momentum 6M (%)'])
pe_relativo_norm = df_master['P/E Relativo Promedio (%)'] / 100.0
momentum_multiplier = (1.0 - pe_relativo_norm).clip(0.3, 1.8)
score_momentum = normalize(score_momentum_base * momentum_multiplier)
df_master['Momentum Multiplier'] = momentum_multiplier

score_pe_relativo = normalize(df_master['P/E Relativo Promedio (%)'], reverse=True)
score_peg = normalize(df_master['PEG Ratio'], reverse=True)
score_volatility = normalize(df_master['Volatilidad 1Y (%)'], reverse=True)
score_roe = normalize(df_master['ROE (%)'], reverse=False)
score_roa = normalize(df_master['ROA (%)'], reverse=False)

df_master['Score'] = (
    (score_sharpe * 0.15) +
    (score_momentum * 0.15) +
    (score_pe_relativo * 0.15) +
    (score_peg * 0.10) +
    (score_volatility * 0.20) +
    (score_roe * 0.15) +
    (score_roa * 0.10)
)

# Extraer el Top 15 Global para graficar exclusivamente este subconjunto
df_top15 = df_master.sort_values(by='Score', ascending=False).head(15).copy()

# 4. GRÁFICOS AVANZADOS EXCLUSIVOS DEL TOP 15
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(20, 20))

# Sharpe (Top 15)
sc1 = axes[0, 0].scatter(df_top15['Sharpe 3Y'], df_top15['Sharpe 5Y'], c=df_top15['Sharpe 1Y'], cmap='viridis', s=80, edgecolor='k', alpha=0.9)
fig.colorbar(sc1, ax=axes[0, 0], label='Sharpe 1 Año')
m1, b1 = np.polyfit(df_top15['Sharpe 3Y'], df_top15['Sharpe 5Y'], 1)
axes[0, 0].plot(df_top15['Sharpe 3Y'], m1*df_top15['Sharpe 3Y'] + b1, color='red', linestyle='--', linewidth=1.5, label='Tendencia Lineal')
axes[0, 0].set_title('Top 15 - Sharpe Ratio: 3Y (X) vs 5Y (Y)', fontweight='bold')
axes[0, 0].legend(loc='lower right')
for ticker, row in df_top15.iterrows(): axes[0, 0].annotate(ticker, (row['Sharpe 3Y'], row['Sharpe 5Y']), fontsize=8, fontweight='bold', xytext=(4, 4), textcoords='offset points')

# P/E vs Sector (Top 15)
sc2 = axes[0, 1].scatter(df_top15['Forward P/E'], df_top15['Sector P/E'], c=df_top15['P/E vs Histórico (%)'], cmap='coolwarm', s=80, edgecolor='k', alpha=0.9)
fig.colorbar(sc2, ax=axes[0, 1], label='P/E vs Histórico (%)')
m2, b2 = np.polyfit(df_top15['Forward P/E'], df_top15['Sector P/E'], 1)
axes[0, 1].plot(df_top15['Forward P/E'], m2*df_top15['Forward P/E'] + b2, color='black', linestyle='--', linewidth=1.5, label='Tendencia Lineal')
axes[0, 1].set_title('Top 15 - Forward P/E (X) vs Mediana Sector (Y)', fontweight='bold')
axes[0, 1].legend(loc='lower right')
for ticker, row in df_top15.iterrows(): axes[0, 1].annotate(ticker, (row['Forward P/E'], row['Sector P/E']), fontsize=8, fontweight='bold', xytext=(4, 4), textcoords='offset points')

# Momentum vs Value Tilt (Top 15)
sc3 = axes[1, 0].scatter(df_top15['Momentum 6M (%)'], df_top15['Momentum Multiplier'], c=df_top15['Score'], cmap='plasma', s=80, edgecolor='k', alpha=0.9)
fig.colorbar(sc3, ax=axes[1, 0], label='Score Final')
m3, b3 = np.polyfit(df_top15['Momentum 6M (%)'], df_top15['Momentum Multiplier'], 1)
axes[1, 0].plot(df_top15['Momentum 6M (%)'], m3*df_top15['Momentum 6M (%)'] + b3, color='cyan', linestyle='--', linewidth=1.5, label='Tendencia Lineal')
axes[1, 0].set_title('Top 15 - Momentum 6M (X) vs Multiplicador Value Tilt (Y)', fontweight='bold')
axes[1, 0].legend(loc='lower left')
for ticker, row in df_top15.iterrows(): axes[1, 0].annotate(ticker, (row['Momentum 6M (%)'], row['Momentum Multiplier']), fontsize=8, fontweight='bold', xytext=(4, 4), textcoords='offset points')

# Forward P/E vs PEG Ratio (Top 15)
sc4 = axes[1, 1].scatter(df_top15['Forward P/E'], df_top15['PEG Ratio'], c=df_top15['Score'], cmap='YlGnBu', s=80, edgecolor='k', alpha=0.9)
fig.colorbar(sc4, ax=axes[1, 1], label='Score Final')
x_peg, y_peg = df_top15['Forward P/E'], df_top15['PEG Ratio']
sorted_idx = np.argsort(x_peg)
x_sorted, y_sorted = x_peg.iloc[sorted_idx], y_peg.iloc[sorted_idx]
coeffs_peg = np.polyfit(x_sorted, y_sorted, min(2, len(x_sorted)-1))
poly_curve = coeffs_peg[0] * (x_sorted ** 2) + coeffs_peg[1] * x_sorted + coeffs_peg[2]
axes[1, 1].plot(x_sorted, poly_curve, color='red', linestyle='--', linewidth=1.5, label='Tendencia')
axes[1, 1].set_title('Top 15 - Forward P/E (X) vs PEG Ratio (Y)', fontweight='bold')
axes[1, 1].legend(loc='upper right')
for ticker, row in df_top15.iterrows(): axes[1, 1].annotate(ticker, (row['Forward P/E'], row['PEG Ratio']), fontsize=8, fontweight='bold', xytext=(4, 4), textcoords='offset points')

# ROE vs ROA (Top 15)
sc5 = axes[2, 0].scatter(df_top15['ROE (%)'], df_top15['ROA (%)'], c=df_top15['Score'], cmap='magma', s=80, edgecolor='k', alpha=0.9)
fig.colorbar(sc5, ax=axes[2, 0], label='Score Final')
m5, b5 = np.polyfit(df_top15['ROE (%)'], df_top15['ROA (%)'], 1)
axes[2, 0].plot(df_top15['ROE (%)'], m5*df_top15['ROE (%)'] + b5, color='blue', linestyle='--', linewidth=1.5, label='Tendencia Lineal')
axes[2, 0].set_title('Top 15 - ROE % (X) vs ROA % (Y)', fontweight='bold')
axes[2, 0].set_xlabel('ROE (%)')
axes[2, 0].set_ylabel('ROA (%)')
axes[2, 0].legend(loc='lower right')
for ticker, row in df_top15.iterrows(): axes[2, 0].annotate(ticker, (row['ROE (%)'], row['ROA (%)']), fontsize=8, fontweight='bold', xytext=(4, 4), textcoords='offset points')

axes[2, 1].set_visible(False)
plt.tight_layout()
plt.show()

# 5. GRÁFICO DE BARRAS TOP 15
plt.figure(figsize=(12, 6))
bars = plt.barh(df_top15.index[::-1], df_top15['Score'][::-1], color=sns.color_palette("viridis", 15))
plt.title('🏆 Top 15 Activos por Score Institucional Global', fontsize=14, fontweight='bold')
plt.xlabel('Score Institucional (0 - 100)', fontsize=11)
plt.xlim(0, 100)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f} pts', va='center', ha='left', fontsize=9, fontweight='bold', color='#333333')
plt.tight_layout()
plt.show()

# 6. REPORTE AUTOMATIZADO DE NOTICIAS EN VIVO VÍA GOOGLE NEWS RSS
print("\n" + "="*145)
print("       📰 REPORTE CUALITATIVO Y TITULARES DE MERCADO EN VIVO (TOP 15 GOOGLE NEWS)")
print("="*145)

def fetch_google_news(ticker, query_extra="stock price market"):
    search_term = urllib.parse.quote(f"{ticker} {query_extra}")
    rss_url = f"https://news.google.com/rss/search?q={search_term}&hl=en-US&gl=US&ceid=US:en"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    req = urllib.request.Request(rss_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            items = root.findall('.//item')
            headlines = []
            for item in items[:3]:
                title = item.find('title').text if item.find('title') is not None else 'Sin título'
                source = item.find('source').text if item.find('source') is not None else 'Google News'
                pubDate = item.find('pubDate').text if item.find('pubDate') is not None else ''
                headlines.append((title, source, pubDate))
            return headlines
    except Exception:
        return []

for ticker in df_top15.index:
    score = df_top15.loc[ticker, 'Score']
    sector = df_master.loc[ticker, 'Sector']
    print(f"\n🔹 Ticker: {ticker} | Score: {score:.1f} pts | Sector: {sector}")

    news = fetch_google_news(ticker)
    if news:
        print("   Últimas novedades y titulares de mercado:")
        for idx, (title, source, pdate) in enumerate(news, 1):
            date_clean = pdate[:16] if pdate else "Reciente"
            print(f"     {idx}. [{source}] {title} ({date_clean})")
    else:
        print("   (Sin titulares recientes obtenidos en este momento)")

# 7. DESPLIEGUE DE PANELES SECTORIALES (CON TODAS LAS MÉTRICAS DETALLADAS)
print("\n" + "="*145)
print("       📊 PANELES SECTORIALES DETALLADOS (SHARPE 1Y/3Y/5Y & P/E COMPARACIONES)")
print("="*145)

top_companies_all = []
for sector in sector_database.keys():
    df_sector = df_master[df_master['Sector'] == sector].copy()
    sorted_sector = df_sector.sort_values(by='Score', ascending=False)
    top_companies_all.append(sorted_sector)

    display(HTML(f"<h3 style='color: #2e6c80;'>📊 CUADRO SECTORIAL: {sector}</h3>"))

    columns_to_show = [
        'Sharpe 1Y', 'Sharpe 3Y', 'Sharpe 5Y',
        'Momentum 6M (%)', 'Volatilidad 1Y (%)',
        'Forward P/E', 'P/E vs Sector (%)', 'P/E vs Histórico (%)',
        'ROE (%)', 'ROA (%)', 'PEG Ratio', 'Score'
    ]

    styled_table = sorted_sector[columns_to_show].head(12).style.format({
        'Sharpe 1Y': '{:,.2f}', 'Sharpe 3Y': '{:,.2f}', 'Sharpe 5Y': '{:,.2f}',
        'Momentum 6M (%)': '{:+.1f}%', 'Volatilidad 1Y (%)': '{:+.1f}%',
        'Forward P/E': '{:,.2f}', 'P/E vs Sector (%)': '{:+.1f}%', 'P/E vs Histórico (%)': '{:+.1f}%',
        'ROE (%)': '{:,.1f}%', 'ROA (%)': '{:+.1f}%',
        'PEG Ratio': '{:,.2f}', 'Score': '{:,.1f} pts'
    }).background_gradient(cmap='YlGn', subset=['Score'])

    display(styled_table)

    import json

# 1. Convertir la matriz maestra a JSON legible
master_dict = df_master.to_dict(orient='index')

# 2. Convertir el Top 15 e incluir sus noticias en vivo
top15_list = []
for ticker in df_top15.index:
    item = df_master.loc[ticker].to_dict()
    item['Ticker'] = ticker
    item['News'] = fetch_google_news(ticker)
    top15_list.append(item)

output_data = {
    "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "top15": top15_list,
    "master_database": master_dict
}

# 3. Guardar archivo JSON
with open('data_master.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("✅ Archivo 'data_master.json' generado con éxito para la Web App.")
