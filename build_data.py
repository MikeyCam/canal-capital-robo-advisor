import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import time
import warnings
import contextlib
import requests
from urllib3.exceptions import InsecureRequestWarning

# ------------------------------
# SSL Bypass Context Manager
# ------------------------------
old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()
    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        opened_adapters.add(self.get_adapter(url))
        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings["verify"] = False
        return settings
    requests.Session.merge_environment_settings = merge_environment_settings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings
        for adapter in opened_adapters:
            try: adapter.close()
            except: pass

# ------------------------------
# ETF Universe
# ------------------------------
TICKERS = [
    "VTI","VEA","IEFA","VWO","IEMG","VBR","MTUM","ACWV",
    "BND","LQD","HYG","TLT","TIP","EMB",
    "VNQ","VNQI","DBC","GLD","BIL","PFF"
]

YEARS_BACK = 5
start = date.today() - timedelta(days=int(365.25*YEARS_BACK)+5)

print(f"Downloading {len(TICKERS)} ETFs from {start} to today (no SSL verification)...")

# ------------------------------
# Download each ticker separately (avoid rate limit)
# ------------------------------
all_px = []
for t in TICKERS:
    try:
        with no_ssl_verification():
            df = yf.download(t, start=start, auto_adjust=True, progress=False)[["Close"]]
        if not df.empty:
            df = df.rename(columns={"Close": t})
            all_px.append(df)
            print(f"✅ {t}: {len(df)} rows")
        else:
            print(f"⚠️ {t}: no data")
    except Exception as e:
        print(f"❌ Failed {t}: {e}")
    time.sleep(1)  # small pause to avoid blocks

# ------------------------------
# Combine
# ------------------------------
if not all_px:
    raise RuntimeError("No ETF data downloaded.")

px = pd.concat(all_px, axis=1).dropna(how="all")

# Daily returns
rets_daily = px.pct_change().dropna()

# ------------------------------
# Save
# ------------------------------
px.to_csv("etf_prices.csv")
rets_daily.to_csv("etf_returns_daily.csv")

print(f"✅ Data saved: etf_prices.csv and etf_returns_daily.csv ({rets_daily.shape[0]} rows, {rets_daily.shape[1]} tickers)")
