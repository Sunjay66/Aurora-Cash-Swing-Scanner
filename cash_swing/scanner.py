import json, os, warnings
from datetime import datetime
from io import StringIO
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
IST=ZoneInfo("Asia/Kolkata")
URL="https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
HEADERS={"User-Agent":"Mozilla/5.0","Referer":"https://www.niftyindices.com/"}
CAPITAL=float(os.getenv("CASH_PORTFOLIO_CAPITAL","1000000"))
RISK_PCT=float(os.getenv("CASH_RISK_PER_TRADE_PCT","1.0"))
ALLOC_PCT=float(os.getenv("CASH_MAX_STOCK_ALLOCATION_PCT","20.0"))

def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if v is None or pd.isna(v): return None
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,float)): return float(v) if np.isfinite(v) else None
    if isinstance(v,(pd.Timestamp,datetime)): return v.isoformat()
    if isinstance(v,(np.bool_,)): return bool(v)
    return v

def indicators(f):
    d=f.copy().sort_index()
    for p in (9,20,55): d[f"EMA{p}"]=d.Close.ewm(span=p,adjust=False).mean()
    prev=d.Close.shift()
    tr=pd.concat([d.High-d.Low,(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
    d["ATR14"]=tr.rolling(14).mean()
    d["CROSS_920"]=(d.EMA9>d.EMA20)&(d.EMA9.shift()<=d.EMA20.shift())
    d["CROSS_2055"]=(d.EMA20>d.EMA55)&(d.EMA20.shift()<=d.EMA55.shift())
    d["EMA9_EXIT"]=(d.Close<d.EMA9)
    return d

def main():
    os.makedirs("cash_swing/data/stocks",exist_ok=True); os.makedirs("cash_swing/reports",exist_ok=True)
    r=requests.get(URL,headers=HEADERS,timeout=30); r.raise_for_status()
    u=pd.read_csv(StringIO(r.text)); u["Symbol"]=u.Symbol.astype(str).str.strip().str.upper()
    u=u.drop_duplicates("Symbol")
    if len(u)<180: raise RuntimeError(f"Incomplete Nifty 200 universe: {len(u)}")
    tickers=[x+".NS" for x in u.Symbol]
    # Download in small batches: a single 200-symbol Yahoo request can silently
    # omit many tickers under throttling or response-size limits.
    frames={}; download_errors={}
    for offset in range(0,len(tickers),15):
        batch=tickers[offset:offset+15]
        try:
            block=yf.download(tickers=batch,period="3y",interval="1d",auto_adjust=True,
                              group_by="ticker",threads=False,progress=False)
            if block is not None and not block.empty:
                if isinstance(block.columns,pd.MultiIndex):
                    for ticker in batch:
                        if ticker in block.columns.get_level_values(0):
                            frames[ticker]=block[ticker].copy()
                elif len(batch)==1:
                    frames[batch[0]]=block.copy()
        except Exception as exc:
            for ticker in batch: download_errors[ticker]=str(exc)
    # Retry every missing symbol individually, so missing batch output is not
    # mistaken for "no trading opportunity".
    for ticker in tickers:
        if ticker in frames: continue
        try:
            one=yf.download(tickers=ticker,period="3y",interval="1d",auto_adjust=True,
                            group_by="column",threads=False,progress=False)
            if one is not None and not one.empty:
                if isinstance(one.columns,pd.MultiIndex):
                    one.columns=one.columns.get_level_values(0)
                frames[ticker]=one.copy()
        except Exception as exc:
            download_errors[ticker]=str(exc)
    names=dict(zip(u.Symbol,u.get("Company Name",u.Symbol)))
    fresh=[]; confirm=[]; running=[]; exits=[]; archive=[]; failed=[]; failed_details={}; dates=[]; processed=[]
    for ticker in tickers:
        sym=ticker[:-3]
        try:
            if ticker not in frames:
                raise ValueError("No daily OHLC data returned after batch and individual retry: "+download_errors.get(ticker,"no response"))
            f=frames[ticker].copy()
            f.columns=[str(c).title() for c in f.columns]
            f=f[[c for c in ["Open","High","Low","Close","Volume"] if c in f.columns]].dropna(subset=["Open","High","Low","Close"])
            if len(f)<100: raise ValueError(f"Insufficient usable OHLC history: {len(f)} rows")
            d=indicators(f); dates.append(pd.Timestamp(d.index[-1]).strftime("%Y-%m-%d")); processed.append(sym)
            company=str(names.get(sym,sym)); setups=[]
            sigs=np.flatnonzero(d.CROSS_920.fillna(False).to_numpy())
            for ix in sigs:
                entry_ix=ix+1
                pending_entry=entry_ix>=len(d)
                if pending_entry and ix!=len(d)-1: continue
                sig=d.iloc[ix]; ent=d.iloc[ix] if pending_entry else d.iloc[entry_ix]
                entry=float(sig.Close if pending_entry else ent.Open); atr=float(sig.ATR14) if pd.notna(sig.ATR14) else np.nan
                if not np.isfinite(atr) or atr<=0 or entry<=0: continue
                stop=entry-1.25*atr; risk=entry-stop
                stage2_ix=None
                for j in range(len(d) if pending_entry else entry_ix,len(d)):
                    if bool(d.iloc[j].CROSS_2055) and float(d.iloc[j].EMA9)>float(d.iloc[j].EMA20):
                        stage2_ix=j; break
                second_entry=None
                if stage2_ix is not None and stage2_ix+1<len(d):
                    second_entry=float(d.iloc[stage2_ix+1].Open)
                qty1=max(0,min(int((CAPITAL*RISK_PCT/100*.25)//risk),int((CAPITAL*ALLOC_PCT/100*.25)//entry)))
                qty2=0
                if second_entry and second_entry>0:
                    qty2=max(0,min(int((CAPITAL*RISK_PCT/100*.75)//risk),int((CAPITAL*ALLOC_PCT/100*.75)//second_entry)))
                total=qty1+qty2
                avg=((qty1*entry+qty2*second_entry)/total) if total else entry
                target=avg+2*risk
                stop_date=None; ema_exit_signal=None; exit_date=None; exit_price=None; target_hit=False
                start=entry_ix
                for j in range(start,len(d)):
                    bar=d.iloc[j]; op,hi,lo,cl=map(float,(bar.Open,bar.High,bar.Low,bar.Close))
                    if op<=stop: stop_date=pd.Timestamp(d.index[j]).strftime("%Y-%m-%d"); exit_price=op; exit_date=stop_date; break
                    if lo<=stop: stop_date=pd.Timestamp(d.index[j]).strftime("%Y-%m-%d"); exit_price=stop; exit_date=stop_date; break
                    if hi>=target: target_hit=True
                    if cl<float(bar.EMA9):
                        ema_exit_signal=pd.Timestamp(d.index[j]).strftime("%Y-%m-%d")
                        if j+1<len(d):
                            exit_date=pd.Timestamp(d.index[j+1]).strftime("%Y-%m-%d"); exit_price=float(d.iloc[j+1].Open)
                        break
                last=d.iloc[-1]; latest_date=pd.Timestamp(d.index[-1]).strftime("%Y-%m-%d")
                if stop_date or ema_exit_signal:
                    state="CLOSED"
                elif target_hit: state="TWO_R"
                else: state="OPEN"
                row={"symbol":sym,"company":company,"signal_date":pd.Timestamp(sig.name).strftime("%Y-%m-%d"),
                     "entry_date":"NEXT SESSION (PENDING)" if pending_entry else pd.Timestamp(ent.name).strftime("%Y-%m-%d"),"entry_price":round(entry,2),
                     "atr14":round(atr,2),"initial_stop":round(stop,2),"stage2_signal_date":pd.Timestamp(d.index[stage2_ix]).strftime("%Y-%m-%d") if stage2_ix is not None else None,
                     "stage2_entry_date":pd.Timestamp(d.index[stage2_ix+1]).strftime("%Y-%m-%d") if stage2_ix is not None and stage2_ix+1<len(d) else None,
                     "stage2_entry_price":round(second_entry,2) if second_entry else None,"quantity_stage1":qty1,"quantity_stage2":qty2,
                     "weighted_entry":round(avg,2),"risk_per_share":round(risk,2),"target_2r":round(target,2),
                     "latest_close":round(float(last.Close),2),"latest_date":latest_date,
                     "unrealised_return_pct":round((float(last.Close)/avg-1)*100,2),"target_2r_touched":bool(target_hit),
                     "hard_stop_date":stop_date,"ema9_exit_signal_date":ema_exit_signal,"exit_date":exit_date,"exit_price":round(exit_price,2) if exit_price is not None else None,
                     "status":state,"sessions_held":int(len(d)-entry_ix)}
                setups.append(row)
            if not setups:
                # Valid data, fully scanned, but no qualifying EMA setup.
                continue
            # Keep the latest non-overlapping signal; archive all older completed setups.
            setups.sort(key=lambda x:x["signal_date"])
            for row in setups[:-1]:
                archive.append(row)
            row=setups[-1]
            if row["status"]=="CLOSED":
                latest_day=dates[-1]
                is_today_alert=(row.get("hard_stop_date")==latest_day or row.get("ema9_exit_signal_date")==latest_day)
                if is_today_alert: exits.append(row)
                else: archive.append(row)
            elif row["status"]=="TWO_R":
                running.append(row)
            else:
                # Second-stage crossover is a same-day confirmatory alert only.
                if row["stage2_signal_date"]==dates[-1]: confirm.append(row)
                elif row["signal_date"]==dates[-1]: fresh.append(row)
                else: running.append(row)
            stock={"symbol":sym,"company":company,"market_date":dates[-1],"setups":setups[::-1]}
            with open(f"cash_swing/data/stocks/{sym}.json","w",encoding="utf-8") as fp: json.dump(safe(stock),fp,indent=2)
        except Exception as e:
            print(f"FAILED {sym}: {e}"); failed.append(sym); failed_details[sym]=str(e)
    market_date=max(dates) if dates else None
    payload={"scanner":"Aurora Cash Swing Scanner","version":"2.1 EMA batch-retry","universe":"Nifty 200","market_date":market_date,
      "generated_at_ist":datetime.now(IST).isoformat(),"stocks_in_universe":len(u),"stocks_scanned":len(set(processed)),
      "stocks_with_no_qualifying_opportunity":len(set(processed)-set([x["symbol"] for x in fresh+confirm+running+exits+archive])),
      "processed_symbols":sorted(set(processed)),"failed_count":len(set(failed)),
      "failed_symbols":sorted(set(failed)),"failed_details":failed_details,"fresh_entries":fresh,"confirmatory_signals":confirm,"running_opportunities":running,
      "running_trades":[],"exit_alerts":exits,"two_r_achieved": [x for x in running if x["status"]=="TWO_R"],
      "trade_archive":archive,"entry_rule":"Stage 1: EMA 9 crosses above EMA 20; planned first tranche 25% at next session open. Stage 2: EMA 20 crosses above EMA 55 while EMA 9 remains above EMA 20; planned remaining 75% at next session open.",
      "risk_rule":"Initial hard stop = first entry price minus 1.25 x simple ATR14; stop is fixed. 2R target is based on weighted average entry and initial per-share risk. Maximum planned risk 1% of portfolio; maximum stock allocation 20%.",
      "exit_rule":"Hard stop intraday; EMA 9 close-below is an exit alert for next session open. Same-day stop/target collision resolves stop-first. No automatic time exit.",
      "notes":"All scanner records are hypothetical opportunities, not user-owned trades. Actual Running Trades remains empty until manually entered. 2R opportunities are retained for review; user decides whether to keep or close."}
    with open("cash_swing/data/latest.json","w",encoding="utf-8") as fp: json.dump(safe(payload),fp,indent=2,ensure_ascii=False)
    pd.DataFrame(fresh+confirm+running+exits+archive).to_csv("cash_swing/reports/ema_opportunities.csv",index=False)
    print(f"EMA scan complete: {market_date}; processed {payload['stocks_scanned']}/{len(u)}; failed {payload['failed_count']}; fresh {len(fresh)}; confirm {len(confirm)}; running {len(running)}; exits {len(exits)}")
    if failed: print("FAILED SYMBOL DETAILS: "+json.dumps(failed_details,ensure_ascii=False))
if __name__=="__main__": main()
