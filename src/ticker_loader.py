import random


def get_random_sample(original_list: list, sample_size: int) -> list:
    """
    Returns a sample of strictly unique elements from the original list,
    ensuring the returned list length equals sample_size.
    """
    # 1. Deduplicate the original list using a set, then convert back to a list
    unique_base_list = list(set(original_list))
    
    # 2. Check the guard rail against the *unique* count, not the original count
    if sample_size > len(unique_base_list):
        raise ValueError(
            f"Sample size ({sample_size}) cannot be larger than the number of "
            f"unique elements in the original list ({len(unique_base_list)})."
        )
        
    # 3. random.sample guarantees unique selection and exact length
    return random.sample(unique_base_list, sample_size)

def load_total_tickers_nasdaq_2022():
    return 'ATVI,SNPS,CPRT,SBUX,SPLK,AMZN,AVGO,NFLX,KLAC,ABNB,JD,ILMN,INTU,FTNT,AMAT,ENPH,KHC,CSGP,AMD,DOCU,ANSS,ROST,MNST,TMUS,TRI,PTON,MDB,ADSK,MU,CDNS,LRCX,BKNG,CEG,NVDA,META,PEP,EXC,FAST,VRSN,INSM,ZM,COST,CTSH,ON,TEAM,MAR,NTES,INTC,PANW,VRTX,FI,PDD,AZN,MRVL,CTAS,GOOG,XEL,CSX,TXN,GOOGL,ORLY,VRSK,ASML,ZS,DDOG,TTD,ALGN,TSLA,AMGN,ADP,SMCI,NXPI,CSCO,BIDU,MELI,MTCH,ADBE,HON,MDLZ,EBAY,CRWD,LULU,PYPL,WDAY,REGN,DXCM,MCHP,AAPL,GILD,CHTR,SWKS,CMCSA,OKTA,LCID,SIRI,WBA,KDP,IDXX,XLNX,CDW,QCOM,AEP,PCAR,DLTR,ISRG,EA,MRNA,BIIB,PAYX,MSFT,SGEN,ADI'.split(',')

def load_xlu_tickers():
    return [
    "NEE", "SO", "DUK", "CEG", "AEP", "SRE", "D", "ETR", "XEL", "VST",
    "EXC", "ED", "PEG", "WEC", "PCG", "DTE", "AEE", "ATO", "CNP", "EIX",
    "AWK", "ES", "FE", "CMS", "LNT", "EVRG", "AES", "NI", "NRG", "PNW"
    ]


def load_xlp_tickers():
    return [
    "WMT",
    "COST",
    "PG",
    "KO",
    "PM",
    "MDLZ",
    "MO",
    "PEP",
    "CL",
    "TGT",
    "MNST",
    "KDP",
    "KR",
    "ADM",
    "SYY",
    "KVUE",
    "KMB",
    "HSY",
    "DG",
    "CHD"]

def load_xlre_tickers():
    return [
    "WELL",
    "PLD",
    "EQIX",
    "AMT",
    "DLR",
    "SPG",
    "VTR",
    "CBRE",
    "PSA",
    "O",
    "CCI",
    "IRM",
    "VICI",
    "EXR",
    "AVB",
    "SBAC",
    "EQR",
    "WY",
    "ESS",
    "INVH"
]

def load_overall_market():
    return ['SPY','QQQ','QQQE','IWM','XSW','XLY','XLV','XLU','XLRE','XLP','XLK','XLI','XLF','XLE','XLC','XLB','XBI']
    

def fix_data(universe_data):
    df_price=universe_data['price']
    df_vol=universe_data['volume']
    print(f'price shape before drop: {df_price.shape}')
    df_price=df_price.dropna(axis=1)
    print(f'price shape after drop: {df_price.shape}')
    df_vol=df_vol[df_price.columns]
    print(f'volume shape after drop: {df_vol.shape}')
    universe_data['price']=df_price
    universe_data['volume']=df_vol
    return universe_data