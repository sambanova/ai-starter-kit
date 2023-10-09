import pandas as pd
from llama_hub.sec_filings.base import SECFilingsLoader

SNP500_CSV_FILE = "ticker_to_download.csv"


def main():
    df = pd.read_csv(SNP500_CSV_FILE)
    forms = ["10-K", "10-Q"]
    last_n = 1
    i = 0
    for i, ticker in enumerate(sorted(df.Symbol)):
        # if i <222:
        #     i+=1
        #     continue
        ticker = ticker.lower()
        for form in forms:
            try:
                loader = SECFilingsLoader(
                    tickers=[ticker], amount=last_n, filing_type=form
                )
                loader.load_data()
                print(i, ticker)
            except Exception as ex:
                print(ticker, form, ex)
        i += 1


if __name__ == "__main__":
    main()
