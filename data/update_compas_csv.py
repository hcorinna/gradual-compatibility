import pandas as pd

def main():
    df=pd.read_csv('propublica-recidivism_numerical-binsensitive.csv')
    cols = [c for c in df.columns if c.lower()[:14] != 'c_charge_desc_']
    df=df[cols]
    df.to_csv('propublica-recidivism_numerical-binsensitive_slim.csv', index=False)

if __name__ == "__main__":
    main()