#this script is a test for the data layer :
#can we load and clean data, are the shapes correct, can we compute log returns
#are the Inf/NaN values

import argparse
import torch

from src.data.candle import(
    load_candle_splits,
    clean_candle_splits,
    describe_split,
    compute_close_log_returns,
    validate_clean_split
)

def main() -> None:
    #documentation for the command line interface
    parser = argparse.ArgumentParser(
        description='Inspect and validate candle dataset'
    )
    #add an argument to the parser that tells us the directory the data is in
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing train.pt,val,pt,test.pt'
    )
    #this stores the command typed into terminal so we can access it later
    args = parser.parse_args()

    print("Loading raw candle splits...")
    train_raw,val_raw,test_raw = load_candle_splits(args.data_dir)
    print("\nRaw Splits")
    describe_split(train_raw,"Raw Train Data")
    describe_split(val_raw,"Raw Val Data")
    describe_split(test_raw,"Raw Test Data")
    
    print("\nCleaning splits...")
    train,val,test = clean_candle_splits(train_raw,val_raw,test_raw)
    print("\nCleaned Splits")
    describe_split(train,"Train Data")
    describe_split(val,"Val Data")
    describe_split(test,"Test Data")

    print("\nValidating cleaned splits...")
    validate_clean_split(train)
    validate_clean_split(val)
    validate_clean_split(test)

    print("\nComputing first-day close log returns...")
    x, aux, day = train["samples"][0]
    returns = compute_close_log_returns(x, train)
    print("day:", day)
    print("x shape:", tuple(x.shape))
    print("returns shape:", tuple(returns.shape))
    print("returns mean:", returns.mean().item())
    print("returns std:", returns.std().item())
    print("returns min:", returns.min().item())
    print("returns max:", returns.max().item())
    print("returns has NaN:", torch.isnan(returns).any().item())
    print("returns has Inf:", torch.isinf(returns).any().item())

#this just says if we run this script directly, the run main
#if this script gets imported as part of another script, then
#dont automatically run main
if __name__ == "__main__":
    main()



