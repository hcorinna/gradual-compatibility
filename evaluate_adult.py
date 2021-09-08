import evaluate_regularizers
import config
import warnings
warnings.filterwarnings('ignore')

def main():
    evaluate_regularizers.regularize(config.adult)

if __name__ == "__main__":
    main()