
def main():
    print("Welcome to LifeFinder!\n")
    
    print("To run the training script, use:")
    print("python -m scripts.train --arg1 value1 --arg2 value2")
    print("Replace --arg1 and --arg2 with the actual arguments required by train.py.\n")

    print("For example, to train with default settings:")
    print("python -m scripts.train --input_limit 1000 --epochs 10 --batch_size 32 --learning_rate 0.001")

    print("\nTo run the prediction script, use:")
    print("python -m scripts.predict --input path/to/input.csv --hidden_dim 64 --dropout 0.3")
    print("Replace path/to/input.csv with the actual path to your input CSV file.")

if __name__ == "__main__":
    main()
