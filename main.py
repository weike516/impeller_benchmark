from Impeller.train import train
from Impeller.config import create_args
from Impeller.data import download_example_data, load_and_process_example_data

def main():
    dataset = "10XVisium"
    # dataset = "Stereoseq"
    # dataset = "SlideseqV2"
    download_example_data(dataset)
    data, val_mask, test_mask, x, original_x = load_and_process_example_data(dataset)
    args = create_args()
    test_l1_distance, test_cosine_sim, test_rmse = train(args, data, val_mask, test_mask, x, original_x)
    print(f"Final l1_distance: {test_l1_distance}, test_cosine_sim: {test_cosine_sim}, test_rmse: {test_rmse}.")

if __name__ == "__main__":
    main()