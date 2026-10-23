from datasets import load_dataset


class DataLoader:

    DATASET_NAME = "KFUPM-JRCAI/arabic-generated-abstracts"

    def __init__(self):
        self.dataset = None

    def load_data(self):

        print("Loading dataset...")

        self.dataset = load_dataset(self.DATASET_NAME)

        print("Dataset loaded successfully!")

        return self.dataset