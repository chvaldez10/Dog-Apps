# Import Libraries
import torch

# Function definitions and classes

# Main loop
def main_loop():
    """
    The main loop where the core logic of the script is executed.

    Returns:
    None
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

# Main entry point
if __name__ == "__main__":
    print("Starting the training")

    # Call the main loop function
    main_loop()

    print("Finish training")