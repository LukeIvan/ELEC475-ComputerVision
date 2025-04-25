import csv
import re


log = """
2024-11-19T04:19:15.635247-0500 INFO Epoch [1/30] - Train Loss: 2.77176, Val Loss: 2.49046
2024-11-19T04:20:52.494771-0500 INFO Epoch [2/30] - Train Loss: 2.13792, Val Loss: 1.72731
2024-11-19T04:22:29.357511-0500 INFO Epoch [3/30] - Train Loss: 1.86719, Val Loss: 1.41687
2024-11-19T04:24:06.222340-0500 INFO Epoch [4/30] - Train Loss: 1.79143, Val Loss: 1.23246
2024-11-19T04:25:43.086149-0500 INFO Epoch [5/30] - Train Loss: 1.71638, Val Loss: 1.13606
2024-11-19T04:27:19.979951-0500 INFO Epoch [6/30] - Train Loss: 1.92122, Val Loss: 1.06924
2024-11-19T04:28:56.867541-0500 INFO Epoch [7/30] - Train Loss: 1.56249, Val Loss: 0.99736
2024-11-19T04:30:33.736470-0500 INFO Epoch [8/30] - Train Loss: 1.49112, Val Loss: 0.96546
2024-11-19T04:32:10.613781-0500 INFO Epoch [9/30] - Train Loss: 1.35002, Val Loss: 0.93380
2024-11-19T04:33:47.471713-0500 INFO Epoch [10/30] - Train Loss: 1.38334, Val Loss: 0.89799
2024-11-19T04:35:24.328182-0500 INFO Epoch [11/30] - Train Loss: 1.39174, Val Loss: 0.87812
2024-11-19T04:37:01.203041-0500 INFO Epoch [12/30] - Train Loss: 1.09206, Val Loss: 0.84929
2024-11-19T04:38:38.046520-0500 WARNING Validation loss did not improve. Patience counter: 4
2024-11-19T04:38:38.046877-0500 INFO Epoch [13/30] - Train Loss: 1.10978, Val Loss: 0.85300
2024-11-19T04:40:14.938356-0500 INFO Epoch [14/30] - Train Loss: 0.99268, Val Loss: 0.82195
2024-11-19T04:41:51.800194-0500 INFO Epoch [15/30] - Train Loss: 1.08313, Val Loss: 0.80683
2024-11-19T04:43:28.642293-0500 INFO Epoch [16/30] - Train Loss: 1.15901, Val Loss: 0.79644
2024-11-19T04:45:05.486770-0500 INFO Epoch [17/30] - Train Loss: 1.24665, Val Loss: 0.79489
2024-11-19T04:46:42.352897-0500 INFO Epoch [18/30] - Train Loss: 0.92969, Val Loss: 0.78839
2024-11-19T04:48:19.228735-0500 INFO Epoch [19/30] - Train Loss: 1.20163, Val Loss: 0.77745
2024-11-19T04:49:56.088165-0500 INFO Epoch [20/30] - Train Loss: 0.70369, Val Loss: 0.77329
2024-11-19T04:51:32.945682-0500 INFO Epoch [21/30] - Train Loss: 0.92022, Val Loss: 0.76519
2024-11-19T04:53:09.766709-0500 INFO Epoch [22/30] - Train Loss: 1.39811, Val Loss: 0.75850
2024-11-19T04:54:46.611062-0500 WARNING Validation loss did not improve. Patience counter: 4
2024-11-19T04:54:46.611438-0500 INFO Epoch [23/30] - Train Loss: 0.77884, Val Loss: 0.76697
2024-11-19T04:56:23.478699-0500 INFO Epoch [24/30] - Train Loss: 1.11547, Val Loss: 0.75620
2024-11-19T04:58:00.337535-0500 INFO Epoch [25/30] - Train Loss: 1.30270, Val Loss: 0.75000
2024-11-19T04:59:37.209379-0500 INFO Epoch [26/30] - Train Loss: 0.86025, Val Loss: 0.74133
2024-11-19T05:01:14.059775-0500 INFO Epoch [27/30] - Train Loss: 0.83426, Val Loss: 0.73564
2024-11-19T05:02:50.897194-0500 WARNING Validation loss did not improve. Patience counter: 4
2024-11-19T05:02:50.897580-0500 INFO Epoch [28/30] - Train Loss: 0.91585, Val Loss: 0.75267
2024-11-19T05:04:27.739005-0500 WARNING Validation loss did not improve. Patience counter: 3
2024-11-19T05:04:27.739445-0500 INFO Epoch [29/30] - Train Loss: 1.12006, Val Loss: 0.74786
2024-11-19T05:06:04.609309-0500 WARNING Validation loss did not improve. Patience counter: 2
2024-11-19T05:06:04.609708-0500 INFO Epoch [30/30] - Train Loss: 0.71349, Val Loss: 0.74064


"""

def save_loss_to_csv(log):
    # Define a regular expression pattern to extract the loss values
    pattern = r"Epoch $$(\d+)/\d+$$ - Train Loss: ([\d.]+), Val Loss: ([\d.]+)"

    # Initialize lists to store the epoch, train loss, and validation loss
    epochs = []
    train_losses = []
    val_losses = []

    # Iterate through the log and extract the loss values
    for line in log.split("\n"):
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    # Save the data to a CSV file
    with open("loss_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
        for i in range(len(epochs)):
            writer.writerow([epochs[i], train_losses[i], val_losses[i]])

    print("CSV file saved successfully!")

# Example usage
save_loss_to_csv(log)
