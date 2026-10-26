
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/history.csv")

# Accuracy graph
plt.figure()
plt.plot(df["epoch"], df["train_accuracy"],
         marker="o", label="Training")
plt.plot(df["epoch"], df["val_accuracy"],
         marker="o", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("results/accuracy.png")
plt.close()

# Loss graph
plt.figure()
plt.plot(df["epoch"], df["train_loss"],
         marker="o", label="Training")
plt.plot(df["epoch"], df["val_loss"],
         marker="o", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("results/loss.png")
plt.close()

print("Graphs saved to results/")
