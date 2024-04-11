import matplotlib.pyplot as plt
import json

data = json.load(open("result.json", "r"))

print(data.keys())
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Experiment 2 (mask by horizontal lines)', fontsize=20)

ax[0].plot(data["num_classes"], data["acc"])
ax[0].set_title("Accuracy")
ax[0].set_ylim([0, 1.1])

ax[1].plot(data["num_classes"], data["epochs"])
ax[1].set_title("Epochs")
plt.show()
