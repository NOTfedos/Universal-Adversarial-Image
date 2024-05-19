import matplotlib.pyplot as plt
import json

data = json.load(open("result.json", "r"))

print(data.keys())
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Experiment 1 (mask all image by mask)', fontsize=20)

ax[0].plot(data["num_classes"], data["acc"])
ax[0].set_xlabel("Count of classes")
ax[0].set_title("Accuracy")
ax[0].set_ylim([0, 1.1])

ax[1].plot(data["num_classes"], data["epochs"])
ax[1].set_xlabel("Count of classes")
ax[1].set_title("Epochs elapsed")
plt.show()
