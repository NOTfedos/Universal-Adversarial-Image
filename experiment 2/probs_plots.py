import matplotlib.pyplot as plt
import json

data = json.load(open("result.json", "r"))

print(data.keys())
print(len(data["probs"]))
fig, ax = plt.subplots(1, 4, figsize=(12, 4), layout='tight')
fig.suptitle('Experiment 2 (mask by horizontal lines) - probs of predictions', fontsize=20)

k = 0
for num_classes, probs_arr in zip(data["num_classes"][1::4], data["probs"][1::4]):
    # probs_arr = data["probs"][i]
    ax[k].hlines(y=sum(probs_arr)/len(probs_arr), xmin=0, xmax=num_classes, color='r', label=f"avg prob={round(sum(probs_arr)/len(probs_arr) ,2)}")
    ax[k].scatter(range(len(probs_arr)), probs_arr)
    ax[k].set_ylim([0, 1.1])
    ax[k].set_title(f"num_classes={num_classes}")
    ax[k].legend(loc='lower right')
    k += 1

fig.supylabel('Probability of prediction')
fig.supxlabel("Class index")
plt.show()
