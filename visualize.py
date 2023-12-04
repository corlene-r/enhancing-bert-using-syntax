#import matplotlib.pyplot as plt
#import pandas as pd
#
#pd.set_option("display.max.columns", None)
#
#p_df = pd.read_csv("./evals.csv")
#
#dev = p_df[p_df["mode"] == "dev"]
#
#dev_modified_goemotions = dev[dev["model_type"] != "goemotions"]
#dev_goemotions = dev[dev["model_type"] == "goemotions"]
#
#print(dev.groupby("model_type")["accuracy"].mean())
#
#def show_plot(df: pd.DataFrame, title):
#    df.groupby("run_id") \
#      .plot(x="epoch", y=["macro_f1"])
#
#    plt.title(title)
#    plt.ylabel("Macro F1")
#    plt.xlabel("Epoch")
#    plt.ylim(0.3, 0.55)
#    plt.xlim(1, 10)
#    plt.show()
#
#show_plot(dev_goemotions, "GoEmotions")
#show_plot(dev_modified_goemotions, "Modified GoEmotions")
#
#
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max.columns", None)

p_df = pd.read_csv("./evals.csv")

dev = p_df[p_df["mode"] == "dev"]

dev_modified_goemotions = dev[dev["model_type"] != "goemotions"]
dev_goemotions = dev[dev["model_type"] == "goemotions"]

print(dev.groupby("model_type")["accuracy"].mean())

def show_plot(df: pd.DataFrame, title, axs, axidx):

    groups = df.groupby("run_id") 
    for name, group in groups:
        axs[axidx].plot(x=group["epoch"], y=group["macro_f1"])
        axs[axidx].ylabel("Macro F1")
        axs[axidx].set_title(title)
        axs[axidx].xlabel("Epoch")
        axs[axidx].ylim(0.3, 0.55)
        axs[axidx].xlim(1, 10)
        axidx += 1


num_plots = 8
fig, axs = plt.subplots(4, 2)

show_plot(dev_goemotions, "GoEmotions", axs, 0)
show_plot(dev_modified_goemotions, "Modified GoEmotions", axs, 4)

plt.show()
