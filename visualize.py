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

df = pd.read_csv("./evals.csv")

#print(df.groupby(["mode", "model_type", "adjacency_threshhold", "tree_threshhold"])["macro_f1"].mean())

df = df[
    (df["model_type"] != "goemotions_syntax_interleaved")
    | ((df["tree_threshhold"] == 200) & (df["adjacency_threshhold"] == 12))
]

# Plot the plots

dev = df[df["mode"] == "dev"]

dev_interleaved_goemotions = dev[dev["model_type"] == "goemotions_syntax_interleaved"]
dev_modified_goemotions = dev[dev["model_type"] == "goemotions_modified"]
dev_goemotions = dev[dev["model_type"] == "goemotions"]

def show_plot(df: pd.DataFrame, title, color):
    df.set_index('epoch', inplace=True)
    df.groupby('seed')['macro_f1'].plot(legend=False)
    plt.title(title)
    plt.ylabel("Macro F1")
    plt.xlabel("Epoch")
    plt.ylim(0.3, 0.55)
    plt.xlim(1, 10)
    #plt.plot(df["epoch"], df["macro_f1"], color)
    plt.show()

#num_plots = 8
#fig, axs = plt.subplots(4, 2)

show_plot(dev_goemotions, "GoEmotions", "blue")
show_plot(dev_modified_goemotions, "Modified GoEmotions", "red")
show_plot(dev_interleaved_goemotions, "Interleaved GoEmotions", "green")
