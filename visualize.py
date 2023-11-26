import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max.columns", None)

p_df = pd.read_csv("./evals.csv")

dev = p_df[p_df["mode"] == "dev"]

dev_modified_goemotions = dev[dev["model_type"] != "goemotions"]
dev_goemotions = dev[dev["model_type"] == "goemotions"]

print(dev.groupby("model_type")["accuracy"].mean())

def show_plot(df: pd.DataFrame, title):
    df.groupby("run_id") \
      .plot(x="epoch", y=["macro_f1"])

    plt.title(title)
    plt.ylabel("Macro F1")
    plt.xlabel("Epoch")
    plt.ylim(0.3, 0.55)
    plt.xlim(1, 10)
    plt.show()

show_plot(dev_goemotions, "GoEmotions")
show_plot(dev_modified_goemotions, "Modified GoEmotions")
