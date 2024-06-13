import pandas as pd
import numpy as np



def seperate_df(df, partA, partB):
    if partA and partB:
        return df
    elif partA:
        df = df[df.iloc[:, 0] == 'A']
        return df
    elif partB:
        df = df[df.iloc[:, 0] == 'B']
        return df
    else:
        print("choose at least one region")

def prepare_df(file_path):
    df = pd.read_csv(file_path, skiprows=1)
    mean_values = df[['FICO Score', 'Savings Rate (%)', 'Credit History (months)']].mean(axis=1)
    median_values = mean_values.median()
    true_labels = df['Label'].values
    pred_labels = (mean_values > median_values).astype(int).values
    print("pred_labels: ",pred_labels)
    error_rate = np.mean(pred_labels != true_labels)
    print("true_labels: ",true_labels)
    print(error_rate)
    df["pred_label"] = pred_labels
    return df

def cal_df(df):
    true_labels = df['Label'].values
    pred_labels = df['pred_label'].values

    # Calculate False Positives (FP)
    FP = np.sum((pred_labels == 1) & (true_labels == 0))
    print("False Positives (FP):", FP)

    # Calculate False Negatives (FN)
    FN = np.sum((pred_labels == 0) & (true_labels == 1))
    print("False Negatives (FN):", FN)

    # Calculate True Positives (TP)
    TP = np.sum((pred_labels == 1) & (true_labels == 1))
    print("True Positives (TP):", TP)

    # Calculate True Negatives (TN)
    TN = np.sum((pred_labels == 0) & (true_labels == 0))
    print("True Negatives (TN):", TN)



    approved_num = np.sum(pred_labels)
    print("Num Approved: {}/{}".format(approved_num, len(pred_labels)))


    accuracy = np.mean(pred_labels == true_labels)
    print("Accuracy :", accuracy)

    # equality of FPR/FNR
    FPR = FP / (FP + TN)
    print("FPR is:", FPR)

    FNR = FN / (FN + TP)
    print("FNR is:", FNR)

    #Positive Predictive Value (PPV) is calculated as TP / (TP + FP)
    PPV = TP / (TP + FP)
    print("PPV is:", PPV)

    # Negative Predictive Value (NPV) is calculated as TN / (TN + FN)
    NPV = TN / (TN + FN)
    print("NPV is:", NPV)



if __name__ == "__main__":
    df = prepare_df("fairness_dataset.csv")
    df_a = seperate_df(df, True, False)
    df_b = seperate_df(df, False, True)

    cal_df(df_a)

    cal_df(df_b)

