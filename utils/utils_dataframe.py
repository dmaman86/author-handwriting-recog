import pandas as pd

def build_results_df(
    y_pred: list[int],
    y_true: list[int],
    confidences: list[float],
    class_names: dict[int, str] | None = None
) -> pd.DataFrame:
  """
    Build a DataFrame summarizing predictions, confidence, and correctness.
    Args:
        y_pred (list[int]): Predicted class indices.
        y_true (list[int]): True class indices.
        confidences (list[float]): Confidence of the predicted class.
        class_names (dict[int, str], optional): Mapping of class indices to names.
    Returns:
        pd.DataFrame: A DataFrame with columns: True Class, Predicted Class, 
        Confidence, Correct.
  """
  df = pd.DataFrame({
      "True Class": y_true,
      "Predicted Class": y_pred,
      "Confidence": confidences
  })
  df["Correct"] = df["True Class"] == df["Predicted Class"]

  if class_names is not None:
    df["True Class"] = df["True Class"].map(class_names)
    df["Predicted Class"] = df["Predicted Class"].map(class_names)

  return df