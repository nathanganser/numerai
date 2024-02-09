import numpy as np
import pandas as pd
from scipy import stats


def neutralize(
  df, columns, neutralizers=None, proportion=1.0, era_col="era"
):
  if neutralizers is None:
      neutralizers = []
  unique_eras = df[era_col].unique()
  computed = []
  for u in unique_eras:
      df_era = df[df[era_col] == u]
      scores = df_era[columns].values
      scores2 = []
      for x in scores.T:
          x = pd.Series(x)
          x = (x.rank(method="first") - 0.5) / len(x.dropna())
          x = stats.norm.ppf(x)
          scores2.append(x)
      scores = np.array(scores2).T
      exposures = (
          df_era[neutralizers]
          .fillna(df_era[neutralizers].median())
          .fillna(0.5)
          .values
      )

      scores -= proportion * exposures.dot(
          np.linalg.pinv(exposures.astype(np.float32), rcond=1e-6).dot(
              scores.astype(np.float32)
          )
      )

      scores /= pd.DataFrame(scores).std(ddof=0, axis=0, skipna=True).values

      computed.append(scores)

  return pd.DataFrame(np.concatenate(computed), columns=columns, index=df.index)