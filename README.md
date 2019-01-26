# model_enhanced

This module provides classes which extends machine learning Pipeline.

    [DataFrame (raw data)]
      |
      | processor (such as function including trained LabelBinariser)
      V
    [DataFrame (preprocessed)]
      |
      | model / pipeline
      V
    [Series]

- You can integrate a pre-process function to a ML pipeline.
- Given a binary classifier you can use an arbitrary threshold for making a prediction.




