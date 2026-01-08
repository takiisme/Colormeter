- Correction
  (Abstract class is not strictly necessary, since we don't have many inherited classes.)
  - methods:
    - train(df, *hyperparms)
    - test(df)
  - ~~CorrectionByScaling~~ DONE
  - CorrectionByModel
    (Following the sklearn design, set hyperparameters at init)
    - attributes:
      - space: 'rgb' or 'lab' 
      - method: 'matrix', 'channelwise', 'joint'
          - 'matrix' = $f(r, g, b) = A (r,g,b)^\top + b$
          - 'channelwise' = $f(r, g, b, ...) = (f_1(r, ...), f_2(g, ...), f_3(b, ...))$
          - 'joint' = $f(r, g, b, ...) = (f_1(r, g, b, ...), f_2(r, g, b, ...), f_3(r, g, b, ...))$
      - pose: true or false
      - degree: int (always use interactions)
      - reg_degree: float, regularization constant for higher order terms (L1 penalty for sparsity)
      - reg_pose: float, regularization constant for higher order terms (L1 penalty for sparsity)
      - boundary_penalty_factor: float, controls how strong we penalize predictions outside the valid range
    - methods:
      - train
        params:
        - df: full data frame
      - test
        params:
        - df: full data frame
      - other methods for building the design matrices
  

- TransformationToDaylight
  - methods:
    - train
    - test

