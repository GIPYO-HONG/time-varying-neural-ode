# Time-Varying Parameter Estimation of Infectious Disease Models Using Neural ODEs

This project estimates **time-varying parameters** in infectious disease models using Neural ODEs. Neural networks represent the parameters and are trained to minimize the error between ODE solutions and observed data.

## Project Structure

- `src/Example/`: Parameter estimation experiments using basic ODE examples
- `src/Epidemic/`: Epidemic modeling experiments using synthetic, COVID-19, and influenza data
- `src/HIV/`: Patient-specific HIV model parameter estimation using the ACTG315 dataset

The main libraries are JAX, Diffrax, Equinox, and Optax. NumPy, pandas, and Matplotlib are used for data processing and visualization.

## Running Experiments

Run the desired experiment from the project root.

```bash
# Example
python src/Epidemic/covid.py
```

The learning rate, number of training steps, and data period can be adjusted in each script's main block. Model weights and loss histories are saved under `weights/` in the corresponding experiment directory. Evaluation outputs, including predictions, estimated parameters, and loss plots, are saved under `results/<script_name>/` in the same directory.
