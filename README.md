# cu-powermarket-admm
CUDA implementation of the alternating direction method of multipliers (ADMM) for power market simulation

# Background
For details, please see  
- Kraning et al. (2014). Dynamic Network Energy Management via Proximal Message Passing. Foundations and Trends in Optimization, 1(2):70-122.  
- Rintamäki, Tuomas (2015). Demand and wind power scenarios for predicting power prices in a transmission-constrained system

# Compilation
The code utilizes CUDA's dynamic parallelism. Run the following commands

```
nvcc -arch=sm_50 -dc main.cu -o main.o  
nvcc -arch=sm_50 main.o -lcudadevrt -o main
```

# Examples
The input directory includes data files from a real world power system. The power system is modelled with 12 nodes, 18 transmission lines, 12 loads, and 12 generators with piecewise linear cost curves (each curve consists of 19 pieces in total). The example simulates 624 timesteps.
