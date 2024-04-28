# 🧬 Cart-pushing

> ☣ **Warning:** This project was created during studies for educational purposes only. It may contain non-optimal solutions.

### 📑 About

The aim of the task was to **optimize the cart-pushing job**, focusing on maximizing the total distance covered within a given time after subtracting the total effort. A discrete state model describing this problem is expressed as follows:

$$x_1(k+1)=x_2(k)$$
$$x_2(k+1)=2x_2(k)-x_1(k)+\frac{1}{N^2}u_k(k)$$

The following performance index is used as the criterion for control quality:

$$J=x_1(N)-\frac{1}{2N}\sum_{k=0}^{N-1}u^2(k)$$

The task was to be solved for parameters: `N=5,10,15,20,25,30,35,40,45`.

### 🧫 Implementation

> The application is written in **Python 3.11.6**, primarily using `numpy` (for matrix operations) and `matplotlib` (for plotting results), in PyCharm 2023.1 Professional Edition.

**Genetic algorithms** were applied to optimize the presented problem. The appropriate approach to the algorithm components was implemented:
- parent selection — **proportional method** (*roulette wheel*),
- replacement strategy — **partial reproduction strategy** by replacing the worst individuals (*elitism*),
- crossover method — **single-point crossover**,
- mutation type — **phenotypic mutation** (*uniform value change*).

Constants have been defined, presented, amongst others, in table below. The designed `GeneticAlgorithm` class includes methods for accomplishing the task, including the algorithm iteration loop, calculation of state variables and the control quality index, roulette method application, mutation and chosen crossover, plotting, and others.

|Parameter|Value|
|---------|-----|
|N Value|5, 10, 15, 20, 25, 30, 35, 40, 45|
|Population Size|70|
|Elite Count|10% of population|
|Mutation Probability|25%|
|Number of Generations|Up to 5000|

### 📈 Results

Using the presented genetic algorithm configuration, the optimization of the cart-pushing task was conducted for the given values of N. Using an optimal control quality index value found in the literature, a comparison was made of the quality of the maximization performed:

$$J^*=\frac{1}{3}-\frac{3N-1}{6N^2}-\frac{1}{2N^3}\sum_{k=0}^{N-1}k^2$$

Results for less complex cases, where fewer time moments were considered, successfully achieved **zero relative error**. However, for values of N greater than 30, a small error occurs. This is directly related to the complexity of the problem. However, it is worth noting that the trend for other cases suggests that increasing the number of generations would likely achieve zero error.

<p align="center">
  <img src="_readme-img/1-plot_results.png?raw=true" width="300" alt="Results">
</p>

The chart above presents **the fitness level of the best individuals** throughout the solution optimization process. Initially, there is a slight fitness of the individuals. However, it dynamically increases and reaches a set value. Subsequently, only minor changes occur, which increasingly approach the ideal solution. The above chart was plotted for 2500 generations, although the optimization process was conducted for 5000. They are not shown in the chart due to minor changes and the loss of chart aesthetics.
