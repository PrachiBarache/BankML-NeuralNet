# Binary Classification using Neural Network for Bank Marketing Dataset

## Project Overview
This project implements neural network models for binary classification using bank marketing data to predict whether a client will subscribe to a term deposit. The implementation compares both PyTorch and MATLAB environments to analyze differences in performance, efficiency, and ease of implementation.

## Dataset and Initial Analysis

The project uses the Bank Marketing dataset from the UCI Machine Learning Repository ([link to dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)), which contains information about direct marketing campaigns of a Portuguese banking institution.

### Basic Statistics and Data Analysis
After loading the dataset of 41,188 records with 21 attributes, I performed an initial exploratory data analysis:

**Target Variable Distribution:**
- Yes (client subscribed): 4,640 (11.3%)
- No (client did not subscribe): 36,548 (88.7%)
- This shows a significant class imbalance that may affect model training

**Numerical Features Statistics:**
| Feature | Mean | Std. Dev. | Min | Max | Median |
|---------|------|-----------|-----|-----|--------|
| Age | 40.02 | 10.42 | 18 | 95 | 38 |
| Campaign | 2.57 | 2.77 | 1 | 63 | 2 |
| Previous | 0.17 | 0.49 | 0 | 7 | 0 |
| Emp.var.rate | 0.08 | 1.57 | -3.4 | 1.4 | 1.1 |
| Cons.price.idx | 93.58 | 0.59 | 92.2 | 94.8 | 93.8 |
| Cons.conf.idx | -40.50 | 4.63 | -50.8 | -26.9 | -41.8 |
| Euribor3m | 3.62 | 1.73 | 0.6 | 5.0 | 4.9 |
| Nr.employed | 5167.0 | 72.3 | 4964 | 5228 | 5191 |

**Categorical Features Analysis:**
- Job: Most common categories are "admin." (10,422), "blue-collar" (9,254), and "technician" (6,743)
- Marital: Most clients are married (24,928) followed by single (11,568)
- Education: Most common levels are "university.degree" (12,168) and "high.school" (9,515)
- Default: Very few clients have credit in default (1.2%)
- Month: Campaign contacts peak in May, June, July, and August
- Day_of_week: Fairly even distribution with slightly more contacts on Thursdays and Fridays

**Correlation Analysis:**
- Strong positive correlation (0.96) between euribor3m and emp.var.rate
- Moderate negative correlation (-0.75) between cons.conf.idx and cons.price.idx
- Duration of contact shows the strongest correlation with target variable (0.40)

These initial statistics informed data preprocessing decisions and feature engineering approaches.

## Methodology

### Data Preprocessing
- Handled categorical variables using one-hot encoding (9 variables transformed)
- Encoded unknown values as '0' to preserve information
- Applied label encoding to the target variable (Yes=1, No=0)
- Standardized numerical features using 'StandardScaler' to ensure equal scale importance
- Split data into 80% training and 20% testing sets with stratification to maintain class balance

### Neural Network Architecture
The implementation uses a feedforward neural network with:
- Input layer with dimensions matching the feature space (after one-hot encoding)
- Single hidden layer with ReLU activation function 
- Output layer with sigmoid activation for binary classification
- Cross-entropy loss function
- Stochastic Gradient Descent (SGD) optimizer

### Hyperparameter Tuning
Multiple configurations were tested to optimize model performance:
- Various hidden layer sizes (32, 64, 128)
- Different learning rates (0.001, 0.01, 0.1)
- Number of training epochs: 100

## Implementation Details

### PyTorch Implementation
```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

### MATLAB Implementation
```matlab
% Create and configure the network
net = fitnet(hiddenSize);
net.trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation
net.performFcn = 'crossentropy';
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.1;
```

## Comparative Analysis

### Performance Metrics

| Hidden Size | Learning Rate | Training Time (s) Python | Training Time (s) MATLAB | Test Accuracy Python | Test Accuracy MATLAB |
|-------------|---------------|--------------------------|--------------------------|----------------------|----------------------|
| 32          | 0.001         | 0.338868                 | 5.156093                 | 0.807302             | 0.818378             |
| 32          | 0.01          | 0.356874                 | 5.328066                 | 0.894635             | 0.918864             |
| 32          | 0.1           | 0.410693                 | 6.204838                 | 0.886993             | 0.919372             |
| 64          | 0.001         | 0.459042                 | 7.328966                 | 0.880282             | 0.918864             |
| 64          | 0.01          | 0.558832                 | 9.454639                 | 0.892082             | 0.918864             |
| 64          | 0.1           | 0.497526                 | 8.302518                 | 0.884681             | 0.919915             |
| 128         | 0.001         | 0.848839                 | 12.954138                | 0.886502             | 0.919372             |
| 128         | 0.01          | 0.839178                 | 13.356874                | 0.895849             | 0.919372             |
| 128         | 0.1           | 0.526759                 | 14.267626                | 0.891911             | 0.919915             |

### Comparative Visualizations

Below are improved visualizations specifically designed to highlight the differences between Python and MATLAB implementations:

![Performance Comparison and Training Speed Comparison]
(https://github.com/PrachiBarache/BankML-NeuralNet/blob/main/Mtlab_plots_op.png)
*Figure 1: Comparison of Python vs. MATLAB test accuracies across different hidden layer sizes and learning rates*

*Figure 2: Training speed comparison between Python and MATLAB implementations*

Key observations from these visualizations:
1. MATLAB consistently achieves 2-3% higher test accuracy across all configurations
2. Python training is significantly faster (5-15x) than MATLAB for equivalent model configurations
3. MATLAB shows more consistent performance across different hyperparameter settings
4. Python shows greater sensitivity to learning rate changes, especially with larger hidden layers
5. The performance gap between the two implementations narrows as hidden layer size increases

## Critical Evaluation of Results

### Performance Differences Analysis
- **Accuracy Gap**: MATLAB consistently outperforms Python by 2-3%. This likely stems from MATLAB's advanced implementation of Levenberg-Marquardt backpropagation, which is known for better convergence properties than standard SGD used in the Python implementation.

- **Training Speed**: Python demonstrates significantly faster training times (5-15x faster than MATLAB), making it more suitable for rapid prototyping and experimentation. This speed advantage increases proportionally with model complexity.

- **Hyperparameter Sensitivity**: Python implementation shows greater variability in performance across different hyperparameter settings. This indicates that Python requires more careful tuning but potentially allows for more fine-grained optimization.

- **Scaling Behavior**: As hidden layer size increases, Python's relative performance improves compared to MATLAB, suggesting better scaling for more complex architectures.

- **Consistency**: MATLAB shows remarkably consistent test accuracies (91.8-92.0%) regardless of hidden layer size, indicating its optimization algorithm effectively finds similar minima regardless of architecture complexity.

### Implementation-Specific Insights
- **Convergence Patterns**: MATLAB typically reaches optimal performance in fewer iterations due to the second-order optimization method, while Python's SGD approach requires more iterations but each iteration is faster.

- **Memory Usage**: Python maintains lower memory footprint, especially for larger models, making it more suitable for resource-constrained environments.

- **Code Complexity**: The Python implementation required more explicit coding for training loops and evaluation, while MATLAB provided higher-level abstractions at the cost of reduced flexibility.

## Lessons Learned from Language Comparison

The comparison between Python and MATLAB provided valuable insights into their respective strengths and limitations for neural network implementation:

1. **Development Trade-offs**: Python offers greater flexibility and control over implementation details but requires more code. MATLAB provides powerful high-level functions that reduce development time but limit fine-grained control.

2. **Performance Characteristics**: MATLAB's specialized algorithms provide superior accuracy but at the cost of computational efficiency. Python's implementation is significantly faster, enabling more extensive experimentation in the same timeframe.

3. **Learning Curve**: The Python implementation with PyTorch has a steeper initial learning curve but provides better transferability of skills to other deep learning frameworks. MATLAB's neural network toolbox offers easier entry for beginners but with more limited growth potential.

4. **Debugging and Visualization**: MATLAB offers superior built-in visualization tools for neural network analysis, while Python requires more custom code but provides greater customization options.

5. **Production Readiness**: Python code is more easily deployable to production environments and integrates better with modern data pipelines, while MATLAB excels in research and prototype development contexts.

6. **Optimization Techniques**: The significant performance gap highlights the importance of advanced optimization algorithms (like Levenberg-Marquardt) over simpler approaches (like SGD), especially for problems with relatively small datasets.

## Conclusion

This project successfully implemented and compared neural network models for bank marketing prediction across Python and MATLAB environments. The analysis reveals several important findings:

1. **Language-Performance Relationship**: MATLAB achieves higher accuracy (up to 92%) but with significantly longer training times, while Python offers competitive accuracy (up to 89.6%) with much faster training.

2. **Optimal Configuration**: The best performance in Python was achieved with a hidden layer size of 128 neurons and a learning rate of 0.01, while MATLAB performed consistently well across configurations.

3. **Development Efficiency**: Python required more code for implementation but offered greater flexibility and faster iteration cycles, making it potentially more suitable for complex real-world applications.

4. **Practical Applications**: Both implementations demonstrate the viability of neural networks for predicting bank marketing outcomes, with accuracy levels sufficient for practical applications in targeted marketing campaigns.

This comparison highlights that the choice between Python and MATLAB should be driven by specific project requirements: Python for speed, flexibility, and production deployment; MATLAB for ease of implementation, built-in visualization, and slightly higher accuracy when computational resources are not a limiting factor.

## Future Work

Based on the lessons learned, several directions for future work are promising:

1. **Algorithmic Improvements**: Implement Levenberg-Marquardt optimization in PyTorch to determine if the accuracy gap can be closed while maintaining Python's speed advantage.

2. **Architecture Exploration**: Experiment with multi-layer architectures to determine if deeper networks could improve Python's performance to match MATLAB's results.

3. **Feature Engineering**: Apply more sophisticated feature selection and transformation techniques based on the dataset's statistical properties identified in the initial analysis.

4. **Regularization Comparison**: Evaluate how different regularization techniques (dropout, L1/L2) affect the performance gap between the two implementations.

5. **Scale Testing**: Assess how both implementations perform on significantly larger datasets to identify potential scaling limitations.

6. **Framework Comparison**: Extend the comparison to include TensorFlow and scikit-learn implementations to provide a more comprehensive evaluation of available tools.
