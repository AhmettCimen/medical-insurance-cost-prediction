import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("data/medical_insurance_dataset.csv")

class Data:
    def __init__(self, dataframe):
        
    
        self.bmi = dataframe['bmi'].tolist() 
        
       
        self.cost = dataframe['insurance_cost'].tolist() 
        
data = Data(df)


base_m = 6
base_b = 6
step_size = 1e-3
iterations = 50000

def calculate_derivative_m_b(m, b, data):
    derivative_m=0
    derivative_b=0 
    
  
    for i in range(len(data.bmi)): 
        yReal = data.cost[i]
        x = data.bmi[i]
        length = len(data.bmi)
        yPredicted = m * x + b
        
        # https://prnt.sc/RcFyKYok1SSC
        derivative_m += (-2 * x * (yReal - yPredicted)) / length
        derivative_b += (-2 * (yReal - yPredicted)) / length
        
    return derivative_m, derivative_b

def update_m_b(m, b, derivative_m, derivative_b, step_size):
    # https://prnt.sc/C5LHNEhos5jG
    m = m - step_size * derivative_m 
    b = b - step_size * derivative_b
    return m, b

def calculate_Error(m, b, data):
    error = 0
    # https://prnt.sc/oFUp5eZB5tvK
    for i in range(len(data.bmi)): 
        yReal = data.cost[i]
        x = data.bmi[i]
        yPredicted = m * x + b
        error += (yPredicted - yReal) ** 2
    return error / len(data.bmi)

def train(data, base_m, base_b, step_size, iterations):
    m = base_m
    b = base_b
    error = 0
    for i in range(iterations):
        error = calculate_Error(m, b, data)
        if (i % 10000 == 0): # Çıktı sıklığını artırdım
            print(f"Iterations={i} Error={error:.2f}")
            
        derivative_m, derivative_b = calculate_derivative_m_b(m, b, data)
        m, b = update_m_b(m, b, derivative_m, derivative_b, step_size)
    
    show(m, b, data)

def show(m, b, data):
    plt.figure(figsize=(9, 6))
    

    for i in range(len(data.bmi)):
        plt.scatter(data.bmi[i], data.cost[i], color='black', s=5, marker='o', alpha=0.5)
        
    plt.xlabel("BMI (Vücut Kitle İndeksi)")
    plt.ylabel("Insurance Cost ($)")
    
  
    x = np.linspace(min(data.bmi), max(data.bmi), 100)
    y = m * x + b
    plt.plot(x, y, color='red', linewidth=2, label=f"y = {m:.2f}x + {b:.2f}")
    
    plt.title("Simple Linear Regression (BMI vs Cost)")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()
    
train(data, base_m, base_b, step_size, iterations)