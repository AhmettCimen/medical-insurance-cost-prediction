import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random

from sklearn.metrics import mean_squared_error # MSE görebilmek için


import matplotlib.pyplot as plt # gösterim için
import numpy as np

# import data
df = pd.read_csv('data\medical_insurance_dataset.csv')


ordinal_map = {'low': 0, 'medium': 1, 'high': 2} # Exercise frequency'nin low medium high olması, sayısal olarak değer ifade ettiği için ordinal encoder kullanılır.
df['exercise_frequency'] = df['exercise_frequency'].map(ordinal_map)


categorical_cols = ['sex', 'smoker', 'region', 'chronic_disease'] #Cinsiyet , yaşanılan bölge gibi değerler sayısal olarak bir şey ifade etmediği için One-Hot encoder kullanılır.
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

print(f"\n\nInitial row count: {len(df)}")
df = df.dropna() # Datasetteki NaN'ları kaldırıyorum
print(f"Final row count after dropping NaNs: {len(df)}")


y = df['insurance_cost'] # Hedef değer          
X = df.drop('insurance_cost', axis=1) #Hedef değer hariç tüm columnlar, axis 1 olduğu için dikey dropluyor.

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42) # train ve test  data olarak %80 ve %20 olarak ayırıyoruz


model = LinearRegression() 
model.fit(X_train, y_train) 


test_predictions = model.predict(X_test)

comparison_df = pd.DataFrame({'Actual': y_test.values, 'Prediction': test_predictions})


random_patient_index = random.randint(0,len(X_test)-1)

test_sample_X = X_test[random_patient_index].reshape(1, -1)
test_sample_Y = y_test.iloc[random_patient_index]

prediction = model.predict(test_sample_X)

print(f"\nRandomly Selected Patient Features (From Test Dataset):\n{test_sample_X}")
print(f"\nModel Prediction: ${prediction[0]:.2f}   Actual Value: ${test_sample_Y:.2f}")
print(f"Difference: ${(prediction[0]-test_sample_Y):.2f}")
print(f"Percentage Error: %{(prediction[0]-test_sample_Y)/prediction[0]*100:.2f}")

mse = mean_squared_error(y_test, test_predictions)


print(f"\nRoot Mean Squared Error (RMSE): ${np.sqrt(mse):.2f} -> Model deviated by an average of {np.sqrt(mse):.0f} dollars. ")
print(f"MSE: {mse:.2f}")


def show_feature_effect_smart(selected_feature):

    plt.figure(figsize=(8, 6))

    
    if len(X[selected_feature].unique()) <= 10:
        
        temp_df = pd.DataFrame({
            'SelectedFeature': X[selected_feature],
            'Cost': y
        })
        means = temp_df.groupby('SelectedFeature')['Cost'].mean()
        
        bars = plt.bar(means.index.astype(str), means.values, color='skyblue', edgecolor='navy')
        
        plt.title(f"{selected_feature}")
        plt.xlabel(f"{selected_feature}")
        plt.ylabel("Average Cost")
        
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval -500, f"${yval:.0f}",ha='center')
        plt.grid(axis='y', linestyle='--', alpha=0.5)

    else:
        
        plt.scatter(X[selected_feature], y, color='blue', s=10, alpha=0.3, label='Actual Data')
        
        z = np.polyfit(X[selected_feature], y, 2)
        p = np.poly1d(z)
        x_range = np.linspace(X[selected_feature].min(), X[selected_feature].max(), 100)
        
        plt.plot(x_range, p(x_range), color='red', linewidth=2, label=f"Predicted")
        
        plt.title(f"{selected_feature} ")
        plt.xlabel(f"{selected_feature}")
        plt.ylabel("Insurance Cost")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()

for col in X.columns:
    show_feature_effect_smart(col)



#for i in range (len(X)):                       # eğer istenirse tüm featurelar ile Y ilişkisi görselleştirilir.
#    selected_feature=X.columns[i]
#    show_selected_feature(selected_feature)




def show_actual_vs_predicted():

    plt.figure(figsize=(9, 7))


    plt.scatter(y_test, test_predictions, color='red',s=9, alpha=0.8,edgecolors='darkred',linewidth=0.1, label='Model Predictions')


    min_val = min(y_test.min(), test_predictions.min())
    max_val = max(y_test.max(), test_predictions.max())


    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2, label='Ideal Prediction (y=x)')
    plt.xlabel("Actual Cost ($)")
    plt.ylabel("Predicted Cost ($)")
    plt.title("Actual vs. Predicted Values\n y=x Ideal Line")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

show_actual_vs_predicted()