import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def getMatrix():
	data = pd.read_csv("./diabetes.csv")

	df = pd.DataFrame(data,columns=['Pregnancies','Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])

	corrMatrix = df.corr()
	sn.heatmap(corrMatrix, annot=True)
	plt.show()

getMatrix()