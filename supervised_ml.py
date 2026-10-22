from sklearn.linear_model import LinearRegression
hours =[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
score =[30,40,45,55,65,70,80,85,90,95]
model = LinearRegression()
model.fit(hours,score)
user_input = input("What is the number of hours studied?: ")
hours_predict= int(user_input)
prediction = model.predict([[hours_predict]])
print(f"estimated score: {round(prediction[0], 2)}")


import matplotlib.pyplot as plt 
hours =[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
score =[30,40,45,55,65,70,80,85,90,95]
plt.scatter(hours, score, color="blue")
plt.plot(hours, score, color="red",label="Regression line")
plt.xlabel("Hours studied")
plt.ylabel("Exam Scores")
plt.title("Relationship between hours studied and exam scores")
plt.legend()
plt.show()