import plotly.offline as py
import plotly.graph_objs as go

file = open("Accuracy.txt", 'r')

epoch = []
train_accuracy = []
test_accuracy =[]

for line in file:
    splitted_line = line.split()
    epoch.append(int(splitted_line[0]))
    train_accuracy.append(100.0 * float(splitted_line[1]))
    test_accuracy.append(100.0 * float(splitted_line[2]))

file.close()

train_plot = go.Scatter(
    x = epoch,
    y = train_accuracy,
    mode = "lines+markers",
    name = "Training accuracy"
)

test_plot = go.Scatter(
    x = epoch,
    y = test_accuracy,
    mode = "lines+markers",
    name = "Test accuracy"
)

layout = go.Layout(
    xaxis = dict(
        title = "Iteration",
        showgrid = False
    ),
    yaxis = dict(
        title = "Accuracy (%)",
        showgrid = False 
    )
)

data = [train_plot, test_plot]
py.plot({"data": data, "layout": layout})