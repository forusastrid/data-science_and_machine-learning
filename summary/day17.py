#95

mglearn.plotsplot_logistic_regression_graph()

#96

mglearn.plots.plot_single_hidden_layer_graph()

#97

line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label="tanh")
plt.plot(line, np.maximum(line, 0), linestyle='--', label="relu")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")
plt.show()

#98

mglearn.plots.plot_two_hidden_layer_graph()
