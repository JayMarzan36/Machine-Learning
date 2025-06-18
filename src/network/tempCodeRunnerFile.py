    predictions = net.forward(X)
    # Plot results
    plot.plot(X, y, label="True Function")
    plot.plot(X, predictions, label="NN Prediction")
    plot.legend()
    plot.title("Neural Network Approximation")
    plot.grid(True)
    plot.show()